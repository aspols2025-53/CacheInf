import struct
from typing import List, Tuple
import socket
import time
import pickle
from threading import Thread
from queue import Queue
# from asyncio import Queue
# from asyncio import StreamReader, StreamWriter
from collections import deque
import torch
import torch.distributed as dist
import numpy as np


to_MB = 1./1024./1024.
bw_count_size = 1024 * 20    # 20KB
bw_count_size_MB = bw_count_size * to_MB
dtype_memory_size_dict = {
    torch.float64: 64/8,
    torch.double: 64/8,
    torch.float32: 32/8,
    torch.float: 32/8,
    torch.float16: 16/8,
    torch.half: 16/8,
    torch.int64: 64/8,
    torch.long: 64/8,
    torch.int32: 32/8,
    torch.int: 32/8,
    torch.int16: 16/8,
    torch.short: 16/6,
    torch.uint8: 8/8,
    torch.int8: 8/8,
}
def get_tensor_size(t: torch.Tensor):
    return np.prod(t.shape) * dtype_memory_size_dict[t.dtype]


class SockLog:
    def __init__(self, prefix="", send_num=1) -> None:
        self.prefix = prefix
        self.suffix = ""
        self.send_num = send_num
        self.send_len = 0
        self.send_took = 0.
        self.msg = self.prefix + " data collection not complete."

    def record_send(self, send_record):
        send_record: np.ndarray = np.array(send_record)
        self.send_len = send_record[..., 0].sum()
        self.send_took = send_record[..., 1].sum()

    def __repr__(self) -> str:
        return self.msg

    def compute_info(self):
        send_size = self.send_len * to_MB
        send_took = self.send_took
        bw = send_size / send_took
        self.info = [send_size, send_took, bw]
        self.msg = self.prefix + f"send took {send_took:.2e}s ({send_size:.4f}MB, {self.send_num} items), bandwidth {bw:.4f}MB/s" + self.suffix

    def fix_info(self):
        self.msg = self.prefix + f"send took {0.:.2e}s ({0.:.4f}MB, {self.send_num} items), bandwidth {0.:.4f}MB/s" + self.suffix


max_recv_size = 10 * 1024 * 1024    # 10M
max_send_trunck = 1024 * 1024 * 100 # 100M
send_notification = torch.zeros(1, dtype=torch.int8, device="cuda:0")
recv_notification = torch.zeros(1, dtype=torch.int8, device="cuda:0")
empty_preamble = torch.randn(size=[1024, 1024], dtype=torch.float32, device="cuda:0")
preamble_size = get_tensor_size(empty_preamble)

SEND = 0
RECV = 1
RECV_NEW_TENSOR = 2
SEND_NEW_TENSOR = 5
SEND_OBJ = 3
RECV_OBJ = 4
class TorchDistributedStream:
    def __init__(self, rank: int, opposite_rank: int, log=print) -> None:
        self.rank = rank
        self.opposite_rank = opposite_rank
        self.log = log
        self.temp_send_record: deque[Tuple[torch.cuda.Event, torch.cuda.Event, int]] = deque(maxlen=10)
        self.send_record = []  # Record collected when sending
        self.order_queue = Queue()
        self.recv_queue = Queue()
        self.current_bw = 40.

        self.recording_log: SockLog = None
        self.last_recording_logs: List[SockLog] = []
        self.latest_sock_log: SockLog = SockLog()
        self.latest_sock_log.info = [40.]  # Init bw with max bw 40MB/s
        Thread(target=self.main_work, name="main_work", daemon=True).start()

    def main_work(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # In a new cuda stream
            self.log("Sending & Recving in a new cuda stream...")
            while True:
                order, tensors = self.order_queue.get()
                if order == SEND:
                    tensor_args = [[[t.shape], {"dtype": t.dtype, "device": t.device}] for t in tensors]
                    dist.broadcast_object_list([tensor_args], self.rank)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    size = 0
                    start.record()
                    for tensor in tensors:
                        dist.send(tensor, self.opposite_rank, tag=1)
                        size += get_tensor_size(tensor)
                    dist.recv(send_notification, self.opposite_rank, tag=0)
                    end.record()
                    if size > bw_count_size:
                        self.temp_send_record.append([start, end, size])
                    else:
                        self.temp_send_record.append([start, end, 0])
                elif order == RECV:
                    tensor_args = [None]
                    dist.broadcast_object_list(tensor_args, self.opposite_rank)
                    tensor_args = tensor_args[0]

                    ret_t: List[torch.Tensor] = []
                    for _tensor_args in tensor_args:
                        args, kwargs = _tensor_args
                        tensor = torch.zeros(*args, **kwargs)
                        dist.recv(tensor, self.opposite_rank, tag=1)
                        ret_t.append(tensor)
                    dist.send(recv_notification, self.opposite_rank, tag=0)
                    self.recv_queue.put(ret_t)
                elif order == SEND_OBJ:
                    dist.broadcast_object_list(obj, src=self.rank)
                elif order == RECV_OBJ:
                    objs = [None]
                    dist.broadcast_object_list(objs, src=self.opposite_rank)
                    self.recv_queue.put(objs[0])
                else:
                    raise RuntimeError(f"Not supported order {order}")

    def send_obj(self, obj):
        self.order_queue.put([SEND_OBJ, [obj]])

    def recv_obj(self):
        self.order_queue.put([RECV_OBJ, None])
        return self.recv_queue.get()

    def send_tensor(self, tensors: List[torch.Tensor]):
        self.order_queue.put([SEND, tensors])

    def send_finished(self):
        if not self.temp_send_record or self.temp_send_record[-1][1].query():
            return True
        return False

    def wait_finished(self):
        if self.temp_send_record and not self.temp_send_record[-1][1].query():
            self.temp_send_record[-1][1].synchronize()

    def recv_tensor(self):
        self.order_queue.put([RECV, None])
        return self.recv_queue.get()

    def recv_new_tensor(self):
        self.order_queue.put([RECV, None])
        return self.recv_queue.get()

    def start_record_log(self, prefix="", send_num=1):
        """Mark the start of new sock transmission for an inference

        Args:
            prefix (str, optional): Log prefix. Defaults to "".
            num (int, optional): Expected number of messages to receive. Defaults to 1.
        """
        if self.recording_log:
            self.last_recording_logs.append(self.recording_log)
        self.recording_log = SockLog(prefix, send_num)
        finish_num = 0
        # Sort logs for socket activities of each inference
        while self.temp_send_record and self.temp_send_record[0][1].query():
            start, end, size = self.temp_send_record.popleft()
            self.send_record.append([size, start.elapsed_time(end)/1e3])

        for recording_log in self.last_recording_logs:
            _send_num = recording_log.send_num
            if len(self.send_record) >= _send_num:
                if _send_num > 0:
                    recording_log.record_send(self.send_record[:_send_num])
                    recording_log.compute_info()
                    if recording_log.info[-1] > 0:
                        self.latest_sock_log = recording_log
                    self.send_record = self.send_record[_send_num:]
                else:
                    recording_log.fix_info()
                self.log(recording_log)
                finish_num += 1
            else:
                break
        self.last_recording_logs = self.last_recording_logs[finish_num:]
        if finish_num == 0:
            self.record_staled = True
        else:
            self.record_staled = False

    def add_suffix_to_log(self, suffix=""):
        self.recording_log.suffix += "\n" + suffix

    @property
    def last_bandwidth(self):
        if self.temp_send_record:
            if not self.temp_send_record[-1][1].query():    # Last send not completed
                self.log("Warning: bandwidth staled.")
                return 0.   # Staled

            temp_send_records = []
            for temp_send_record in self.temp_send_record:
                start, end, size = temp_send_record
                temp_send_records.append([size, start.elapsed_time(end)/1e3])
            self.temp_send_record.clear()
            self.send_record += temp_send_records
            bw_record = np.array(temp_send_records).sum(0)
            bw = bw_record[0] / bw_record[1] * to_MB
        else:
            bw = self.latest_sock_log.info[-1]
        return bw

    def get_bandwidth(self):
        return self.last_bandwidth
