import struct
from typing import List, Tuple
import socket
import time
import pickle
import asyncio
import numpy as np
from asyncio import Queue
from asyncio import StreamReader, StreamWriter
from collections import deque


to_MB = 1./1024./1024.
bw_count_size = 1024 * 20    # 20KB
bw_count_size_MB = bw_count_size * to_MB
class SockLog:
    def __init__(self, prefix="", send_num=1, recv_num=1) -> None:
        self.prefix = prefix
        self.suffix = ""
        self.send_num = send_num
        self.recv_num = recv_num
        self.send_len = 0
        self.send_took = 0.
        self.recv_len = 0
        self.recv_took = 0.
        self.dumps_took = 0.
        self.loads_took = 0.
        self.msg = self.prefix + " data collection not complete."

    def record_send(self, send_record):
        send_record: np.ndarray = np.array(send_record)
        self.send_len = send_record[:, 0].sum()
        self.send_took = send_record[:, 1].sum()

    def record_recv(self, recv_record):
        recv_record: np.ndarray = np.array(recv_record)
        self.recv_len = recv_record[:, 0].sum()
        self.recv_took = recv_record[:, 1].sum()

    def record_pickle(self, dumps_took, loads_took):
        self.dumps_took = dumps_took
        self.loads_took = loads_took

    def __repr__(self) -> str:
        return self.msg

    def compute_info(self):
        send_size = self.send_len * to_MB
        recv_size = self.recv_len * to_MB
        send_took = self.send_took
        recv_took = self.recv_took
        if send_size + recv_size < bw_count_size_MB:
            bw = 0.
        else:
            bw = (send_size + recv_size) / (send_took + recv_took)
        self.info = [send_size, recv_size, send_took, recv_took, bw]
        self.msg = self.prefix + f"send took {send_took:.2e}s ({send_size:.4f}MB, {self.send_num} items), recv took {recv_took:.2e}s ({recv_size:.4f}MB, ({self.recv_num} items)), dumps took {self.dumps_took:.2e}s, loads took {self.loads_took:.2e}s total {sum([send_took, recv_took, self.dumps_took, self.loads_took]):.4f}s bandwidth {bw:.4f}MB/s" + self.suffix


max_recv_size = 10 * 1024 * 1024    # 10M
max_send_trunck = 1024 * 1024 * 100 # 100M
class AsyncTCPMessageStream:
    """TODO: Trying to use asyncio to reduce I/O latency, since we notice data exchange latency is significantly greater than pickling time plus transmission time. The reason could be latency from threading (single core CPU contention) or multiprocessing (data passing between process)).
    """
    def __init__(self, reader_writer: Tuple[StreamReader, StreamWriter],
                 ctrl_reader_writer: Tuple[StreamReader, StreamWriter],
                 log=print) -> None:
        self.log = log
        self.reader, self.writer = reader_writer
        self.ctrl_reader, self.ctrl_writer = ctrl_reader_writer
        self.writer.get_extra_info("socket").setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.writer.get_extra_info("socket").setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
        self.ctrl_writer.get_extra_info("socket").setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.ctrl_writer.get_extra_info("socket").setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
        # self.buffer = bytearray()
        self.header_fmt = 'Ld' # [long, double]: [msize, send took]
        self.header_size = struct.calcsize(self.header_fmt)
        self.last_send_took = 0.
        self.last_recv_size = 0
        self.pickle_dumps_took = 0.
        self.pickle_loads_took = 0.
        self.send_record = []
        self.recv_record = []
        self.temp_send_record = deque(maxlen=4) # Small record size to keep records fresh
        self.temp_recv_record = deque(maxlen=4)
        self.record_staled = False
        self.temp_send_record.append([0, 1e-6])
        self.temp_recv_record.append([0, 1e-6])
        self.current_bw = 0.


        self.recording_log = SockLog("Init: ", 100, 100)
        self.last_recording_logs: List[SockLog] = []
        self.latest_sock_log: SockLog = SockLog("Init", 100, 100)
        self.latest_sock_log.info = [40.]  # Init bw with max bw 40MB/s
        self.sending: bool = False
        self.send_queue = Queue()
        self.recv_queue = Queue()

        self.running_tasks: List[asyncio.Task] = []
        self.skip_recv = 0
        self.staled = False

    async def _send_msg(self, msg):
        stime = time.time()
        self.writer.write(msg)
        if len(msg) > bw_count_size:
            await self.ctrl_reader.read(1)
        return time.time() - stime

    async def send_msg(self, msg):
        msg = struct.pack(self.header_fmt, len(msg), self.last_send_took) + msg
        send_dur = await self._send_msg(msg)
        if (size := len(msg)) > bw_count_size:
            self.temp_send_record.append([size, send_dur])
        self.send_record.append([size, send_dur])
        self.last_send_took = send_dur

    async def send_obj(self, obj):
        stime = time.time()
        msg = pickle.dumps(obj)
        self.pickle_dumps_took += time.time() - stime
        await self.send_msg(msg)

    async def _recv_msg(self):
        header = await self.reader.readexactly(self.header_size)
        if not header:
            raise EOFError
        # Last send bandwidth from the opposite
        msize, last_recv_took = struct.unpack(self.header_fmt,header)
        if msize < max_recv_size:
            buffer = await self.reader.readexactly(msize)
        else:
            buffer = bytearray()
            while len(buffer) < msize:
                _buffer = await self.reader.readexactly(min(max_recv_size, msize - len(buffer)))
                if not _buffer:
                    raise EOFError
                buffer += _buffer
        if msize + self.header_size > bw_count_size:
            self.ctrl_writer.write(b"0") # Send one byte to confirm transmission
        return buffer, last_recv_took

    async def recv_msg(self):
        msg, last_recv_took = await self._recv_msg()
        if self.last_recv_size: # skip first recv took
            if self.last_recv_size > bw_count_size:
                self.temp_recv_record.append([self.last_recv_size, last_recv_took])
        self.recv_record.append([self.last_recv_size, last_recv_took])
        self.last_recv_size = len(msg) + self.header_size
        return msg

    async def recv_obj(self):
        msg = await self.recv_msg()
        stime = time.time()
        obj = pickle.loads(msg)
        self.pickle_loads_took += time.time() - stime
        return obj

    def fix_init_sock_log(self, prefix=""):
        self.recording_log.prefix = prefix
        self.recording_log.send_num = len(self.send_record)
        if self.last_recv_size:
            init_recv_num = len(self.recv_record) + 1
        else:
            init_recv_num = 0
        self.recording_log.recv_num = init_recv_num

    def start_record_log(self, prefix="", send_num=1, recv_num=1):
        """Mark the start of new sock transmission for an inference

        Args:
            prefix (str, optional): Log prefix. Defaults to "".
            num (int, optional): Expected number of messages to receive. Defaults to 1.
        """
        self.temp_send_record.clear()
        self.temp_recv_record.clear()
        self.last_recording_logs.append(self.recording_log)
        self.recording_log = SockLog(prefix, send_num, recv_num)
        finish_num = 0
        # Sort logs for socket activities of each inference
        for recording_log in self.last_recording_logs:
            _send_num, _recv_num = recording_log.send_num, recording_log.recv_num
            if len(self.send_record) >= _send_num and len(self.recv_record) >= _recv_num:
                recording_log.record_send(self.send_record[:_send_num])
                recording_log.record_recv(self.recv_record[:_recv_num])
                recording_log.record_pickle(self.pickle_dumps_took, self.pickle_loads_took)
                recording_log.compute_info()
                if recording_log.info[-1] > 0:
                    self.latest_sock_log = recording_log
                self.log(recording_log)
                self.send_record = self.send_record[_send_num:]
                self.recv_record = self.recv_record[_recv_num:]
                finish_num += 1
            else:
                break
        self.pickle_dumps_took = self.pickle_loads_took = 0.
        self.last_recording_logs = self.last_recording_logs[finish_num:]
        if finish_num == 0:
            self.record_staled = True
        else:
            self.record_staled = False

    def add_suffix_to_log(self, suffix=""):
        self.recording_log.suffix += "\n" + suffix

    def close(self):
        try:
            for task in self.running_tasks:
                task.cancel()
            self.writer.write_eof()
            self.writer.close()
        except RuntimeError:
            pass

    @property
    def last_bandwidth(self):
        if self.recording_log.recv_num - len(self.recv_record) > 1 or \
            self.recording_log.send_num > len(self.send_record):
            self.log("Warning: bandwidth staled.")
            return 0.   # Staled
        if len(self.temp_send_record) > 0 or len(self.temp_recv_record) > 0:
            bw_record = np.array(self.temp_send_record + self.temp_recv_record).sum(0)
            bw = bw_record[0] / bw_record[1] * to_MB
        else:
            bw = self.latest_sock_log.info[-1]
        return bw

    def get_bandwidth(self):
        return self.last_bandwidth

    def start_send_loop(self):
        async def _send_loop():
            self.log("Started send loop.")
            try:
                while True:
                    msg = await self.send_queue.get()
                    self.sending = True
                    await self.send_msg(msg)
                    self.sending = False
            except Exception as e:
                self.log(str(e))
                raise e
        self.running_tasks.append(asyncio.create_task(_send_loop()))

    def skip_next_recv(self):
        self.skip_recv += 1

    def start_recv_loop(self):
        async def _recv_loop():
            self.log("Started recv loop.")
            try:
                while True:
                    msg = await self.recv_msg()
                    if self.skip_recv:
                        self.skip_recv -= 1
                    else:
                        await self.recv_queue.put(msg)
            except EOFError as e:
                await self.recv_queue.put(e)
                self.close()
            except Exception as e:
                self.log(str(e))
                raise e
        self.running_tasks.append(asyncio.create_task(_recv_loop()))

    async def queued_send(self, obj):
        stime = time.time()
        msg = pickle.dumps(obj)
        self.pickle_dumps_took += time.time() - stime
        await self.send_queue.put(msg)

    async def queued_recv(self):
        ret = await self.recv_queue.get()
        if isinstance(ret, EOFError):
            raise ret
        stime = time.time()
        obj = pickle.loads(ret)
        self.pickle_loads_took += time.time() - stime
        return obj
