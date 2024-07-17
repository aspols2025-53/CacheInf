# from multiprocessing import Process, Queue
from typing import List
from collections import deque   # FIFO list with fixed size
import atexit
import time
import socket
import pickle
import struct
import torch
from torch.multiprocessing import Process, Queue
ctx = torch.multiprocessing.get_context("spawn")
Queue = ctx.Queue
Process = ctx.Process

class SockLog:
    def __init__(self, prefix="") -> None:
        self.prefix = prefix
        self.send_num = 0
        self.recv_num = 0
        self.send_len = []
        self.recv_len = []
        self.send_took_record = []
        self.recv_took_record = []
        self.complete_info = None

    def record_send(self, size):
        self.send_num += 1
        self.send_len.append(size)

    def record_recv(self, size):
        self.recv_num += 1
        self.recv_len.append(size)

    def __repr__(self) -> str:
        if not self.complete:
            return self.prefix + " data collection not complete."
        send_size, recv_size, send_took, recv_took, bw = self.complete_info
        return self.prefix + f" send {send_size:.4f}MB ({self.send_num} items), recv {recv_size:.4f}MB ({self.recv_num} items), send took {send_took:.4f}s, recv took {recv_took:.4f}s, bandwidth {bw:.4f}MB/s"

    @property
    def complete(self):
        if self.complete_info:
            return True
        if self.send_record_complete and self.recv_record_complete:
            send_size = sum(self.send_len)/1024/1024
            recv_size = sum(self.recv_len)/1024/1024
            send_took = sum(self.send_took_record)
            recv_took = sum(self.recv_took_record)
            bw = (send_size + recv_size) / (send_took + recv_took + 1e-8)
            self.complete_info = [send_size, recv_size, send_took, recv_took, bw]
            return True
        return False

    @property
    def send_record_complete(self):
        return len(self.send_took_record) == self.send_num

    @property
    def recv_record_complete(self):
        return len(self.recv_took_record) == self.recv_num

    def record_send_took(self, dur):
        self.send_took_record.append(dur)

    def record_recv_took(self, dur):
        self.recv_took_record.append(dur)

def empty_queue(q: Queue):
    ret = []
    while not q.empty():
        ret.append(q.get())
    return ret

def send(inp_q: Queue, sock: socket.socket, ctrl_sock: socket.socket,
        dur_q: Queue, end_q: Queue):
    header_fmt = 'id' # [int, double]: [msize, send took]
    dur = 0.
    try:
        while end_q.empty():
            msg = inp_q.get()
            msg = struct.pack(header_fmt, len(msg), dur) + msg
            stime = time.time()
            sock.sendall(msg)
            assert len(ctrl_sock.recv(1)) == 1
            etime = time.time()
            dur = etime - stime
            dur_q.put(dur)
    except KeyboardInterrupt:
        end_q.put(None)

def recv(out_q: Queue, sock: socket.socket, ctrl_sock: socket.socket,
         dur_q: Queue, end_q: Queue):
    header_fmt = 'id' # [int, double, double]: [msize, send bw, send took]
    header_size = struct.calcsize(header_fmt)
    max_recv_size = 2 * 1024 * 1024

    buffer = bytearray()
    try:
        while end_q.empty():
            while len(buffer) < header_size:
                buffer += sock.recv(max_recv_size)
            msize, last_recv_took = struct.unpack(header_fmt,
                buffer[:header_size])  # Last send bandwidth from the opposite
            dur_q.put(last_recv_took)
            buffer = buffer[header_size:]
            while len(buffer) < msize:
                buffer += sock.recv(max_recv_size)
            ctrl_sock.send(b"0")    # Send one byte to confirm transmission
            out_q.put(buffer[:msize])
            buffer = buffer[msize:]
    except KeyboardInterrupt:
        end_q.put(None)
        out_q.put(None)


class TCPMessageStream:
    """Multiprocess version of TCPMessageStream
    """
    def __init__(self, sock: socket.socket, ctrl_sock: socket.socket, log=print) -> None:
        self.input_queue = Queue(maxsize=5)
        sock.settimeout(None)
        ctrl_sock.settimeout(None)
        self.output_queue = Queue(maxsize=5)
        self._send_took = Queue()
        self._recv_took = Queue()
        self.sock = sock
        self.ctrl_sock = ctrl_sock
        self.end_q = Queue()
        self.log = log
        self.processes = [
            Process(target=send, args=(self.input_queue, sock, ctrl_sock,
                                       self._send_took, self.end_q),
                    name="send"),
            Process(
                target=recv, args=(self.output_queue, sock, ctrl_sock,
                                   self._recv_took, self.end_q),
                name="recv")
        ]
        for p in self.processes:
            p.start()
        atexit.register(self.close)

        self.recording_log = SockLog("Init")
        self.last_recording_logs: List[SockLog] = []
        self.recorded_logs: List[SockLog] = deque(maxlen=10)

    def start_record_log(self, prefix=""):
        assert 0. == self._recv_took.get()   # remove first empty last recv took
        self._start_record_log(prefix)
        self.start_record_log = self._start_record_log

    def _start_record_log(self, prefix=""):
        """record log at the start of new inference
        """

        self.last_recording_logs.append(self.recording_log)
        self.recording_log = SockLog(prefix)

        send_took = self._send_took
        recv_took = self._recv_took
        while True:
            if send_took.empty() and recv_took.empty() or len(self.last_recording_logs) == 0:
                break
            last_recording_log = self.last_recording_logs[0]
            if not (last_recording_log.send_record_complete or send_took.empty()):
                last_recording_log.record_send_took(send_took.get())
            if not (last_recording_log.recv_record_complete or recv_took.empty()):
                last_recording_log.record_recv_took(recv_took.get())
            if last_recording_log.complete and last_recording_log not in self.recorded_logs:
                self.log(last_recording_log)
                self.recorded_logs.append(last_recording_log)
                self.last_recording_logs.pop(0)

    def close(self):
        self.input_queue.close()
        self.output_queue.close()
        self._send_took.close()
        self._recv_took.close()
        self.end_q.put(None)
        self.sock.close()
        self.ctrl_sock.close()
        for p in self.processes:
            p.kill()

    def send_obj(self, obj):
        msg = pickle.dumps(obj)
        self.send_msg(msg)

    def send_msg(self, msg):
        self.input_queue.put(msg)
        self.recording_log.record_send(len(msg))

    def recv_obj(self):
        msg = self.recv_msg()
        obj = pickle.loads(msg)
        return obj

    def recv_msg(self):
        msg = self.output_queue.get()
        self.recording_log.record_recv(len(msg))
        return msg

    @property
    def speed_stale(self):
        send_len = [log.send_len for log in self.recorded_logs]
        if len(send_len) == 0 or sum(send_len) / len(send_len) > 1024*500:
            return False
        return True

    @property
    def last_bandwidth(self):
        if len(self.recorded_logs) > 0:
            return self.recorded_logs[0].complete_info[-1]
        return 20.

    def get_bandwidth(self):
        if self.speed_stale:
            return 0
        return self.last_bandwidth


def test():
    from threading import Thread
    ip = "127.0.0.1"
    port = 12345
    ctrl_port = 12346
    def server():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((ip, port))
        sock.listen(1)

        ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ctrl_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ctrl_sock.bind((ip, ctrl_port))
        ctrl_sock.listen(1)

        client_sock, client_address1 = sock.accept()
        client_ctrl_sock, client_address2 = ctrl_sock.accept()

        true_sock = MPSocket(client_sock, client_ctrl_sock)
        obj = true_sock.recv()
        print("server received.")
        true_sock.send(obj)
        print("server send.")
        time.sleep(2)
        true_sock.close()
        print("server terminated.")
    Thread(target=server, daemon=True).start()
    time.sleep(2)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    ctrl_sock.connect((ip, ctrl_port))
    true_sock = MPSocket(sock, ctrl_sock)
    tensor = torch.randn([3, 4, 5, 6, 7, 10], device="cuda:0")
    true_sock.send(tensor)
    recv_tensor = true_sock.recv()
    print(true_sock.last_bandwidth)
    assert torch.allclose(tensor, recv_tensor)
    true_sock.close()

if __name__ == "__main__":
    test()
