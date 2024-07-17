import time
import torch
from typing import List, Union
from queue import Queue

from .hook_tensor import TorchOPProfile
from ._utils import EndOfExec, iterate_tensor

last_stamp = [None]

def test_hook_client(args: Union[torch.Tensor, List[torch.Tensor]], 
                     profile: TorchOPProfile=None, **kwargs):
    torch.cuda.synchronize()
    stamp = time.time()
    if last_stamp[0] is not None:
        diff = stamp - last_stamp[0]
    else:
        diff = 0.
    profile.ops_time = diff
    # print(profile)
    last_stamp[0] = stamp
    return args


def select_all_offload_hook_client(args: Union[torch.Tensor, List[torch.Tensor]],
                     send_queue: Queue=None,
                     offload=False, end=False,
                     profile: TorchOPProfile=None, **kwargs):
    # print(profile)
    if offload:
        num_output = profile.num_output
        if num_output is None:    # only one arg
            send_queue.put(args.as_subclass(torch.Tensor))
        elif num_output > 0:
            send = []
            iterate_tensor(args, lambda t: send.append(t))
            send_queue.put(send)
        raise EndOfExec
    return args
