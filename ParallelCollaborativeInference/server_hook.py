from typing import List, Union
from queue import Queue
import time
import torch

from .hook_tensor import TorchOPProfile
from ._utils import iterate_tensor


last_stamp = [None]

def test_hook_server(args: Union[torch.Tensor, List[torch.Tensor]],
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



def select_all_offload_hook_server(args: Union[torch.Tensor, List[torch.Tensor]],
                     recv_queue: Queue=None,
                     offload=False, end=False,
                     profile: TorchOPProfile=None, **kwargs):
    # print(profile)
    if offload: # offload in server means worker offloading to server
        num_output = profile.num_output
        if num_output is None:    # only one arg
            args = args.__class__(recv_queue.get())
            # args = torch.cat([args, recv_queue.get()], 0)
        elif num_output > 0:
            recved_tensors = recv_queue.get()
            idx = [0]
            # def concat(arg):
            #     ret = torch.cat([arg, recved_tensors[idx[0]]], 0)
            #     idx[0] += 1
            #     return ret
            # args = iterate_tensor(args, concat)
            def copy(arg):
                arg = arg.__class__(recved_tensors[idx[0]])
                idx[0] += 1
                return arg
            args = iterate_tensor(args, copy)
    return args