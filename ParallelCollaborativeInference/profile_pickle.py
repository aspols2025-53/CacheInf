import time
import pickle
import torch
import numpy as np
from .hook_tensor import OffloadProfile
from ._utils import iterate_tensor
from typing import List

def profile_pickle(profile_result: OffloadProfile, eval_num=20, average_times=3,
                   ploy_fit_deg=2, log=print):
    # ploy_fit_deg controls the degree in fitting curve.

    size = []
    shapes = []
    dtypes = []
    _dtype = [None]
    def get_dtype(t: torch.Tensor):
        _dtype[0] = t.dtype
    for p in profile_result.profile.values():
        if p.output_size == 0:
            continue
        _shapes = []
        _dtype[0] = None
        for shape in p.output_shapes:
            _shapes.append(shape)
        iterate_tensor(p.func_args, get_dtype)
        assert _dtype[0]
        dtypes.append(_dtype[0])
        size.append(p.output_size)
        shapes.append(_shapes)
    size_acend_idx = np.argsort(size)
    size = np.array(size)[size_acend_idx]
    shapes = [shapes[i] for i in size_acend_idx]
    dtypes = np.array(dtypes)[size_acend_idx]

    dump_dur = []
    load_dur = []
    test_sizes = []
    step_size = len(size_acend_idx) // eval_num
    for _size, _shapes, _dtype in zip(size[::step_size],
        shapes[::step_size], dtypes[::step_size]):
        _t: List[torch.Tensor] = []
        for _shape in _shapes:
            _t.append(torch.empty(_shape, dtype=_dtype, device="cuda:0"))
        for _ in range(average_times):
            stime = time.time()
            d = pickle.dumps([__t.cpu() for __t in _t])
            dump_dur.append(time.time() - stime)

            stime = time.time()
            loaded_t = [__t.cuda() for __t in pickle.loads(d)]
            load_dur.append(time.time() - stime)
            test_sizes.append(_size)
    test_sizes = np.array(test_sizes)
    load_fit = np.polyfit(test_sizes, load_dur, ploy_fit_deg)
    dump_fit = np.polyfit(test_sizes, dump_dur, ploy_fit_deg)
    size_to_loads_time = np.poly1d(load_fit)
    size_to_dumps_time = np.poly1d(dump_fit)
    log(f"pickle.dumps size to time poly: \n{size_to_dumps_time}")
    log(f"pickle.loads size to time poly: \n{size_to_loads_time}")

    # import matplotlib.pyplot as plt
    # plt.plot(test_sizes/1024/1024, load_dur,'*', label='original load dur')
    # plt.plot(test_sizes/1024/1024, size_to_loads_time(test_sizes),
    #                  'r',label='polyfit load dur')
    # plt.xlabel('Tensor size/M')
    # plt.ylabel('Load dur/s')
    # plt.legend(loc=4)  #指定legend的位置,读者可以自己help它的用法
    # plt.title('Polyfitting load dur')
    # plt.savefig('load_dur.png')

    # plt.figure()
    # plt.plot(test_sizes/1024/1024, dump_dur,'*', label='original dump dur')
    # plt.plot(test_sizes/1024/1024, size_to_dumps_time(test_sizes),
    #                  'r',label='polyfit dump dur')
    # plt.xlabel('Tensor size/M')
    # plt.ylabel('Dump dur/s')
    # plt.legend(loc=4)  #指定legend的位置,读者可以自己help它的用法
    # plt.title('Polyfitting dump dur')
    # plt.savefig('dump_dur.png')
    profile_result.size_to_loads_time = size_to_loads_time
    profile_result.size_to_dumps_time = size_to_dumps_time
    return size_to_loads_time, size_to_dumps_time
    
