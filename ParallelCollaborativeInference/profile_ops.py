from typing import List, Dict
import numpy as np
import torch
from torch.autograd.profiler_util import FunctionEvent
from torch.autograd import DeviceType
from ._utils import iterate_tensor
import copy


NUM_PROFILE_OPS_START = 5 # number of extra ops introduced by profile at start
NUM_PROFILE_OPS_END = 0 # number of extra ops introduced by profile at end

class OpsStamp:
    def __init__(self, _id, name, cpu_start, cpu_end,
                  cuda_end, cuda_dur) -> None:
        self.id = _id
        self.name = name
        self.cpu_start = cpu_start
        self.cpu_end = cpu_end
        self.cuda_end = cuda_end
        self.cuda_dur = cuda_dur

    def __add__(self, obj):
        # Support computation to enable average of multiple records
        # assert isinstance(obj, OpsStamp)
        # assert self.id == obj.id
        # assert self.name == obj.name, f"{self.name} != {obj.name}"
        return OpsStamp(
            self.id, self.name,
            self.cpu_start + obj.cpu_start,
            self.cpu_end + obj.cpu_end,
            self.cuda_end + obj.cuda_end,
            self.cuda_dur + obj.cuda_dur
        )

    def __truediv__(self, num):
        num = float(num)
        return OpsStamp(
            self.id, self.name,
            self.cpu_start / num,
            self.cpu_end / num,
            self.cuda_end / num,
            self.cuda_dur / num
        )

    def __repr__(self) -> str:
        return f"{self.id} {self.name} cpu start {self.cpu_start} cpu end {self.cpu_end} cuda dur {self.cuda_dur} cuda end {self.cuda_end}"

    def merge(self, _next):
        return OpsStamp(_next.id, _next.name,
                        max(_next.cpu_start, self.cpu_start),
                        max(_next.cpu_end, self.cpu_end),
                        max(_next.cuda_end, self.cuda_end),
                        max(_next.cuda_dur, self.cuda_dur))


from .hook_tensor import OffloadProfile
async def profile_ops(profile_result: OffloadProfile, func, args, kwargs,
                 wait=2, warmup=2, active=3, log=print):
    """Possible issue with WSL: https://github.com/pytorch/pytorch/issues/99615"""
    assert active > 0
    profiler =  torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],

        # In this example with wait=1, warmup=1, active=2, repeat=1,
        # profiler will skip the first step/iteration,
        # start warming up on the second, record
        # the third and the forth iterations,
        # after which the trace will become available
        # and on_trace_ready (when set) is called;
        # the cycle repeats starting with the next step

        schedule=torch.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=1)
    )
    profiler.start()
    for _ in range(wait + warmup + active):
        ret = await func(*args, **kwargs)
        torch.cuda.synchronize()
        profiler.step()
    profiler.stop()
    events = profiler.profiler.function_events

    profile_steps = []

    id_event = {}
    for event in events:
        if event.name=="ProfilerStep*":
            profile_steps.append(event)
        id_event[event.id] = event


    cuda_corr_map: Dict[int, List[float]] = {}
    trace_start_us = profiler.profiler.kineto_results.trace_start_us()
    for kineto_event in profiler.profiler.kineto_results.events():
        corr_id = kineto_event.linked_correlation_id()
        if corr_id > 0 and kineto_event.device_type() == DeviceType.CUDA:
            if corr_id not in cuda_corr_map:
                cuda_corr_map[corr_id] = []
            rel_start_us = kineto_event.start_us() - trace_start_us
            rel_end_us = rel_start_us + kineto_event.duration_us()
            cuda_corr_map[corr_id].append(rel_end_us)

    for profile_step in profile_steps:
        ops_stamps = parse_function_events(profile_step, cuda_corr_map)
        merge_ops_stamp_with_offload_profile(ops_stamps, profile_result, log)
    last_cuda_end = 0.
    for p in profile_result.profile.values():
        if active > 1:
            p.ops_stamp /= active
        p.ops_stamp /= 1e6     # us to s
        p.ops_time = p.ops_stamp.cuda_end - last_cuda_end
        last_cuda_end = p.ops_stamp.cuda_end
    return ret


def parse_function_events(
        profile_step: FunctionEvent,
        cuda_corr_map: Dict[int, List[FunctionEvent]])->List[OpsStamp]:
    # result.events() has most of the events - PyTorch op-level and device-level events
    ops_stamps: List[OpsStamp] = []
    trace_start_us = profile_step.cpu_children[0].time_range.start
    last_cuda_end = [trace_start_us]
    # Merge cuda events to cpu events
    def get_last_cuda_stamp(event: FunctionEvent):
        cuda_end = max(event.time_range.end, last_cuda_end[0])
        if event.id in cuda_corr_map:
            cuda_end = max(cuda_end, *cuda_corr_map[event.id])
        for _event in event.cpu_children:
            _cuda_end = get_last_cuda_stamp(_event)
            cuda_end = max(cuda_end, _cuda_end)
        last_cuda_end[0] = cuda_end
        return cuda_end
    for event in profile_step.cpu_children:
        cpu_start_us = event.time_range.start - trace_start_us
        cpu_end_us = event.time_range.end - trace_start_us
        cuda_end_us = get_last_cuda_stamp(event) - trace_start_us
        cuda_dur = event.cuda_time
        ops_stamps.append(OpsStamp(
            len(ops_stamps),
            name=event.name.split(":")[-1],
            cpu_start=cpu_start_us,
            cpu_end = cpu_end_us,
            cuda_end=cuda_end_us,
            cuda_dur=cuda_dur
        ))
    return ops_stamps

def merge_ops_stamp_with_offload_profile(
        ops_stamps: List[OpsStamp], profile_result: OffloadProfile, log=print):
    ops_stamps_iter = iter(ops_stamps)

    def find_until(iterator, name):
        _next: OpsStamp = next(iterator)
        while not (_next.name == name or _next.name == name + "_" or
                   name[-1] == "*" and _next.name.startswith(name[:-1])):
            _next: OpsStamp = next(iterator)
        return _next

    index_num = [0]
    slice_num = [0]
    select_num = [0]
    last_stamp = None
    for p in profile_result.profile.values():
        if p.func_name == "_start":
            stamp = OpsStamp(p.idx, p.func_name, 0., 0., 0., 0.)
        elif p.func_name == "_end":
            stamp = OpsStamp(p.idx, p.func_name, 0., 0., 0., 0.)
            if last_stamp:
                stamp = OpsStamp(p.idx, p.func_name, last_stamp.cpu_end, last_stamp.cpu_end, last_stamp.cuda_end, 0.)
        elif p.func_name == "type":
            stamp = find_until(ops_stamps_iter, "to")
        elif p.func_name == "nonzero":
            stamp = find_until(ops_stamps_iter, "nonzero*")
        elif p.func_name == "interpolate":
            stamp = find_until(ops_stamps_iter, "upsample_nearest*")
        elif p.func_name.startswith(("int", "float", "double", "bool")):
            stamp = find_until(ops_stamps_iter, "to")
        elif p.func_name == "__setitem__":
            stamp = find_until(ops_stamps_iter, "copy_")
        elif p.func_name == "unbind":
            stamp = find_until(ops_stamps_iter, "unbind")
        elif p.func_name == "__getitem__":
            index_num[0] = len(p.input_shapes) - 1
            slice_num[0] = 0
            select_num[0] = 0
            def find_index_num(_):
                index_num[0] += 1
            iterate_tensor(p.func_args, find_index_num, torch.Tensor)
            def find_slice_num(s: slice):
                slice_num[0] += 1
            iterate_tensor(p.func_args, find_slice_num, slice)
            def find_select_num(_):
                select_num[0] += 1
            iterate_tensor(p.func_args, find_select_num, int)
            assert slice_num[0] + index_num[0] + select_num[0]
            _last_stamp = None
            while index_num[0] + slice_num[0] + select_num[0] > 0:
                try:
                    _next: OpsStamp = next(ops_stamps_iter)
                except StopIteration as e:
                    log(f"Cannot find cuda info for {p.idx}th op {p.func_name}")
                    raise e
                if _next.name == "slice":
                    slice_num[0] -= 1
                elif _next.name == "index":
                    index_num[0] -= 1
                elif _next.name == "select":
                    select_num[0] -= 1
                last_stamp = _next
            stamp = last_stamp
            assert last_stamp
        else:
            try:
                temp_iter = copy.deepcopy(ops_stamps_iter)
                stamp = find_until(temp_iter, p.func_name)
                ops_stamps_iter = temp_iter
            except StopIteration:
                log(f"Warning. {p.func_name} not present in ops_stamps.")
                
                if last_stamp:
                    stamp = OpsStamp(p.idx, p.func_name, last_stamp.cpu_end, last_stamp.cpu_end, last_stamp.cuda_end, 0.)
                else:
                    stamp = OpsStamp(p.idx, p.func_name, 0., 0., 0., 0.)
        # print(f"{p.idx} got {stamp}")
        last_stamp = stamp
        if p.ops_stamp is None:
            p.ops_stamp = stamp
        else:
            p.ops_stamp += stamp
