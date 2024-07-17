import time
from typing import List, Dict
from functools import partial

import numpy as np
from functools import partial
import torch
import torch.distributed as dist
from .hook_tensor import OffloadProfile, TorchOPProfile, InputSlot
from ._utils import iterate_tensor, log_dur, iterate_all_close
from .test_torch_dist import TorchDistributedStream

from .reorganize_input import (pad_shapes_for_conv, pad_dim_for_conv, evolve_shape, interpolate,
                              clip_along_w_slice, move_along_w_slice, recover_img)

def fill_slot(arg: torch.Tensor, slots_iter):
    for slot in next(slots_iter):
        slot.container[slot.index] = arg

def fill_slot_cuda(arg: torch.Tensor, slots_iter):
    for slot in next(slots_iter):
        slot.container[slot.index] = arg.cuda()

def fill_output_to_input_slots(origin_input, output_dict: Dict[int, List[InputSlot]]):
    slots_iter = iter(output_dict.values())
    def fill_slot(arg: torch.Tensor):
        for slot in next(slots_iter):
            slot.fill(arg)
    iterate_tensor(origin_input, fill_slot)

def empty_input_slots(input_slots: List[InputSlot]):
    for slot in input_slots:
        slot.empty()

def slice_output(origin_input, send_slice: List[slice], keep_slice: List[slice]):
    send_tensor = []    # Always send a list of tensor
    send_slice_iter = iter(send_slice)
    keep_slice_iter = iter(keep_slice)
    def slice_tensor(arg: torch.Tensor):
        send_tensor.append(arg[next(send_slice_iter)].cpu())
        return arg[next(keep_slice_iter)].contiguous()
    keep_tensor = iterate_tensor(origin_input, slice_tensor)
    return keep_tensor, send_tensor

def cat_output(origin_input, recv_tensor: List[torch.Tensor], keep_slice: List[slice], order: int=0, dim=-1):
    recv_tensor_iter = iter(recv_tensor)
    keep_slice_iter = iter(keep_slice)
    if order == 0:
        def cat_tensor(arg: torch.Tensor):
            return torch.cat([arg[next(keep_slice_iter)], next(recv_tensor_iter).cuda()], dim)
    else:
        def cat_tensor(arg: torch.Tensor):
            return torch.cat([next(recv_tensor_iter).cuda(), arg[next(keep_slice_iter)]], dim)
    return iterate_tensor(origin_input, cat_tensor)

def align_tensor_shapes(obj, local_dim, align_mode=1):
    origin_input = []
    iterate_tensor(obj, origin_input.append)
    shape_len = len(origin_input[0].shape)
    align_dim_len = min(t.shape[local_dim] for t in origin_input)
    if align_mode == 1:
        slices = [slice(None)] * local_dim + [slice(align_dim_len)] + [...]
    else:   # 2 at the server side
        slices = [slice(None)] * local_dim + [slice(-align_dim_len, None)] + [...]
    def _align_shapes(t: torch.Tensor):
        return t[slices]
    return iterate_tensor(obj, _align_shapes)

def get_slice_at_dim(dim_idx, start, end):
    return [slice(None)] * dim_idx + [slice(start, end)]

def compile_plan_to_static_exec(
    profile_result: OffloadProfile, plans: Dict[int, Dict[str, list]],
    sock: TorchDistributedStream, log=print, log_during_compile=print, merge=True):
    def plain_skip(p: TorchOPProfile):
        for slot in p.input_slots:
            slot.container[slot.index] = None
    def skip_with_recv(p: TorchOPProfile, tensor_args: List):
        with log_dur(log, prefix=f"op {p.idx} {p.func_name} recv"):
            recved_tensor = sock.recv_new_tensor(tensor_args)
        iterate_tensor(recved_tensor,
                    partial(fill_slot_cuda,
                            slots_iter=iter(p.output_idx_slots.values())))
        for slot in p.input_slots:
            slot.container[slot.index] = None
    def plain_exec(p: TorchOPProfile, align_shape):
        if align_shape:
            args, kwargs = align_tensor_shapes(p.func_args, p.local_dim, align_shape)
        else:
            args, kwargs = p.func_args
        intermediates: torch.Tensor = p.func(*args, **kwargs)
        iterate_tensor(intermediates,
                    partial(fill_slot,
                            slots_iter=iter(p.output_idx_slots.values())))
        for slot in p.input_slots:
            slot.container[slot.index] = None
    def merged_plain_exec(
        ps: List[TorchOPProfile], align_shapes: List[int]):
        with log_dur(log, prefix=f"op {ps[0].idx}-{ps[-1].idx} merged exec"):
            for p, align_shape in zip(ps, align_shapes):
                if align_shape:
                    args, kwargs = align_tensor_shapes(p.func_args, p.local_dim, align_shapes)
                else:
                    args, kwargs = p.func_args
                try:
                    intermediates: torch.Tensor = p.func(*args, **kwargs)
                except Exception as e:
                    print(p)
                iterate_tensor(intermediates,
                    partial(fill_slot, slots_iter=iter(p.output_idx_slots.values())))
                for slot in p.input_slots:
                    slot.container[slot.index] = None
    def exec_with_recv(p: TorchOPProfile, keep_slice, cat_order, cat_dim,
                       recv_tensor_args, align_shape: int):
        if align_shape:
            args, kwargs = align_tensor_shapes(p.func_args, p.local_dim, align_shape)
        else:
            args, kwargs = p.func_args
        intermediates: torch.Tensor = p.func(*args, **kwargs)
        with log_dur(log, prefix=f"op {p.idx} {p.func_name} recv"):
            recv_intermediates = sock.recv_new_tensor(recv_tensor_args)
            intermediates = cat_output(
                intermediates, recv_intermediates,
                keep_slice=keep_slice,
                order=cat_order, dim=cat_dim)
        iterate_tensor(intermediates,
                    partial(fill_slot,
                            slots_iter=iter(p.output_idx_slots.values())))
        for slot in p.input_slots:
            slot.container[slot.index] = None
    def exec_with_offload(p: TorchOPProfile, send_slice, keep_slice, align_shape: List[int]):
        if align_shape:
            args, kwargs = align_tensor_shapes(p.func_args, p.local_dim, align_shape)
        else:
            args, kwargs = p.func_args
        intermediates: torch.Tensor = p.func(*args, **kwargs)
        with log_dur(log, prefix=f"op {p.idx} {p.func_name} send"):
            intermediates, send_intermediates = slice_output(
                intermediates, send_slice=send_slice, keep_slice=keep_slice)
            sock.send_tensor(send_intermediates)
        iterate_tensor(intermediates,
                    partial(fill_slot,
                            slots_iter=iter(p.output_idx_slots.values())))
        for slot in p.input_slots:
            slot.container[slot.index] = None
    def _plan_to_static_exec(skip: list, offload: list, recv: list,
                send_slice: dict, send_keep_slice: dict, recv_keep_slice: dict,
                cat_dim: dict, cat_order: dict, recv_tensor_args: dict,
                align_shape: list, **kwargs):
        func_calls = []
        for p in profile_result.profile.values():
            idx = p.idx
            if skip[idx] and np.all(skip[p.input_from]) and not recv[idx]:
                continue    # Ignore completely skipped op

            if skip[idx]:
                if recv[idx]:
                    func_calls.append([skip_with_recv, [p, recv_tensor_args[idx]]])
                else:
                    func_calls.append([plain_skip, [p]])
            elif offload[idx]:
                func_calls.append([
                    exec_with_offload, [p, send_slice[idx], send_keep_slice[idx], align_shape[idx]]])
            elif recv[idx]:
                func_calls.append([
                    exec_with_recv, [p, recv_keep_slice[idx],
                                     cat_order[idx], cat_dim[idx],
                                     recv_tensor_args[idx], align_shape[idx]]])
            else:
                func_calls.append([plain_exec, [p, align_shape[idx]]])
        merged_op = []
        merged_align_shape = []
        ret_func_calls = []
        in_plain_exec = False
        for func_call, args in func_calls:
            if func_call is plain_exec and merge:
                in_plain_exec = True
                merged_op.append(args[0])
                merged_align_shape.append(args[1])
            else:
                if in_plain_exec:
                    if len(merged_op) > 3:
                        ret_func_calls.append([merged_plain_exec, [merged_op, merged_align_shape]])
                    else:
                        for op, _align_shape in zip(merged_op, merged_align_shape):
                            ret_func_calls.append([plain_exec, [op, _align_shape]])
                    merged_op = []
                    merged_align_shape = []
                ret_func_calls.append([func_call, args])
                in_plain_exec = False
        if len(merged_op) > 0:
            if len(merged_op) > 3:
                ret_func_calls.append([merged_plain_exec, [merged_op, merged_align_shape]])
            else:
                for op, _align_shape in zip(merged_op, merged_align_shape):
                    ret_func_calls.append([plain_exec, [op, _align_shape]])
        return ret_func_calls
    for bw, plan in plans.items():
        profile_result.exec_plan[bw] = _plan_to_static_exec(**plan)
        to_offload = np.nonzero(plan["offload"])
        to_recv = np.nonzero(plan["recv"])
        est_time = plan["est_time"]
        log_during_compile(f"bw {bw}MB/s offload at {to_offload[0].tolist()} recv at {to_recv[0].tolist()} est time {est_time:.4f}s.")


from .img_matcher import ImageMatcher
from .reorganize_input import slice_to_indices_grid, move_along_w_slice, slices_to_final_len
def compile_plan_to_static_exec_cached(
    profile_result: OffloadProfile, plans: Dict[int, Dict[str, list]],
    sock: TorchDistributedStream, log=print, log_during_compile=print, role="client", offload_method="cached", matcher: ImageMatcher=None):
    if role == "client":
        cat_order = 0
    else:
        cat_order = 1
    reorganized_shape: List[np.ndarray] = [None]
    reorganize_info: dict = {}

    def send_reorg_info():  # At the start of exec, send reorg info at client
        reorganized_info.clear()
        reorganized_info.update(matcher.reorganize_info)
        reorganized_shape[0] = reorganize_info["reorg_shape"]
        sock.send_obj(reorganized_info)

    def recv_reorg_info():  # At the start of exec, send reorg info at server
        reorganize_info.clear()
        reorganize_info.update(sock.recv_obj())
        reorganized_shape[0] = reorganize_info["reorg_shape"]
    # TODO compute selective indices for view, cat, select; manage the interpolation of these indices to manage cache

    def plain_skip(p: TorchOPProfile):
        for slot in p.input_slots:
            slot.container[slot.index] = None
    def skip_with_recv(p: TorchOPProfile):
        with log_dur(log, prefix=f"op {p.idx} {p.func_name} recv"):
            recved_tensor = sock.recv_new_tensor(tensor_args)
        iterate_tensor(recved_tensor, partial(fill_slot_cuda, slots_iter=iter(p.output_idx_slots.values())))
        for slot in p.input_slots:
            slot.container[slot.index] = None
    def plain_exec(p: TorchOPProfile):
        args, kwargs = p.func_args
        intermediates: torch.Tensor = p.func(*args, **kwargs)
        iterate_tensor(intermediates, partial(fill_slot, slots_iter=iter(p.output_idx_slots.values())))
        for slot in p.input_slots:
            slot.container[slot.index] = None
    def plain_exec_cached(p: TorchOPProfile, cache: bool):
        args, kwargs = p.func_args
        intermediates: torch.Tensor = p.func(*args, **kwargs)
        if cache:
            with log_dur(log, prefix=f"op {p.idx} {p.func_name} merge cache"):
                assert p.cached_output is not None, \
                    f"p.cached_output is None {p.cached_output is None}"
                cache = interpolate(p.cached_output,
                                    reorganize_info["M"], reorganize_info["h"], reorganize_info["w"])
                intermediates = recover_img(intermediates, cache, formula=p.formula, **reorganize_info)
        iterate_tensor(intermediates, partial(fill_slot, slots_iter=iter(p.output_idx_slots.values())))
        p.cached_output = intermediates
        for slot in p.input_slots:
            slot.container[slot.index] = None
    def merged_plain_exec(ps: List[TorchOPProfile]):
        with log_dur(log, prefix=f"op {ps[0].idx}-{ps[-1].idx} merged exec"):
            for p in ps:
                args, kwargs = p.func_args
                try:    # TODO remove at running
                    intermediates: torch.Tensor = p.func(*args, **kwargs)
                except Exception as e:
                    print(p)
                    raise e
                iterate_tensor(intermediates, partial(fill_slot, slots_iter=iter(p.output_idx_slots.values())))
                for slot in p.input_slots:
                    slot.container[slot.index] = None
    def exec_with_recv(p: TorchOPProfile, current_x: float, next_x: float, formula, cache: bool):
        args, kwargs = p.func_args
        intermediates: torch.Tensor = p.func(*args, **kwargs)
        with log_dur(log, prefix=f"op {p.idx} {p.func_name} recv"):
            recv_intermediates = sock.recv_new_tensor()
            slice_dim = p.local_dim
            shape = evolve_shape(reorganized_shape[0], p.formula)
            keep_dim_len = int(np.around(shape[slice_dim] * current_x))
            if cat_order == 0:  # client
                keep_slices = get_slice_at_dim(slice_dim, 0, keep_dim_len)
                intermediates = intermediates[keep_slices]
                intermediates = torch.cat([intermediates, recv_intermediates[0]], slice_dim)
            else:
                keep_slices = get_slice_at_dim(slice_dim, -keep_dim_len, None)
                intermediates = intermediates[keep_slices]
                intermediates = torch.cat([recv_intermediates[0], intermediates], slice_dim)
        if cache:
            with log_dur(log, prefix=f"op {p.idx} {p.func_name} recv and merge cache"):
                assert next_x == 1. and p.cached_output is not None, \
                    f"next_x {next_x} p.cached_output is None {p.cached_output is None}"
                cache = interpolate(p.cached_output,
                                    reorganize_info["M"], reorganize_info["h"], reorganize_info["w"])
                intermediates = recover_img(intermediates, cache, formula=p.formula, **reorganize_info)
                p.cached_output = intermediates
        iterate_tensor(intermediates, partial(fill_slot, slots_iter=iter(p.output_idx_slots.values())))
        for slot in p.input_slots:
            slot.container[slot.index] = None
    def exec_with_recv_cache(p: TorchOPProfile, current_x: float, formula, cache: bool):
        args, kwargs = p.func_args
        intermediates: torch.Tensor = p.func(*args, **kwargs)
        with log_dur(log, prefix=f"op {p.idx} {p.func_name} recv and merge cache"):
            recved_cache = sock.recv_new_tensor()
            shape = evolve_shape(reorganized_shape[0], p.formula)
            slice_dim = p.local_dim
            dim_len = shape[slice_dim]
            if current_x < 1.:  # only partial intermediates, merge with cache first 
                valid_len = int(np.around(dim_len * current_x))
                if cat_order == 0: 
                    valid_slices = get_slice_at_dim(slice_dim, 0, valid_len)
                    # clip reorg bboxes
                    slice_orig, slice_orig_valid, slice_reorg = clip_along_w_slice(
                        valid_len, **reorganize_info)
                else:
                    valid_slices = get_slice_at_dim(slice_dim, -valid_len, None)
                    slice_orig, slice_orig_valid, slice_reorg = move_along_w_slice(
                        dim_len - valid_len, **reorganize_info)
                intermediates = intermediates[valid_slices]
                intermediates = recover_img(intermediates, recved_cache, formula=p.formula,
                                            **reorganize_info)
            else:
                intermediates = recover_img(intermediates, recved_cache, formula=p.formula,
                                            **reorganize_info)
        p.cached_output = intermediates
        iterate_tensor(intermediates, partial(fill_slot, slots_iter=iter(p.output_idx_slots.values())))
        for slot in p.input_slots:
            slot.container[slot.index] = None
    def exec_with_offload(p: TorchOPProfile, current_x: float, next_x:float,
                          offset: int, recv_formula):
        # formula is the formula of the recv op idx
        args, kwargs = p.func_args
        intermediates: torch.Tensor = p.func(*args, **kwargs)
        with log_dur(log, prefix=f"op {p.idx} {p.func_name} send"):
            if next_x > 0.: # should keep offset for both remain local intermediates and send intermediates
                # the reorganized input's shape after computation
                shape = evolve_shape(reorganized_shape[0], p.formula)
                slice_dim = p.local_dim
                dim_len = shape[slice_dim]
                diff = current_x - next_x
                send_len = pad_dim_for_conv(int(np.around(dim_len * diff)) + offset, recv_formula)
                send_start = int(np.around(dim_len*next_x))
                keep_dim_len = pad_dim_for_conv(send_start + offset, recv_formula)
                if cat_order == 0:  # client
                    keep_slices = get_slice_at_dim(slice_dim, 0, keep_dim_len)
                    send_slices = get_slice_at_dim(slice_dim, send_start, send_start+send_len)
                else:
                    raise RuntimeError("Server should not send it partial intermediates back")
                    # keep_slices = get_slice_at_dim(slice_dim, -keep_dim_len, 0)
                    # send_slices = get_slice_at_dim(slice_dim, -send_start, -send_start + send_len)
                send_intermediates = intermediates[send_slices]
                intermediates = intermediates[keep_slices]
                sock.send_tensor([send_intermediates])
                iterate_tensor(intermediates, partial(
                    fill_slot, slots_iter=iter(p.output_idx_slots.values())))
            elif current_x < 1.:
                shape = evolve_shape(reorganized_shape[0], p.formula)
                slice_dim = p.local_dim
                valid_dim_len = int(np.around(shape[slice_dim]*next_x))
                if cat_order == 0:
                    send_slices = get_slice_at_dim(slice_dim, 0, valid_dim_len)
                else:
                    send_slices = get_slice_at_dim(slice_dim, -valid_dim_len, None)
                sock.send_tensor([intermediates[send_slices]])
            else:   # current x: 1., next x: 0.
                sock.send_tensor([intermediates])
        for slot in p.input_slots:
            slot.container[slot.index] = None
    def skip_with_offload_cache(p: TorchOPProfile):
        with log_dur(log, prefix=f"op {p.idx} {p.func_name} skip and offload cache"):
            assert p.cached_output
            cache = interpolate(p.cached_output,
                                reorganize_info["M"], reorganize_info["h"], reorganize_info["w"])
            sock.send_tensor([cache])
        for slot in p.input_slots:
            slot.container[slot.index] = None
    def exec_with_offload_cache(p: TorchOPProfile, current_x: float, next_x:float):
        # send whole output of p 
        args, kwargs = p.func_args
        intermediates: torch.Tensor = p.func(*args, **kwargs)
        with log_dur(log, prefix=f"op {p.idx} {p.func_name} merge and send cache"):
            shape = evolve_shape(reorganized_shape[0], p.formula)
            slice_dim = p.local_dim
            dim_len = shape[slice_dim]
            assert p.cached_output is not None
            # interpolate old cache
            cache = interpolate(p.cached_output,
                                reorganize_info["M"], reorganize_info["h"], reorganize_info["w"])
            if current_x < 1.:  # only partial intermediates, merge with cache first 
                valid_len = int(np.around(dim_len * current_x))
                if cat_order == 0: 
                    valid_slices = get_slice_at_dim(slice_dim, 0, valid_len)
                    # clip reorg bboxes
                    slice_orig, slice_orig_valid, slice_reorg = clip_along_w_slice(valid_len, **reorganize_info)
                else:
                    valid_slices = get_slice_at_dim(slice_dim, -valid_len, None)
                    slice_orig, slice_orig_valid, slice_reorg = move_along_w_slice(dim_len - valid_len, **reorganize_info)
                intermediates = intermediates[valid_slices]
                filled_cache = recover_img(intermediates, cache, formula=p.formula, **reorganize_info)
            else:
                filled_cache = recover_img(intermediates, cache, formula=p.formula, **reorganize_info)
            p.cached_output = filled_cache
            sock.send_tensor([filled_cache])
        for slot in p.input_slots:
            slot.container[slot.index] = None
    def _plan_to_static_exec(x: np.ndarray, next_x: np.ndarray, offload: np.ndarray, recv: np.ndarray,
                             skip: np.ndarray, cache_location: int, cached_op: list, use_cache: bool, offset: np.ndarray, formula: List[tuple], **_):
        func_calls = []
        local_cache = False
        if role == "client" and cache_location == 1 or role == "server" and cache_location == 0:
            local_cache = True
        if use_cache and np.any(offload):
            if role == "client":
                func_calls.append([send_reorg_info, []])
            else:
                func_calls.append([recv_reorg_info, []])
        for i, p in enumerate(profile_result.profile.values()):
            if skip[i]:
                if i in cached_op and use_cache and local_cache and i > 0:
                    func_calls.append([skip_with_offload_cache, [p]])
                elif recv[i]:
                    func_calls.append([skip_with_recv, [p]])
            elif offload[i]:
                if i in cached_op and use_cache and local_cache and i > 0:
                    func_calls.append([exec_with_offload_cache, [p, x[i], next_x[i]]])
                else:
                    func_calls.append([exec_with_offload, [p, x[i], next_x[i], offset[i], formula[i]]])
            elif recv[i]:
                if i in cached_op and use_cache and not local_cache and i > 0:
                    func_calls.append([exec_with_recv_cache, [p, x[i], formula[i], use_cache]])
                else:
                    recv_and_merge_cache = i > 0 and i in cached_op and use_cache
                    func_calls.append([exec_with_recv, [p, x[i], next_x[i], formula[i],
                                                       recv_and_merge_cache]])
            elif i in cached_op:
                func_calls.append([plain_exec_cached, [p, use_cache]])
            else:
                func_calls.append([plain_exec, [p]])
        merged_op = []
        ret_func_calls = []
        in_plain_exec = False
        for func_call, args in func_calls:
            if func_call is plain_exec:
                in_plain_exec = True
                merged_op.append(args[0])
            else:
                if in_plain_exec:
                    if len(merged_op) > 3:
                        ret_func_calls.append([merged_plain_exec, [merged_op]])
                    else:
                        for op in merged_op:
                            ret_func_calls.append([plain_exec, [op]])
                    merged_op = []
                ret_func_calls.append([func_call, args])
                in_plain_exec = False
        if len(merged_op) > 0:
            if len(merged_op) > 3:
                ret_func_calls.append([merged_plain_exec, [merged_op]])
            else:
                for op in zip(merged_op):
                    ret_func_calls.append([plain_exec, [op]])
        return ret_func_calls
    for key, plan in plans.items():
        profile_result.exec_plan[key] = _plan_to_static_exec(**plan)
        to_offload = np.nonzero(plan["offload"])
        to_recv = np.nonzero(plan["recv"])
        est_time = plan["est_time"]
        log_during_compile(f"key {key} offload at {to_offload[0].tolist()} recv at {to_recv[0].tolist()} use cache {plan['use_cache']} est time {est_time:.4f}s.")
    cached_ops = plan["cached_op"]
    last_cached_op = max(cached_ops.keys())
    profile_result.profile[0].cached_offset = profile_result.profile[last_cached_op].offset
    profile_result.profile[0].cached_formula = profile_result.profile[last_cached_op].formula


@torch.no_grad()
def random_exec_compiled(profile_result: OffloadProfile, key):
    # list(map(lambda x: x[0](*x[1]), profile_result.exec_plan[bw]))
    for exec_func, args in profile_result.exec_plan[key]:
        exec_func(*args)
    torch.cuda.default_stream().synchronize()   # Synchronize only on default stream;
                                                # Non default stream is handling preamble
    return profile_result.ret_store


def local_random_exec_cuda_profile(profile_result: OffloadProfile, warm_up=1, repeat=2):
    torch.cuda.synchronize()
    # Start exec
    events: List[List[torch.cuda.Event]] = []
    for i in range(warm_up + repeat):
        _events = []
        for p in profile_result.profile.values():
            args, kwargs = p.func_args
            intermediates: torch.Tensor = p.func(*args, **kwargs)

            fill_output_to_input_slots(intermediates, p.output_idx_slots)
            empty_input_slots(p.input_slots)
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            _events.append(event)
        if i >= warm_up:
            events.append(_events)
    torch.cuda.synchronize()
    # Parse events to get ops_time
    for profile in profile_result.profile.values():
        profile.ops_time = 0.
    for _events in events:
        last_event: torch.cuda.Event = None
        for i, _event in enumerate(_events):
            if last_event is None:
                assert i == 0
                profile_result.profile[i].ops_time += 0.
            else:
                profile_result.profile[i].ops_time += last_event.elapsed_time(_event)
            last_event = _event
    for profile in profile_result.profile.values():
        profile.ops_time /= repeat * 1e3    # ms to s
    return profile_result.ret_store

def local_random_exec(profile_result: OffloadProfile):
    # Start exec
    for i, p in enumerate(profile_result.profile.values()):
        args, kwargs = p.func_args
        intermediates: torch.Tensor = p.func(*args, **kwargs)
        # fill arguments for following ops
        fill_output_to_input_slots(intermediates, p.output_idx_slots)
        empty_input_slots(p.input_slots)
    torch.cuda.synchronize()
    return profile_result.ret_store


def local_random_exec_cuda_profile_partial(profile_result: OffloadProfile, log=print):
    torch.cuda.synchronize()
    # Start exec
    input_ratio = np.array(list(range(2, 11, 2)), dtype=float)/10.
    x = np.linspace(0, 1, 6)    # [0., 0.2, 0.4, 0.6, 0.8, 1.0]
    profile_result.partial_ops_time = np.zeros((len(input_ratio), len(x), len(profile_result.profile)))
    origin_time = 0.
    for i, p in enumerate(profile_result.profile.values()):
        if p.hook_kwargs["barrier"]:
            break
        origin_time += p.ops_time
    stop_idx = i    #
    offset = profile_result.profile[stop_idx].offset
    formula = profile_result.profile[stop_idx].formula  # [float, float]
    origin_input: List[torch.Tensor] = []
    iterate_tensor(profile_result.profile[0].func_args, origin_input.append)
    assert len(origin_input) == 1, "Only support one tensor input"
    input_shape = np.array(origin_input[0].shape)
    h, w = input_shape[-2:]

    warmup = 3
    repeat = 3
    for i, _input_ratio in enumerate(input_ratio):
        for j, _x in enumerate(x):
            new_shape = np.array(input_shape)
            new_shape[-2:] = (new_shape[-2:] * _input_ratio).astype(int)
            new_shape = pad_shapes_for_conv(new_shape, formula)
            if _x == 0.:
                ratio = 0.
            else:
                new_shape[-1] = int(np.around(new_shape[-1] * _x)) + offset
                new_shape = pad_shapes_for_conv(new_shape, formula)
                h_offset = h//2 - new_shape[-2]//2
                w_offset = w//2 - new_shape[-2]//2
                new_input = [origin_input[0][..., h_offset:h_offset+new_shape[-2], w_offset:w_offset+new_shape[-1]]]
                profile_result.profile[0].func_args = [new_input, {}]

                torch.cuda.synchronize()
                for l in range(warmup + repeat):
                    if l == warmup:
                        stime = time.time()
                    for k, p in enumerate(profile_result.profile.values()):
                        args, kwargs = p.func_args
                        intermediates: torch.Tensor = p.func(*args, **kwargs)
                        fill_output_to_input_slots(intermediates, p.output_idx_slots)
                        empty_input_slots(p.input_slots)
                        if k >= stop_idx:
                            break
                    torch.cuda.synchronize()
                avg_dur = (time.time() - stime) / repeat
                ratio = avg_dur / origin_time
            for k, p in enumerate(profile_result.profile.values()):
                if k >= stop_idx:
                    profile_result.partial_ops_time[i,j,k] = p.ops_time
                else:
                    profile_result.partial_ops_time[i,j,k] = p.ops_time * ratio
            log(f"Partial profile for input_ratio {_input_ratio:.2f} x {_x:.2f} finished. Time ratio {ratio}")