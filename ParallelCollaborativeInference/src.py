#! /usr/bin/env python
import os
import ptvsd
import atexit
import time
import copy
import os.path as osp
import pickle
from collections import OrderedDict
import socket
import socket
import gc
import numpy as np
import torch
import torch.distributed as dist
import dill
import torch.backends
from .hook_tensor import (profile_tensor_factory,
                        OffloadProfile, TorchOPProfile,
                        keep_funcs)
from .recursive_pickle_obj import recur_dump_obj
from ._utils import iterate_tensor, iterate_all_close, log_dur
from .test_torch_dist import TorchDistributedStream, empty_preamble
from .schedule import PCI_scheduler
# from .random_exec import (compile_plan_to_static_exec, random_exec_compiled, local_random_exec_cuda_profile,
#                           local_random_exec)

from .img_matcher import ImageMatcher
from .random_exec_cached import (compile_plan_to_static_exec_cached, 
                                 local_random_exec_cuda_profile_partial,
                                 local_random_exec, local_random_exec_cuda_profile,
                                 random_exec_compiled)
from .profile_pickle import profile_pickle

import numba.cuda


# None zero GPU Memory occupancy will be observed due to cuda context
# https://discuss.pytorch.org/t/nvidia-smi-does-not-drop-after-empty-cache-although-cuda-memory-summary-says-there-is-no-current-usage/143741
# print(f"{torch.cuda.memory_allocated()} {torch.cuda.max_memory_allocated()}")

class ParallelCollaborativeInference:
    def __init__(self, offload=True,  parallel_approach = "select",
                debug=False, constraint_latency=False, log=print, warmup=2, repeat=3) -> None:
        self.offload = offload
        self.log = log
        self.debug = debug
        self.parallel_approach = parallel_approach
        self.constraint_latency = constraint_latency
        self.fixed_bw = None
        self.warmup = int(warmup)
        self.repeat = int(repeat)
        assert self.warmup >= 0 and self.repeat > 0

        log("Configuring torch to use deterministic behaviors.")
        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)
        # torch.backends.cudnn.deterministic = True

        log(f"parallel approach {parallel_approach}")
        log(f"constraint_latency {constraint_latency}")

    def offload_order(self, bw: int):
        self.fixed_bw = bw

    def start_client(self, model: torch.nn.Module, init_forward_count=0):
        torch.device("cuda:0")
        try:
            numba.cuda.current_context()
        except Exception:
            self.log("Error: Unable to start CUDA Context", exc_info=True)
        assert dist.is_available(), "torch.distributed unavailable."
        assert dist.is_mpi_available(), "torch.distributed mpi backend unavailable."
        self.log(f"Initializing torch.distributed with mpi backend...")
        # issue: https://github.com/openucx/ucx/issues/4707
        dist.init_process_group("mpi")
        host_name = socket.gethostname()
        server_host_name = [None]

        rank = dist.get_rank()
        dist.broadcast_object_list(server_host_name, 0)
        server_host_name = server_host_name[0]
        world_size = dist.get_world_size()
        assert rank > 0, f"Client has a wrong rank {rank}"
        self.log(f"Client {host_name}: rank {rank} world size {world_size}.")

        if str(rank) in os.environ.get("DEBUG_RANKS", "").split(","):
            addr = ('0.0.0.0', 11000+rank)
            print(f"Client rank {rank}: Waiting for debug attach at {addr}...")
            ptvsd.enable_attach(address=addr)
            ptvsd.wait_for_attach()

        warned = False
        param_num = 0
        for p in model.parameters():
            if p.requires_grad and not warned:
                warned = True
                self.log("Warning: model still requires grad. Setting requires grad to False.")
            p.requires_grad = False
            param_num += p.numel()
        self.log(f"Model parameter number {param_num/1e6:.4f}M.")
        prefix = f"Client {host_name}: " 
        model_name = f"{model.__class__.__name__}_{int(param_num/1e6)}M_{host_name}"
        model_obj_list = [recur_dump_obj(model), init_forward_count, model_name]

        dist.broadcast_object_list(model_obj_list, rank, device=torch.device("cpu"))
        self.log(f"Send model to server {len(model_obj_list[0])/1024/1024:.4f}MB.")
        del model_obj_list
        gc.collect()
        torch.cuda.empty_cache()

        model.forward = self.common_process(model, rank, opposite_rank=0, init_forward_count=init_forward_count, role="client", prefix=prefix, log=self.log, model_hash=model_name, local_test=host_name==server_host_name)
        def close():
            dist.destroy_process_group()
            gc.collect()
            torch.cuda.empty_cache()
            self.log(prefix + "terminated.")
        atexit.register(close)

    def start_server(self):
        torch.device("cuda:0")
        try:
            numba.cuda.current_context()
        except Exception:
            self.log("Error: Unable to start CUDA Context", exc_info=True)

        self.log("Starting ParallelCollaborativeInference server...")
        assert dist.is_available(), "torch.distributed unavailable."
        assert dist.is_mpi_available(), "torch.distributed mpi backend unavailable."
        self.log(f"Initializing torch.distributed with mpi backend...")
        dist.init_process_group("mpi")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        host_name = socket.gethostname()
        prefix = f"Server {host_name}: "

        dist.broadcast_object_list([host_name], rank)

        assert rank == 0, f"Server has a wrong rank {rank}"
        self.log(prefix + f"rank {rank} world size {world_size}")

        if str(rank) in os.environ.get("DEBUG_RANKS", "").split(","):
            addr = ('0.0.0.0', 11000+rank)
            print(f"Server rank {rank}: Waiting for debug attach at {addr}...")
            ptvsd.enable_attach(address=addr)
            ptvsd.wait_for_attach()

        recv_info = [None, None, None]
        dist.broadcast_object_list(recv_info, rank + 1, device=torch.device("cpu"))
        model_msg, init_forward_count, model_name = recv_info
        model: torch.nn.Module = dill.loads(model_msg)
        del model_msg
        del recv_info
        
        model.forward = self.common_process(
            model, rank, opposite_rank=rank+1, init_forward_count=init_forward_count,
            role="server", prefix=prefix, log=self.log,
            model_hash=model_name)
        def close():
            dist.destroy_process_group()
            gc.collect()
            torch.cuda.empty_cache()
            self.log(prefix + "terminated.")
        atexit.register(close)
        while True:
            model() # Forward without input

    @torch.no_grad()
    def common_process(self, model: torch.nn.Module, rank, opposite_rank,
                       init_forward_count=0, role="client",
                       prefix="client", log=print, model_hash="", local_test=False):
        old_forward = model.forward
        profile_result = OffloadProfile()
        sock = TorchDistributedStream(rank, opposite_rank, log)
        
        matcher = ImageMatcher(debug=True)
        if role == "client":
            scheduler = PCI_scheduler(self.parallel_approach)
            parallel_approach = self.parallel_approach
            dist.broadcast_object_list([self.parallel_approach], rank, device=torch.device("cpu"))
        else:
            objs = [None]
            dist.broadcast_object_list(objs, opposite_rank, device=torch.device("cpu"))
            parallel_approach = objs[0]
            scheduler = PCI_scheduler(parallel_approach)

        @torch.no_grad()
        def _profile_forward(*args, **kwargs):
            # change cls before calculating id of the tensors
            hook_level = [0]
            profile_tensor_cls = profile_tensor_factory(profile_result, hook_level)
            args, kwargs = iterate_tensor([args, kwargs], profile_tensor_cls)
            hook_level[0] = 0   # Clear hook level
            profile_result.idx = 0

            # profile input
            args, kwargs = profile_tensor_cls.__torch_function__(None, None, args, kwargs)
            ret = old_forward(*args, **kwargs)
            # profile output
            ret = profile_tensor_cls.__torch_function__(None, None, [ret], None)

            ret = iterate_tensor(ret, torch.Tensor)
            orig_ops_num = len(profile_result.profile)
            self.parse_profile(profile_result)
            profile_pickle(profile_result, log=log)
            profile_result.end_idx = len(profile_result.profile) - 1
            return ret, orig_ops_num

        @torch.no_grad()
        def profile_forward(*args, **kwargs):
            warmup, repeat = self.warmup, self.repeat
            if "img" in kwargs:
                del kwargs["img"]
            if role == "client":
                log(prefix + "sending init input to server")
                dist.broadcast_object_list([args, kwargs], rank, device=torch.device("cpu"))
            else:
                msg = [None, None]
                log(prefix + "recving init input from client")
                dist.broadcast_object_list(msg, opposite_rank, device=torch.device("cpu"))
                args, kwargs = msg
                time.sleep(10)
            origin_input = copy.deepcopy([args, kwargs])
            log(f"Input size {len(pickle.dumps([args, kwargs]))/1024/1024:.4f}MB")
            log(f"Forwarding for {init_forward_count}(+{warmup} warmup and {repeat} repeat) times for initialization.")
            count = 0
            while count != warmup:
                orig_ret = old_forward(*args, **kwargs)
                torch.cuda.synchronize()
                count += 1

            count = 0
            _init_forward_count = init_forward_count + repeat
            stime = time.time()
            while count != _init_forward_count:
                orig_ret = old_forward(*args, **kwargs)
                torch.cuda.synchronize()
                count += 1
            dur = (time.time() - stime)/count   # Average duration for each forward
            log(f"Forward of the original model takes average {dur:.4f}s.")
            ret1, orig_ops_num = _profile_forward(*origin_input[0], **origin_input[1])
            assert iterate_all_close(orig_ret, ret1)

            log(f"Output size {len(pickle.dumps(ret1))/1024/1024:.4f}MB")

            num = 5
            profile_result.profile[0].func_args = [args, kwargs]
            stime = time.time()
            for _ in range(num):
                ret3 = local_random_exec(profile_result)
            _dur = (time.time() - stime) / num
            assert iterate_all_close(ret3, ret1)
            log(f"Local random exec takes average {_dur:.4f}s.")

            ret2 = local_random_exec_cuda_profile(profile_result,
                                                  warm_up=warmup, repeat=repeat)
            assert iterate_all_close(ret2, ret1)
            log(f"Start profiling partial input time...")
            factor = dur / sum(p.ops_time for p in profile_result.profile.values())
            for p in profile_result.profile.values():
                p.ops_time *= factor
            local_random_exec_cuda_profile_partial(profile_result)


            if role == "client" and local_test:    # For debug
                factor = 20.
            log(f"Operator records (align ops time with factor {factor:.4f}): ")
            profile_result.partial_ops_time *= factor
            accumulated_time = 0.
            for p in profile_result.profile.values():
                p.ops_time *= factor
                accumulated_time += p.ops_time
                log(f"{p} accu_time {accumulated_time:.4f}s")
            profile_result.local_comp_time = sum(
                p.ops_time for p in profile_result.profile.values())
            log(f"total {len(profile_result.profile)} ops (filtered from {orig_ops_num} ops); time {sum(p.ops_time for p in profile_result.profile.values()):.4f}s (aligned by {factor:.4f}).\n")

            if role == "client":
                dist.broadcast_object_list([
                    {"ops": profile_result.copy_for_transmission(),
                     "constraint_latency": self.constraint_latency}], rank,
                                           device=torch.device("cpu"))
                log("Waiting for graph processing at the server")
                plan_objs = [None]
                dist.broadcast_object_list(plan_objs, opposite_rank,
                                           device=torch.device("cpu"))
                scheduler.recv_plan(plan_objs[0])
                log("Got graph plan from server")
            elif role == "server":
                ops_dict_obj = [None]
                dist.broadcast_object_list(ops_dict_obj, opposite_rank,
                                           device=torch.device("cpu"))
                ops_dict = ops_dict_obj[0]
                robot_ops = ops_dict["ops"]
                scheduler.update_ops(robot_ops, profile_result)
                constraint_latency = ops_dict["constraint_latency"]
                store_plan_path = f"mixed_plan_{model_hash}.pkl"
                if not osp.exists(store_plan_path):
                    if constraint_latency:
                        # fix latency requirement to 1Hz
                        scheduler.required_latency = 1.
                        self.log(f"Setting required_latency to {scheduler.required_latency:.4}s.")
                    self.log("Computing plan for client.")
                    scheduler.build_graph()
                    with open(store_plan_path, "wb") as f:
                        pickle.dump({"server plan": scheduler.server_plans,
                            "client plan": scheduler.client_plans}, f)
                else:
                    with open(store_plan_path, "rb") as f:
                        stored_plan = pickle.load(f)
                    scheduler.server_plans, scheduler.client_plans = \
                        stored_plan["server plan"], stored_plan["client plan"]
                    self.log(f"Loaded precomputed mixed plan from {store_plan_path}.")
                dist.broadcast_object_list([scheduler.client_plans], rank,
                                           device=torch.device("cpu"))
                scheduler.recv_plan(scheduler.server_plans)  # Server does not output

            self.log(prefix + "init forward complete.")
            if self.offload:
                compile_plan_to_static_exec_cached(profile_result, scheduler.graph_plan,
                                        sock, sock.add_suffix_to_log, log, role, matcher)
            else:
                compile_plan_to_static_exec_cached(profile_result, scheduler.graph_plan,
                                        sock, log, log, role, matcher)

            for key, plan in scheduler.graph_plan.items():
                plan["send_num"] = int(sum(plan["offload"]))
            profile_result.profile[0].func_args = [None, None]
            if not self.offload:
                model.forward = local_forward
            elif role == "server":
                model.forward = server_forward
            elif role == "client":
                model.forward = client_forward
            else:
                raise RuntimeError(f"Wrong role {role}")
            nonlocal start_profile
            nonlocal all_offload_size
            nonlocal local_computation_time
            nonlocal local_use_cache_ratio_max
            nonlocal no_cache

            start_profile = profile_result.profile[0]
            all_offload_size = start_profile.output_size
            local_computation_time = profile_result.local_comp_time
            local_use_cache_ratio_max = 1- use_cache_overhead / local_computation_time - 0.2
            no_cache = False
            if local_use_cache_ratio_max < 0:
                no_cache = True
                log(f"Warning: cache overhead longer than local computation time! Disabling any cache.")
            return ret1

        count = [0]
        use_cache_overhead = 0.032  # second
        cache_location = [1]
        start_profile: TorchOPProfile = None    # To be filled after profile
        all_offload_size = None     # To be filled after profile
        local_computation_time = None   # To be filled after profile
        local_use_cache_ratio_max = None    # To be filled after profile
        no_cache = False


        def local_forward(*args, **kwargs):
            img = kwargs.pop("img")
            with log_dur(log, prefix=prefix + f"{count[0]} th inference; est bw {0.}MBps, est exec time {profile_result.local_comp_time:.4f}s; "):
                if parallel_approach == "cached" and not no_cache:
                    tensors = []
                    iterate_tensor([args, kwargs], tensors.append)
                    with log_dur(self.log, prefix=prefix + f"track input takes"):
                        matcher.match(img)
                    if matcher.can_interpolate and start_profile.cached_output is not None:
                        # reorganize uncached areas into a new rectangle image and estimate its shape
                        size = 0
                        input_tensors = []
                        for current_feature_map, last_feature_map in zip(tensors, start_profile.cached_output):
                            reorg_tensor: torch.Tensor = matcher.reorganize_uncached_zones(
                                last_feature_map, current_feature_map, min_diff=0.4,
                                offset=start_profile.cached_offset,
                                formula=start_profile.cached_formula)
                            size += reorg_tensor.nelement() * reorg_tensor.element_size()
                            input_tensors.append(reorg_tensor)
                        input_ratio = min(int(np.around(size / all_offload_size/0.2)*2), 10)
                    else:
                        input_ratio = 10
                        input_tensors = tensors
                    if input_ratio > 0:
                        start_profile.func_args = [input_tensors, {}]
                        ret = random_exec_compiled(profile_result, (0, input_ratio, 1))
                        start_profile.cached_output = [tensors[0]]
                    else:
                        ret = profile_result.ret_store
                else:
                    ret = old_forward(*args, **kwargs)
                torch.cuda.default_stream().synchronize()
            count[0] += 1
            return ret

        def server_forward():
            last_bandwidth, input_ratio, _cache_location, recv_preamble = sock.recv_obj()
            if recv_preamble:
                sock.recv_tensor()
            if input_ratio > 0:
                plan = scheduler.graph_plan[(last_bandwidth, input_ratio, _cache_location)]
            sock.start_record_log(prefix + f"{count[0]} th sock: ", plan["send_num"])
            with log_dur(sock.add_suffix_to_log, prefix=prefix + f"{count[0]} th inference est bw {last_bandwidth}MBps, input_ratio {input_ratio} cache location {_cache_location} est exec time {plan['est_time']:.4f}s"):
                if input_ratio > 0:
                    random_exec_compiled(profile_result, (last_bandwidth, input_ratio, _cache_location))
            count[0] += 1

        def client_forward(*args, **kwargs):
            img = kwargs.pop("img")
            last_bandwidth = int(min(sock.last_bandwidth, scheduler.max_bw)) \
                    if self.fixed_bw is None else self.fixed_bw
            input_tensors = []
            tensors = []
            iterate_tensor([args, kwargs], tensors.append)
            with log_dur(self.log, prefix=prefix + f"track and reorganize input"):
                if parallel_approach == "cached" and not no_cache:
                # TODO change log to sock.add_suffix_to_log
                    img: np.ndarray = kwargs["img"]
                    matcher.match(img)
                    if matcher.can_interpolate and start_profile.cached_output is not None:
                        # reorganize uncached areas into a new rectangle image and estimate its shape
                        max_uncached_ratio = min(local_use_cache_ratio_max,
                                        1 - use_cache_overhead / (all_offload_size/last_bandwidth/1024/1024)) - 0.2
                        if max_uncached_ratio >= 1:
                            input_ratio = 10
                        else:
                            size = 0
                            for current_feature_map, last_feature_map in zip(tensors, start_profile.cached_output):
                                reorg_tensor: torch.Tensor = matcher.reorganize_uncached_zones(
                                    last_feature_map, current_feature_map, min_diff=2/255.,
                                    offset=start_profile.cached_offset,
                                    formula=start_profile.cached_formula,
                                    max_ratio=max_uncached_ratio)
                                size += reorg_tensor.nelement() * reorg_tensor.element_size()
                                input_tensors.append(reorg_tensor)
                            input_ratio = min(int(np.ceil(size / start_profile.output_size/0.2)*2), 10)
                    else:
                        input_ratio = 10
                        input_tensors = tensors
                        matcher.reorganize_info = {"reorg_shape": tensors[0].shape}
                    if input_ratio > 0:
                        start_profile.cached_output = [tensors]
                else:
                    input_ratio = 10
                    input_tensors = tensors
                    matcher.reorganize_info = {"reorg_shape": tensors[0].shape}
            plan = scheduler.graph_plan[(last_bandwidth, input_ratio, cache_location[0])]
            start_profile.func_args = [input_tensors, {}]

            send_preamble = sock.send_finished() and (plan["send_num"] == 0 or input_ratio == 0.)
            send_num = 1 if send_preamble else plan["send_num"]

            sock.start_record_log(prefix + f"{count[0]} th sock: ", send_num+1)
            sock.send_obj([last_bandwidth, input_ratio, cache_location[0], send_preamble])
            if send_preamble:
                sock.send_tensor([empty_preamble])

            with log_dur(sock.add_suffix_to_log, prefix=prefix + f"{count[0]} th inference est bw {last_bandwidth}MBps, est exec time {plan['est_time']:.4f}s"):
                if input_ratio > 0:
                    ret = random_exec_compiled(profile_result, last_bandwidth)
                else:
                    ret = profile_result.ret_store
            count[0] += 1
            cache_location[0] = plan["cache_location"]
            return ret
        return profile_forward


    def parse_profile(self, profile_result: OffloadProfile):
        all_profiles = profile_result.profile
        idx_array = list(all_profiles.keys())
        for idx in idx_array:
            profile: TorchOPProfile = all_profiles[idx]
            input_ids: list = profile.input_ids
            output_ids: list = profile.output_ids

            # parse input/output relationship by querying id in previous output
            for i, _id in enumerate(input_ids):
                for _idx in range(0, idx):
                    if _id in all_profiles[_idx].output_ids:
                        hit_idx = all_profiles[_idx].output_ids.index(_id)
                        if _idx in profile.input_from:
                            self.log(f"Warning: {idx}op has duplicated input from {_idx}op")
                        else:
                            profile.input_from.append(_idx)
                        if idx in all_profiles[_idx].output_to:
                            self.log(f"Warning: {_idx}op has duplicated output to {idx}op")
                        else:
                            all_profiles[_idx].output_to.append(idx)
                        if hit_idx not in all_profiles[_idx].output_idx_slots:
                            all_profiles[_idx].output_idx_slots[hit_idx] = [
                                profile.input_slots[i]]
                            # {output_idx: [(op_idx, input_idx)]}
                        else:
                            all_profiles[_idx].output_idx_slots[hit_idx].append(
                                profile.input_slots[i])

            # Since id can be reused, remove any duplicated id in previous output
            for _id in output_ids:
                for _idx in range(0, idx):
                    if _id in all_profiles[_idx].output_ids:
                        hit_idx = all_profiles[_idx].output_ids.index(_id)
                        all_profiles[_idx].output_ids[hit_idx] = None


        for idx in reversed(idx_array):    # ignore end
            profile = all_profiles[idx]
            output_idx_slots = profile.output_idx_slots
            if len(output_idx_slots) > 1:
                # sort keys of ordered dict in ascending order
                sorted_output_idx_slots = OrderedDict()
                sorted_keys = sorted(list(output_idx_slots.keys()))
                for _idx in sorted_keys:
                    sorted_output_idx_slots[_idx] = output_idx_slots[_idx]
                profile.output_idx_slots = sorted_output_idx_slots
            if len(profile.output_to) == 0 and not profile.keep:
                # if no output and not explicitly keep, remove this profile
                for i, _idx in enumerate(profile.input_from):
                    all_profiles[_idx].output_to.remove(idx)
                    # Remove output slots that fills the input of the current profile
                    for key, slots in all_profiles[_idx].output_idx_slots.items():
                        remain_slots = []
                        for slot in slots:
                            if slot.idx != idx:
                                remain_slots.append(slot)
                        all_profiles[_idx].output_idx_slots[key] = remain_slots
                for _idx in range(idx+1, idx_array[-1]+1):
                    if _idx in all_profiles:
                        _profile = all_profiles[_idx]
                        _profile.idx -= 1
                for _profile in all_profiles.values():
                    for i, _ in enumerate(_profile.output_to):
                        if _profile.output_to[i] > idx:
                            _profile.output_to[i] -= 1
                    for i, _ in enumerate(_profile.input_from):
                        if _profile.input_from[i] > idx:
                            _profile.input_from[i] -= 1
                del all_profiles[idx]
        profile_result.profile = OrderedDict()
        for i, profile in enumerate(all_profiles.values()):
            for slot in profile.input_slots:
                slot.idx = i
            profile_result.profile[i] = profile

        # patch __setitem__: __setitem__ does not have a return value
        # but only modifies input_from[0] inplace;
        # Correct the data dependency here and also __setitem__ should not be offloaded.
        for i, profile in enumerate(profile_result.profile.values()):
            if profile.func_name == "__setitem__":
                inplace_mod_idx = profile.input_from[0]
                for idx in profile_result.profile[inplace_mod_idx].output_to:
                    if idx > i:     # op after this inplace __setitem__ also depends on this op
                        _p = profile_result.profile[idx]
                        _p.input_from.append(i)
                        profile.output_to.append(idx)

        # Check profile valid
        new_all_profiles = profile_result.profile
        for key, profile in new_all_profiles.items():
            assert key == profile.idx
            assert len(profile.output_to) > 0 or profile.func_name in keep_funcs
            if not len(profile.output_idx_slots) == len(profile.output_shapes) and profile.func_name != "_end":
                raise RuntimeError(str(profile))
            for idx in profile.output_to:
                assert idx in new_all_profiles
                assert key in new_all_profiles[idx].input_from
            for idx in profile.input_from:
                assert idx in new_all_profiles
                assert key in new_all_profiles[idx].output_to

        end_profile = new_all_profiles[len(new_all_profiles)-1]
        for i, ret_slot in enumerate(profile_result.ret_slots):
            ret_slot.idx = end_profile.idx
            end_profile.output_idx_slots[i] = [ret_slot]
            end_profile.output_to = [-1] * len(profile_result.ret_slots)

        # Temp fix for branches
        idx_array = list(new_all_profiles.keys())
        for idx in idx_array:
            profile = new_all_profiles[idx]
            valid_output_len = len(profile.output_to)
            if valid_output_len > 1:
                current_end = set(profile.output_to)
                while len(current_end) > 1:
                    current_idx = min(current_end)
                    current_end = list(current_end)
                    del current_end[current_end.index(current_idx)]
                    current_end += new_all_profiles[current_idx].output_to
                    current_end = set(current_end)

                for _idx in range(idx, list(current_end)[0] + 1):
                    new_all_profiles[_idx].masked = True
        new_all_profiles[idx_array[-1]].masked = False

        profile_result.profile[0].local_dim = len(profile_result.profile[0].output_shapes[0]) - 1
        start_input_shape = profile_result.profile[0].input_shapes[0]
        for profile in profile_result.profile.values():
            idx: int = profile.idx
            func: str = profile.func_name
            func_args = profile.func_args
            input_shapes = profile.input_shapes
            output_shapes = profile.output_shapes
            if idx > 0:
                parent_profile = profile_result.profile[profile.input_from[0]]
                last_local_dim = parent_profile.local_dim
            else:
                parent_profile = None
                last_local_dim = None
            if idx in [profile_result.end_idx, 0]:
                barrier = idx != 0
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile}
                if parent_profile:
                    profile.local_dim = last_local_dim
            elif func in ["__getitem__"]:
                args, kwargs = func_args
                slice_arg: list = args[1]
                assert len(kwargs) == 0
                record_selected_indices = False
                if last_local_dim is None:
                    barrier = True
                    profile.local_dim = None
                elif isinstance(slice_arg, int): # get item at the first dim
                    if slice_arg == last_local_dim:
                        barrier = True
                        profile.local_dim = None
                    else:
                        barrier = False
                        profile.local_dim = last_local_dim - 1
                elif isinstance(slice_arg, (torch.Tensor)):
                    # TODO: this situation is very complicated to analyse, leave it to future
                    if len(input_shapes[0]) != len(output_shapes[0]):
                        barrier = True
                        profile.local_dim = None
                    else:
                        barrier = False
                        profile.local_dim = last_local_dim
                        if len(input_shapes) > 1:   # select from another tensor
                            record_selected_indices = True
                elif isinstance(slice_arg, list):
                    slice_arg = list(slice_arg)
                    try:
                        find_ellipsis = slice_arg.index(...)
                    except ValueError:
                        find_ellipsis = None
                    if find_ellipsis is not None:
                        origin_shape_len = len(input_shapes[0])
                        ellipsis_len = origin_shape_len - len(slice_arg)
                        ellipsis_idx = slice_arg.index(...)
                        [slice_arg.insert(ellipsis_idx, None) for _ in range(ellipsis_len)]
                    elif (len_diff:=len(input_shapes[0])-len(slice_arg)) > 0:
                        slice_arg += len_diff * [None]
                    if isinstance(slice_arg[last_local_dim], int):
                        barrier = True
                        profile.local_dim = None
                    else:
                        barrier = False
                        dim_reduced = 0
                        for i, a in enumerate(slice_arg):
                            if i == last_local_dim:
                                profile.local_dim = last_local_dim - dim_reduced
                                break
                            if isinstance(a, int):
                                dim_reduced += 1
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile, "record_selected_indices": record_selected_indices} # Regular offloading
            elif func in ["__setitem__"]:
                barrier = last_local_dim is None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
                profile.local_dim = last_local_dim
            elif func.startswith(("cat")):
                barrier = last_local_dim is None
                args, kwargs = func_args
                if len(profile.input_shapes) > 1:
                    largest = np.argmin([profile_result.profile[i].formula[1] for i in profile.input_from[:len(profile.input_shapes)]])
                    profile.formula = profile_result.profile[profile.input_from[largest]].formula
                if "dim" in kwargs:
                    dim = kwargs["dim"]
                elif len(args) > 1:
                    dim = args[1]
                else:
                    dim = 0
                cat_at_dense_dim = False
                if dim != last_local_dim:
                    align_shape = True
                else:
                    cat_at_dense_dim = True
                    for i in profile.input_from[:len(profile.input_shapes)]:
                        profile.origin_shapes.extend(profile_result.profile[i].origin_shapes)
                    profile.current_dense_dim_len = output_shapes[dim]
                    align_shape = False
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile,
                    "apply_rate": True, "align_shape": align_shape, "cat_at_dense_dim": cat_at_dense_dim} # Regular offloading
                # If cat at dense dim, need to apply x, record indices
                profile.local_dim = last_local_dim
            elif func.startswith((
                "add", "sub", "rsub", "div", "mul",
                "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
                "exp", "pow",
                )) or func in [
                    "le", "lt", "gt", "ge", "eq","nms",
                ]: # Element-wise operations that keep tensor shape
                align_shape = False
                if last_local_dim is not None and len(input_shapes) < 2:
                    mat = []
                    for arg in func_args[0]:
                        if isinstance(arg, torch.Tensor) and len(arg.shape) > last_local_dim and arg.shape[last_local_dim] > 1:
                            mat.append(True)
                        else:
                            mat.append(False)
                    align_shape = np.all(mat)
                barrier = last_local_dim is None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile, "align_shape": align_shape} # Regular offloading
                profile.local_dim = last_local_dim
            elif func.startswith(("sin", "cos", "tan", "asin", "acos", "atan", "arc",
                "batch_norm", "layer_norm",
                "relu", "rrelu", "gelu", "sigmoid", "sign", "selu", "hardswish",
                "hardsigmoid", "silu", 
                "sqrt", "rsqrt",)) or func in [
                    "contiguous", "interpolate", "clone", "detach", 
                    "float", "int", "double", "long", "abs", "type"]:
                barrier = last_local_dim is None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
                profile.local_dim = last_local_dim
            elif func in ["view", "reshape"]:
                barrier = False
                flattened = False
                if last_local_dim is not None:
                    input_shape = input_shapes[0]
                    output_shape = output_shapes[0]
                    search_size = np.prod(input_shape[:last_local_dim+1])
                    searched = np.nonzero(np.cumprod(output_shape) == search_size)[0]
                    if len(searched):
                        profile.local_dim = searched[0]
                        if input_shape[last_local_dim] < output_shape[searched[0]]:
                            flattened = True
                            if searched[0] < last_local_dim:
                                profile.origin_shapes = [input_shape[searched[0]:last_local_dim+1]]
                                profile.current_dense_dim_len = output_shape[searched[0]]
                            print(f"{profile} has flattened shape: input shape {input_shape} output shape {output_shape}")
                        args, kwargs = func_args
                        searched_idx = searched[0]
                        if -1 in args[1:] and args[1+searched_idx] != -1:
                            idx = args[1:].index(-1)
                            args[1+idx] = output_shape[idx]
                        args[1+searched_idx] = -1   # Change the shape of local dim to be flexible
                    else:
                        profile.local_dim = None
                        barrier = True
                else:
                    barrier = True
                    profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile, "flattened": flattened} # Regular offloading
            elif func in ["flatten", "ravel"]:
                if last_local_dim is not None and last_local_dim == len(input_shapes[0])-1:
                    profile.local_dim = 0
                    barrier = False
                else:
                    profile.local_dim = None
                    barrier = True
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
            elif func in ["squeeze", "unsqueeze"]:
                barrier = False
                if last_local_dim is not None:
                    input_shape = input_shapes[0]
                    output_shape = output_shapes[0]
                    search_size = np.prod(input_shape[:last_local_dim+1])
                    searched = np.nonzero(np.cumprod(output_shape) == search_size)[0]
                    if len(searched):
                        profile.local_dim = searched[0]
                    else:
                        profile.local_dim = None
                        barrier = True
                else:
                    barrier = True
                    profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
            elif func in ["unbind"]:
                args, kwargs = func_args
                if "dim" in kwargs:
                    dim = kwargs["dim"]
                elif len(args) > 1:
                    dim = args[1]
                else:
                    dim = 0
                if last_local_dim is not None and dim != last_local_dim:
                    barrier = False
                    if dim > last_local_dim:
                        profile.local_dim = last_local_dim
                    else:
                        profile.local_dim = last_local_dim - 1
                else:
                    barrier = True
                    profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
            elif func in ["max", "min", "any", "all", "argmax", "argmin"]:
                args, kwargs = func_args
                if "dim" in kwargs:
                    dim = kwargs["dim"]
                elif len(args) > 1:
                    dim = args[1]
                else:
                    dim = None
                if last_local_dim is not None and dim is not None and dim != last_local_dim:
                    barrier = False
                    if "keepdim" in kwargs and kwargs["keepdim"] or len(args) > 2 and args[2] or dim > last_local_dim:
                        profile.local_dim = last_local_dim
                    else:
                        profile.local_dim = last_local_dim - 1
                else:
                    barrier = True
                    profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
            elif func.startswith(("permute")):
                if last_local_dim is not None:
                    barrier = False
                    args, _ = func_args
                    if len(args) == 2 and isinstance(args[1], list):
                        indices = args[1]
                    else:
                        indices = func_args[0][1:]
                    searched = np.nonzero(
                        np.arange(len(input_shapes[0]))[indices] == last_local_dim)[0]
                    if len(searched):
                        profile.local_dim = searched[0]
                    else:
                        profile.local_dim = last_local_dim
                else:
                    barrier = True
                    profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile} # Regular offloading
            elif func.startswith((
                "conv", "conv_transpose2d"
            )):  # Convolution operations
                args, kwargs = func_args
                profile.conv = True
                if "stride" in kwargs:
                    stride = kwargs["stride"][0]
                else:
                    stride = args[3][0]
                if "padding" in kwargs:
                    padding = kwargs["padding"][0]
                else:
                    padding = args[4][0]
                kernel_size = args[1].shape[-1]

                if func.startswith("conv_transpose2d"):
                    profile.unconv = True
                    profile.formula[0] = parent_profile.formula[0] * stride
                    profile.formula[1] = (kernel_size - 2*padding) - stride + parent_profile.formula[1] * stride
                    profile.offset = parent_profile.offset
                    profile.offset_factor = parent_profile.offset_factor / stride
                else:
                    profile.formula[0] = parent_profile.formula[0] / stride
                    profile.formula[1] = (2*padding - kernel_size)/stride+ 1 + parent_profile.formula[1] / stride
                    profile.offset = kernel_size - 1 + parent_profile.offset - padding
                    profile.offset_factor = parent_profile.offset_factor * stride
                profile.hook_kwargs = {
                    "idx": idx, "conv": True,
                    "barrier": False, "profile": profile}
                profile.local_dim = len(input_shapes[0]) - 1
            elif func.startswith((
                "max_pool", "avg_pool",
            )): # Convolution operations
                args, kwargs = func_args
                profile.conv = True
                if "stride" in kwargs:
                    stride = kwargs["stride"]
                else:
                    stride = args[2]
                if "padding" in kwargs:
                    padding = kwargs["padding"]
                else:
                    padding = args[3]
                if "kernel_size" in kwargs:
                    kernel_size = kwargs["kernel_size"]
                else:
                    kernel_size = args[1]
                profile.kernel_size, profile.stride, profile.padding = kernel_size, stride, padding
                profile.formula[0] = parent_profile.formula[0] / stride
                profile.formula[1] = (2*padding - kernel_size)/stride+ 1 + parent_profile.formula[1] / stride
                profile.offset = kernel_size - 1 + parent_profile.offset - padding
                profile.offset_factor = parent_profile.offset_factor * stride
                profile.hook_kwargs = {
                    "idx": idx, "conv": True,
                    "barrier": False, "profile": profile}
                profile.local_dim = len(input_shapes[0]) - 1
            elif func.startswith((
                "bmm",
            )): # Convolution operations
                assert len(input_shapes) == 2, str(profile)
                if profile_result.profile[profile.input_from[0]].local_dim != len(input_shapes[0]) - 1 and \
                    profile_result.profile[profile.input_from[1]].local_dim != len(input_shapes[1]) - 2:
                    profile.local_dim = last_local_dim
                    barrier = False
                else:
                    barrier = True
                    profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile}
            elif func.startswith((
                "softmax",
            )):
                args, kwargs = func_args
                if len(args) > 1:
                    dim = args[1]
                else:
                    dim = kwargs["dim"]
                if dim == -1:
                    dim = len(input_shapes[0]) - 1
                if last_local_dim is not None and dim != last_local_dim:
                    profile.local_dim = last_local_dim
                    barrier = False
                else:
                    profile.local_dim = None
                    barrier = True
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile}
            elif func.startswith(("linear")):
                # linear only applies to the last dim; if local dim is not the last dim, the locality remains
                if last_local_dim is not None and last_local_dim != len(input_shapes[0]) - 1:
                    barrier = False
                    profile.local_dim = last_local_dim
                else:
                    barrier = True
                    profile.local_dim = None
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": barrier, "profile": profile}
            elif func in ["shape", "dim"]:
                raise RuntimeError
            else:
                # Operation that does not support offloading.
                profile.hook_kwargs = {
                    "idx": idx, "conv": False,
                    "barrier": True, "profile": profile}
                profile.local_dim = None   # Operations that destroyed locality
            if not profile.origin_shapes and parent_profile\
                and parent_profile.origin_shapes:
                profile.origin_shapes = parent_profile.origin_shapes
            if profile.formula == [1., 0.] and parent_profile:
                profile.formula = parent_profile.formula
                profile.offset = parent_profile.offset
                profile.offset_factor = parent_profile.offset_factor
            if profile.output_shapes and profile.func_name.startswith((
                "conv", "batch_norm", "layer_norm")):
                assert int(start_input_shape[-1] * profile.formula[0] + profile.formula[1]) == \
                    profile.output_shapes[0][-1]
                
                


