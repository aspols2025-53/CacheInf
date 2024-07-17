from collections import OrderedDict
from typing import Dict, List, Union, Callable, Tuple
import torch
import numpy as np
import warnings
import copy
from ._utils import iterate_tensor, iterate_tensor_with_reference, args_kwargs_to_args
warnings.filterwarnings("ignore")


class InputSlot:
    """Handler of func args / intermediates
    """
    def __init__(self, idx: int, container: Union[List, Dict], index: Union[int, str]) -> None:
        self.idx = idx  # idx that this input slot belongs to
        self.container = container  # container to access the tensor
        self.index = index          # index to access the tensor from container

    def empty(self):
        self.container[self.index] = None

    def fill(self, val: torch.Tensor):
        self.container[self.index] = val

    @property
    def tensor(self)->torch.Tensor:
        return self.container[self.index]

class TorchOPProfile:
    def __init__(self, *, idx: int, func_name: str,
                 func, keep: bool, func_args: Tuple[List[torch.Tensor], dict],
                 num_output: Union[int, None], output_dtypes, output_devices,
                 input_shapes: list, input_size: int,
                 output_shapes: list, output_size: int,
                 input_ids: list, output_ids: list, input_slots: List[InputSlot],
                 hook_kwargs: dict=None) -> None:
        self.idx = idx
        self.func = func
        self.keep = keep
        self.output_dtypes: List[torch.dtype] = output_dtypes
        self.output_devices: List[torch.device] = output_devices

        # function args and kwargs at exec (intermediates already emptied)
        self.func_args: Tuple[List[torch.Tensor], dict] = func_args
        self.cached_output: List[torch.Tensor] = None
        self.cached_offset = 0
        self.cached_formula = [1., 0.]

        self.func_name = func_name
        self.num_output = num_output
        self.input_shapes = input_shapes # Bytes
        self.input_size = input_size
        self.input_ids = input_ids
        self.output_shapes = output_shapes # Bytes
        self.output_size = output_size
        self.output_ids = output_ids
        self.input_slots = input_slots
        self.local_dim = None
        self.origin_shapes = []
        self.current_dense_dim_len = 0
        self.hook_kwargs = hook_kwargs
        self.ops_time = 0.
        from .profile_ops import OpsStamp
        self.ops_stamp: OpsStamp = None
        self.input_from = []
        self.output_to = []
        self.input_idx = []
        self.output_idx_slots: Dict[int, List[InputSlot]] = OrderedDict()
        self.caching = False
        self.masked = False
        self.img: np.ndarray = None
        self.conv = False
        self.unconv = False
        self.kernel_size = 0
        self.stride = 1
        self.padding = 0
        self.offset = 0
        self.formula = [1., 0.]
        self.offset_factor = 1.

    def get_tensor_args_kwargs(self):
        args_kwargs = []
        for shape, dtype, device in zip(self.output_shapes,
                                        self.output_dtypes, self.output_devices):
            args_kwargs.append([[shape], {"dtype": dtype, "device": device}])
        return args_kwargs

    def copy_for_transmission(self):
        """Clear the unpicklable objects: func_args, func and input slots"""
        ret = TorchOPProfile(idx=self.idx, func_name=self.func_name, func=None,
                             keep=self.keep, func_args=[],
                             num_output=self.num_output, input_shapes=self.input_shapes,
                             output_dtypes=self.output_dtypes,
                             output_devices=[],
                            input_size=self.input_size, output_shapes=self.output_shapes,
                            output_size=self.output_size, input_ids=self.input_ids,
                            output_ids=self.output_ids, input_slots=None, hook_kwargs=None)
        ret.input_from = self.input_from
        ret.ops_time = self.ops_time
        ret.ops_stamp = self.ops_stamp
        ret.input_from = self.input_from
        ret.output_to = self.output_to
        ret.masked = self.masked
        ret.hook_kwargs = {}
        ret.hook_kwargs.update(self.hook_kwargs)
        if "profile" in ret.hook_kwargs:
            del ret.hook_kwargs["profile"]
        return ret

    def __repr__(self) -> str:
        return f"{self.idx} {self.func_name}: input_from: {self.input_from}, output_to: {self.output_to}, output_shapes: {self.output_shapes}, barrier: {self.hook_kwargs['barrier']}, local dim: {self.local_dim}; "


class OffloadProfile:
    def __init__(self) -> None:
        """Changed from dataclass to regular class since pickle seems unable to handle dataclass"""
        self.idx: int = 0
        self.end_idx = 0
        self.local_comp_time = 0.
        self.profile: Dict[int, TorchOPProfile] = OrderedDict()
        self.ignore_ops = []
        self.ret_store = None
        self.ret_slots: List[InputSlot] = []
        self.partial_ops_time: np.ndarray = None
        self.size_to_loads_time: np.poly1d = None
        self.size_to_dumps_time: np.poly1d = None
        self.exec_plan: Dict[int, List[Tuple[TorchOPProfile, int, bool]]] = OrderedDict()

    def __getitem__(self, index: Union[int, str]):
        return getattr(self, index)

    def __setitem__(self, index: Union[int, str], val):
        setattr(self, index, val)


    def copy_for_transmission(self):
        new_profile = OffloadProfile()
        new_profile.idx = self.idx
        new_profile.end_idx = self.end_idx
        new_profile.partial_ops_time = self.partial_ops_time
        new_profile.profile = OrderedDict()
        for idx, p in self.profile.items():
            new_profile.profile[idx] = p.copy_for_transmission()
        new_profile.size_to_loads_time = self.size_to_loads_time
        new_profile.size_to_dumps_time = self.size_to_dumps_time
        new_profile.local_comp_time = self.local_comp_time
        return new_profile


def get_ops(profile_result: OffloadProfile):
    ops = []
    all_profiles = profile_result.profile
    idx_array = list(all_profiles.keys())
    for idx in idx_array:
        profile: TorchOPProfile = all_profiles[idx]
        ops.append([profile.idx, profile.ops_time, profile.output_size])
    return np.array(ops)

def get_dependency(profile_result: OffloadProfile):
    ops = []
    all_profiles = profile_result.profile
    idx_array = list(all_profiles.keys())
    for idx in idx_array:
        profile: TorchOPProfile = all_profiles[idx]
        ops.append([profile.idx, profile.input_from, profile.output_to])
    return ops

def return_as_is(*args, **kwargs):
    return args, kwargs

keep_funcs = {"__setitem__", "_start", "_end"}
skip_funcs = {"__get__", "dim", "size"}
def profile_tensor_factory(profile_result: OffloadProfile, hook_level: List, debug=False):
    class ProfileTensor(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            idx = profile_result.idx
            if kwargs is None:
                kwargs = {}
            if func is not None:
                ret = super().__torch_function__(func, types, args, kwargs)
                func_name = func.__name__
            else:
                if idx == 0:
                    func_name = "_start"
                    ret = [args, kwargs]
                else:
                    func_name = "_end"
                    assert len(kwargs) == 0
                    ret = args[0]
                func = return_as_is
            # Avoid recursive call or profiling non-computation functions.
            if hook_level[0]:
                return ret
            hook_level[0] += 1

            outputs = []
            input_shapes = []
            input_size = [0]
            input_ids = []
            output_shapes = []
            output_dtypes = []
            output_devices = []
            output_size = [0]
            output_ids = []
            input_slots = []
            def profile_input(arg: ProfileTensor,
                            container: Union[List, Dict], index: Union[int, str]):
                input_shapes.append(arg.shape)
                input_size[0] += arg.nelement() * arg.element_size()
                _id = id(arg)
                input_ids.append(_id)

                # Handler to fill and modify intermediates.
                assert container is not None
                input_slots.append(InputSlot(idx, container, index))
                return arg.as_subclass(torch.Tensor)[:0]
            store_func_args = iterate_tensor_with_reference(
                [args, kwargs], profile_input, ProfileTensor, None, None)

            def profile_output(_ret: ProfileTensor):
                output_shapes.append(_ret.shape)
                output_size[0] += _ret.nelement()*_ret.element_size()
                outputs.append(True)
                output_dtypes.append(_ret.dtype)
                output_devices.append(_ret.device)
                _id = id(_ret)
                output_ids.append(_id)
                return _ret
            ret = iterate_tensor(ret, profile_output, ProfileTensor)
            if func_name == "_end":
                # A place to hold output of _end op.
                def profile_forward_ret(_ret: ProfileTensor, container, index):
                    profile_result.ret_slots.append(InputSlot(idx, container, index))
                    return _ret.as_subclass(torch.Tensor)
                profile_result.ret_store = iterate_tensor_with_reference(
                    ret, profile_forward_ret, ProfileTensor, profile_result, "ret_store")
            if len(output_ids) == 0:
                assert func_name in keep_funcs or func_name in skip_funcs, \
                    f"op {idx} {func_name} has no output and not handled by us."
            num_output = len(outputs) if len(outputs) != 1 else None

            profile_result.profile[idx] = TorchOPProfile(
                idx=idx,
                func_name=func_name,
                func=func,
                keep=func_name in keep_funcs,
                func_args=store_func_args,
                num_output=num_output,
                output_dtypes=output_dtypes,
                output_devices=output_devices,
                input_shapes=input_shapes,
                input_size=input_size[0],
                output_shapes=output_shapes,
                output_size=output_size[0],
                input_ids=input_ids,
                output_ids=output_ids,
                input_slots=input_slots
            )
            if False:    # For debug
                profile_result.profile[idx].store_ret = iterate_tensor(
                    ret, lambda x: copy.deepcopy(x.as_subclass(torch.Tensor)), ProfileTensor)
            profile_result.idx += 1
            hook_level[0] -= 1
            return ret
    return ProfileTensor

def offload_tensor_factory(hook_func, profile_result: OffloadProfile, hook_level: List,
                           send_queue=None, recv_queue=None):
    profile = profile_result.profile
    class OffloadTensor(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            ret = super().__torch_function__(func, types, args, kwargs)
            if hook_level[0]:  # Avoid recursive call
                return ret
            hook_level[0] += 1

            idx = profile_result.idx
            offload_kwargs = profile[idx].hook_kwargs
            assert func.__name__ == profile[idx].func_name, f"{idx} {func.__name__} != {profile[idx].func_name}"  # TODO remove at deployment
            ret = hook_func(ret, send_queue=send_queue,
                            recv_queue=recv_queue, **offload_kwargs)
            profile_result.idx += 1
            hook_level[0] -= 1
            return ret
    return OffloadTensor

