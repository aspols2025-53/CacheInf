from typing import List, Tuple
import torch
import random
import numpy as np
import cv2
from numpy import inf
from numba import njit
from torchvision.transforms.functional import perspective as tvp
from taichi_ops.multiple_slices_to_index import multiple_slices_to_indices, merge_tensor

use_numba = False
if not use_numba:
    from numpy import int64 as int_
    bool_ = bool
else:
    from numba.types import bool_, int_


@njit
def reorganize_plan(bboxes: np.ndarray, upper_bound: np.ndarray,
               offset: int, formula_factor_0: float=1.0,
               formula_factor_1: float=0.0, max_num=100, dim_len=100):
    """final_h = whole_h * formula_factor_0 + formula_factor_1; the ending shape after ops such as conv
    """
    upper_h, upper_w = upper_bound
    num_bboxes = len(bboxes)
    # [[padded_base_h, padded_base_w, h, w, _valid_h, _valid_w]]
    bboxes_padded = np.zeros((max_num, 6), dtype=int_)
    bboxes_padded_num = num_bboxes
    space = np.zeros(max_num, dtype=int_)
    for i in range(num_bboxes):
        bbox_padded_base_h = max(bboxes[i, 0] - offset, 0)
        bbox_padded_base_w = max(bboxes[i, 1] - offset, 0)
        bbox_padded_upper_h = min(bboxes[i, 2] + offset, upper_h)
        bbox_padded_upper_w = min(bboxes[i, 3] + offset, upper_w)
        _h = bbox_padded_upper_h - bbox_padded_base_h
        _w = bbox_padded_upper_w - bbox_padded_base_w
        bboxes_padded[i] = [bbox_padded_base_h, bbox_padded_base_w,
                            _h, _w,
                            _h - offset, _w - offset]
        space[i] = _h * _w

    # dense_dim_len = dim_len   # align to square
    
    dense_dim_len = int(np.sqrt(space.sum())) + 10   # align to square
    # From large space to small space, slice and fill the bboxes into reorganized img
    slices = np.zeros((max_num, 3, 4), dtype=int_)  #[[slice_orig,slice_reorg,slice_orig_valid]]
    num_slices = 0
    empty_space = np.zeros((max_num, 4), dtype=int_)
    empty_space[0] = [0, 0, dense_dim_len, dense_dim_len]
    empty_space_num = 1
    whole_h = whole_w = dense_dim_len
    while bboxes_padded_num:
        largest_idx = np.argmax(space[:bboxes_padded_num])
        bbox_base_h, bbox_base_w, _h, _w, valid_h, valid_w = bboxes_padded[largest_idx]
        least_space_h = min(bbox_base_h + offset, upper_h) - bbox_base_h
        least_space_w = min(bbox_base_w + offset, upper_w) - bbox_base_w
        closest_idx = -1
        unused_space = inf
        for j in range(empty_space_num):
            _, _, space_h, space_w = empty_space[j]
            if _h <= space_h and _w <= space_w:
                _unused_space = space_h * space_w - _h * _w
            elif _h <= space_h and space_w > least_space_w + 10:
                _unused_space = space_w * (space_h - _h)
            elif _w <= space_w and space_h > least_space_h + 10: # _h > space_h
                _unused_space = space_h * (space_w - _w)
            elif space_w * space_h > 0.8 * _h * _w and\
                space_w > least_space_w + 10 and space_h > least_space_h + 10:   # _h > space_h and _w > space_w
                # find the largest space
                _unused_space = _h * _w - space_w * space_h
            else:
                _unused_space = inf
            if _unused_space < unused_space:
                closest_idx = j
                unused_space = _unused_space
        if closest_idx >= 0:
            space_base_h, space_base_w, space_h, space_w =  empty_space[closest_idx]
        else:   # Temporary add new empty space
            if whole_h >= _h >= 0.4 * whole_h:  # add space along width
                space_base_h, space_base_w = 0, whole_w
                space_h, space_w = whole_h, _w
                whole_w += _w
            elif whole_w >= _w >= 0.4 * whole_w:    # add space along height
                space_base_h, space_base_w = whole_h, 0
                space_h, space_w = _h, whole_w
                whole_h += _h
            else:
                # either _h, _w are too big or too small
                # create an empty space that holds over 70 % of _h*_w
                if _h >= _w:
                    space_base_h, space_base_w = 0, whole_w
                    space_h = whole_h
                    space_w = max(int(np.ceil(_h * _w / whole_h * 0.8)), least_space_w+5)
                    whole_w += space_w
                else:
                    space_base_h, space_base_w = whole_h, 0
                    space_h = max(int(np.ceil(_h * _w / whole_w * 0.8)), least_space_h+5)
                    space_w = whole_w
                    whole_h += space_h
            empty_space[empty_space_num] = [space_base_h, space_base_w, space_h, space_w]
            closest_idx = empty_space_num
            empty_space_num += 1
        if _h <= space_h and _w <= space_w:
            slices[num_slices] = np.array([[bbox_base_h, bbox_base_w, _h, _w],
                                  [bbox_base_h, bbox_base_w, valid_h, valid_w],
                                  [space_base_h, space_base_w, _h, _w]])
            bboxes_padded_num -= 1
            if bboxes_padded_num:
                bboxes_padded[largest_idx] = bboxes_padded[bboxes_padded_num]
                space[largest_idx] = space[bboxes_padded_num]
            if _h < space_h and _w < space_w:
                if space_h - _h >= space_w - _w:   # More h left, keep space_w
                    empty_space[closest_idx] = [space_base_h+_h, space_base_w, space_h - _h, space_w]
                    empty_space[empty_space_num] = [space_base_h, space_base_w+_w, _h, space_w-_w]
                else:                               # More w left, keep space h
                    empty_space[closest_idx] = [space_base_h, space_base_w+_w, space_h, space_w-_w]
                    empty_space[empty_space_num] = [space_base_h+_h, space_base_w, space_h - _h, _w]
                empty_space_num += 1
            elif _h < space_h:   # More h left
                empty_space[closest_idx] = [space_base_h+_h, space_base_w, space_h - _h, _w]
            elif _w < space_w:  # More w left
                empty_space[closest_idx] = [space_base_h, space_base_w+_w, space_h, space_w - _w]
            else:
                empty_space_num -= 1
                if empty_space_num:
                    empty_space[closest_idx] = empty_space[empty_space_num]
        elif _h <= space_h: # _w > space_w
            _valid_w = space_w - least_space_w
            slices[num_slices] = np.array([[bbox_base_h, bbox_base_w, _h, space_w],
                                  [bbox_base_h, bbox_base_w, valid_h, _valid_w],
                                  [space_base_h, space_base_w, _h, space_w]])
            bboxes_padded[largest_idx] = [
                bbox_base_h, bbox_base_w+_valid_w, _h, _w - _valid_w, valid_h, valid_w - _valid_w]
            space[largest_idx] = _h * (_w - _valid_w)
            if _h < space_h:
                empty_space[closest_idx] = [space_base_h+_h, space_base_w, space_h - _h, space_w]
            else:
                empty_space_num -= 1
                if empty_space_num:
                    empty_space[closest_idx] = empty_space[empty_space_num]
        elif _w <= space_w: # _h > space_h
            _valid_h = space_h - least_space_h
            slices[num_slices] = np.array([[bbox_base_h, bbox_base_w, space_h, _w],
                                  [bbox_base_h, bbox_base_w, _valid_h, valid_w],
                                  [space_base_h, space_base_w, space_h, _w]])
            bboxes_padded[largest_idx] = [
                bbox_base_h+_valid_h, bbox_base_w, _h-_valid_h, _w, valid_h-_valid_h, valid_w]
            space[largest_idx] = (_h-_valid_h) * _w
            if _w < space_w:
                empty_space[closest_idx] = [space_base_h, space_base_w+_w, space_h, space_w - _w]
            else:
                empty_space_num -= 1
                if empty_space_num:
                    empty_space[closest_idx] = empty_space[empty_space_num]
        else:   # _h > space_h and _w > space_h
            _valid_h = space_h - least_space_h
            _valid_w = space_w - least_space_w
            slices[num_slices] = np.array([[bbox_base_h, bbox_base_w, space_h, space_w],
                                  [bbox_base_h, bbox_base_w, _valid_h, _valid_w],
                                  [space_base_h, space_base_w, space_h, space_w]])
            bboxes_padded[largest_idx] = \
                [space_base_h+_valid_h, space_base_w, _h - _valid_h, _w, valid_h - _valid_h, valid_w]
            bboxes_padded[bboxes_padded_num] = \
                [space_base_h, space_base_w+_valid_w, space_h, _w - _valid_w, _valid_h, valid_w-_valid_w]
            space[largest_idx] = (_h - _valid_h) * _w
            space[bboxes_padded_num] = space_h *(_w - _valid_w)
            bboxes_padded_num += 1

            empty_space_num -= 1
            if empty_space_num:
                empty_space[closest_idx] = empty_space[empty_space_num]
        num_slices += 1
    slices = slices[: num_slices]
    slice_orig = slices[:, 0]
    slice_orig_valid = slices[:, 1]
    slice_reorg = slices[:, 2]
    for s in [slice_orig, slice_orig_valid, slice_reorg]:   # Turn to [h_start, w_start, h_end, w_end]
        s[:, 2] += s[:, 0]
        s[:, 3] += s[:, 1]
    whole_h = (slice_reorg[:, 2]).max()
    whole_w = (slice_reorg[:, 3]).max()
    # pad to ensure correctness after a sequence of conv ops
    final_h = int(np.ceil(whole_h * formula_factor_0 + formula_factor_1))
    final_w = int(np.ceil(whole_w * formula_factor_0 + formula_factor_1))
    whole_h = int((final_h - formula_factor_1) / formula_factor_0)
    whole_w = int((final_w - formula_factor_1) / formula_factor_0)
    return slice_orig, slice_orig_valid, slice_reorg, [whole_h, whole_w]


def move_along_w_slice(move_w_len, slice_orig, slice_orig_valid, slice_reorg, **_):
    moved_w_start = slice_reorg[:, 1] - move_w_len
    w_reduction = np.where(moved_w_start < 0, -moved_w_start, 0) - move_w_len
    slice_reorg[:, [1, 3]] += w_reduction
    valid_mask = slice_reorg[:, 3] > 0
    slice_orig[:, [1, 3]] += w_reduction
    slice_orig_valid[:, [1, 3]] += w_reduction
    return slice_orig[valid_mask], slice_orig_valid[valid_mask], slice_reorg[valid_mask]

def clip_along_w_slice(w_len, slice_orig, slice_orig_valid, slice_reorg, **_):
    valid_mask = slice_reorg[:, 1] < w_len
    slice_orig, slice_orig_valid, slice_reorg = \
        slice_orig[valid_mask], slice_orig_valid[valid_mask], slice_reorg[valid_mask]
    bound = np.ones(len(slice_orig)) * w_len
    slice_orig[:, 3] = np.min([slice_orig[:, 3], bound], 0)
    slice_orig_valid[:, 3] = np.min([slice_orig_valid[:, 3], bound], 0)
    slice_reorg[:, 3] = np.min([slice_reorg[:, 3], bound], 0)
    return slice_orig, slice_orig_valid, slice_reorg

def evolve_shape(shape, formula):
    h, w = shape[-2:]
    formula_factor_0, formula_factor_1 = formula
    final_h = int(h * formula_factor_0 + formula_factor_1)
    final_w = int(w * formula_factor_0 + formula_factor_1)
    new_shape = np.array(shape)
    new_shape[-2:] = [h, w]
    return new_shape

def pad_shapes_for_conv(shape, formula):
    h, w = shape[-2:]
    formula_factor_0, formula_factor_1 = formula
    final_h = int(h * formula_factor_0 + formula_factor_1)
    final_w = int(w * formula_factor_0 + formula_factor_1)
    if final_h == 0:
        h = int(np.ceil((1 - formula_factor_1) / formula_factor_0))
    if final_w == 0:
        w = int(np.ceil((1 - formula_factor_1) / formula_factor_0))
    h, w = h + h % 2, w + w % 2
    # h = int(np.ceil((final_h - formula_factor_1) / formula_factor_0))
    # w = int(np.ceil((final_w - formula_factor_1) / formula_factor_0))
    new_shape = np.array(shape, dtype=int)
    new_shape[-2:] = [h, w]
    return new_shape

def pad_dim_for_conv(dim_len, formula):
    formula_factor_0, formula_factor_1 = formula
    final_dim_len = int(dim_len * formula_factor_0 + formula_factor_1)
    if final_dim_len == 0:
        final_dim_len = int(np.ceil((1 - formula_factor_1) / formula_factor_0))
    return final_dim_len + final_dim_len % 2

def slices_to_final_len(slices: List[np.ndarray], formula):
    # s: [N, 4]
    formula_factor_0, formula_factor_1 = formula
    for i, s in enumerate(slices):
        s = (s * formula_factor_0).astype(int)
        s[:, 2] = np.max([s[:, 2], s[:, 0]+1], axis=0)
        s[:, 3] = np.max([s[:, 3], s[:, 1]+1], axis=0)
        slices[i] = s
    return slices

def slice_to_mesh_grid(s, device="cuda:0"):
    h_start, w_start, h_end, w_end = s
    h, w = torch.meshgrid(torch.arange(h_start, h_end, device=device), torch.arange(w_start, w_end, device=device), indexing="ij")
    return torch.vstack([h[None], w[None]])

def slice_to_indices_grid(shape, slice_reorg, slice_orig, device="cuda:0"):
    # debug
    assert shape[-2] > slice_reorg[:, 2].max()
    assert shape[-1] > slice_reorg[:, 3].max()
    indices_grid = torch.zeros(shape, dtype=torch.int32, device=device) # [B, 2, H, W]
    for _slice_orig, _slice_reorg in zip(slice_orig, slice_reorg):
        # _slice_reorg outside of 
        indices_grid[..., :, _slice_reorg[0]: _slice_reorg[2], _slice_reorg[1]: _slice_reorg[3]] = \
            slice_to_mesh_grid(_slice_orig)
    return indices_grid

# def multiple_slices_to_indices(slices: np.ndarray, h, w, device="cpu"):
#     h_indices, w_indices = torch.meshgrid(
#             torch.arange(h, device=device),
#             torch.arange(w, device=device), indexing="ij") # [h, w], [h, w]
#     all_indices = torch.stack([h_indices, w_indices], dim=-1).reshape(-1,2).unsqueeze(0) # [1, M, 2]
#     condition = torch.from_numpy(slices).to(device).unsqueeze(1)    # [N, 1, 4]
#     h_indices, w_indices = all_indices[..., 0], all_indices[..., 1]
#     mask = ((h_indices >= condition[..., 0]) & ( # [N, M]
#         h_indices < condition[..., 2]) & (
#             w_indices >= condition[..., 1]) & (
#                 w_indices < condition[..., 3]))  # [N, M]
#     return all_indices.expand(len(slices), all_indices.shape[1], 2)[mask] # [K, 2]

def multiple_slices_to_mask(slices: np.ndarray, h, w, device="cpu"):
    h_indices, w_indices = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device), indexing="ij") # [h, w], [h, w]
    condition = torch.from_numpy(slices).to(device).unsqueeze(1).unsqueeze(1)    # [N, 1, 4]
    h_indices = h_indices[None]
    w_indices = w_indices[None]
    mask = ((h_indices >= condition[..., 0]) & (
        h_indices < condition[..., 2]) & (
            w_indices >= condition[..., 1]) & (
                w_indices < condition[..., 3])) # [N, h, w]
    return mask  # [N, h, w]

# def reorganize_img(orig_img: torch.Tensor, slice_orig: np.ndarray,
#                    slice_reorg: np.ndarray, new_img_shape: Tuple[int, int]):
#     extra_shape = list(orig_img.shape[:-2])
#     new_img = torch.zeros(extra_shape + [new_img_shape[0],new_img_shape[1]],
#                           dtype=orig_img.dtype, device=orig_img.device)
#     reorg_mask = multiple_slices_to_mask(slice_reorg, *new_img_shape, device=orig_img.device)
#     orig_mask = multiple_slices_to_mask(slice_orig, *orig_img.shape[-2:], device=orig_img.device)
#     temp_new_img = new_img.unsqueeze(-3)
#     temp_orig_img = orig_img.unsqueeze(-3)
#     temp_new_img.expand(
#         *new_img.shape[:-2], len(slice_reorg), *new_img.shape[-2:])[..., reorg_mask] = \
#             temp_orig_img.expand(
#         *orig_img.shape[:-2], len(slice_orig), *orig_img.shape[-2:])[..., orig_mask]
#     return new_img

# def recover_img(reorg_img: torch.Tensor, orig_img: torch.Tensor=None, orig_shape=None,
#                 slice_orig_valid: np.ndarray=None,
#                 slice_reorg: np.ndarray=None, formula: List[float]=None, offset=0, **_):
#     device = reorg_img.device
#     if orig_img is None:
#         orig_img = torch.zeros(orig_shape, dtype=reorg_img.dtype, device=device)
#     slice_orig_valid, slice_reorg = slices_to_final_len([slice_orig_valid, slice_reorg], formula)
#     slice_reorg[:, [2,3]] = slice_reorg[:, [0,1]] + \
#         slice_orig_valid[:, [2,3]] - slice_orig_valid[:, [0,1]]
#     reorg_mask = multiple_slices_to_mask(slice_reorg, *reorg_img.shape[-2:], device=device)
#     orig_valid_mask = multiple_slices_to_mask(
#         slice_orig_valid, *orig_img.shape[-2:], device=device)

#     temp_orig_img = orig_img.unsqueeze(-3)
#     temp_reorg_img = reorg_img.unsqueeze(-3)
#     temp_orig_img.expand(
#         *orig_img.shape[:-2], len(slice_reorg), *orig_img.shape[-2:])[..., orig_valid_mask] = \
#         temp_reorg_img.expand(
#             *reorg_img.shape[:-2], len(slice_reorg), *reorg_img.shape[-2:])[..., reorg_mask]
#     return orig_img


def reorganize_img(orig_img: torch.Tensor, slice_orig: np.ndarray,
                   slice_reorg: np.ndarray, reorg_img_shape: Tuple[int, int]):
    reorg_img = torch.zeros(
        list(orig_img.shape[:-2])+reorg_img_shape, device=orig_img.device, dtype=orig_img.dtype)
    merge_tensor(orig_img, reorg_img, slice_orig, slice_reorg)
    return reorg_img

def recover_img(reorg_img: torch.Tensor, orig_img: torch.Tensor=None, orig_shape=None,
                slice_orig_valid: np.ndarray=None,
                slice_reorg: np.ndarray=None, formula: List[float]=None, **_):
    device = reorg_img.device
    if orig_img is None:
        orig_img = torch.zeros(orig_shape, dtype=reorg_img.dtype, device=device)
    # slice_orig_valid, slice_reorg = slices_to_final_len([slice_orig_valid, slice_reorg], formula)
    np.multiply(slice_orig_valid, formula[0], out=slice_orig_valid, casting='unsafe')
    np.multiply(slice_reorg, formula[0], out=slice_reorg, casting='unsafe')
    merge_tensor(reorg_img, orig_img, slice_reorg, slice_orig_valid)
    return orig_img

def draw_vertical_line(img, start_h, end_h, w, val=[255, 0, 0], thickness=2):
    img[start_h:end_h, w:w+thickness] = val

def draw_horizon_line(img, start_w, end_w, h, val=[255, 0, 0], thickness=2):
    img[h:h+thickness, start_w:end_w] = val

def draw_bbox(img, bbox, val=[255, 0, 0]):
    start_h, start_w, end_h, end_w = bbox
    draw_vertical_line(img, start_h, end_h+1, start_w, val)
    draw_vertical_line(img, start_h, end_h+1, end_w, val)
    draw_horizon_line(img, start_w, end_w, start_h, val)
    draw_horizon_line(img, start_w, end_w, end_h, val)

def compare_origin_and_reorg(orig_img: torch.Tensor, reorg_img: torch.Tensor,
              slice_orig: List[List], slice_orig_valid: List[List],
              slice_reorg: List[List], stride=1, formula=[1., 0.], **_):
    ...
    if orig_img.shape[-3] > 3:
        orig_img = orig_img[..., :3, :, :]
        reorg_img = reorg_img[..., :3, :, :]
    tallest_h = max(orig_img.shape[-2], reorg_img.shape[-2]) + 10
    total_img_w = orig_img.shape[-1] + reorg_img.shape[-1] + 80
    offset_reorg_h = (tallest_h - reorg_img.shape[-2]) // 2
    offset_reorg_w = total_img_w - reorg_img.shape[-1] - 3
    img = np.zeros([tallest_h, total_img_w, 4])
    img[:] = 0
    img[0:orig_img.shape[-2], 0:orig_img.shape[-1], :3] = torch.moveaxis(orig_img, -3, -1).cpu().numpy()
    img[0:orig_img.shape[-2], 0:orig_img.shape[-1], 3] = 255
    img[offset_reorg_h:reorg_img.shape[-2]+offset_reorg_h, offset_reorg_w:offset_reorg_w+reorg_img.shape[-1], : 3] = torch.moveaxis(
        reorg_img, -3, -1).cpu().numpy()
    img[offset_reorg_h:reorg_img.shape[-2]+offset_reorg_h, offset_reorg_w:offset_reorg_w+reorg_img.shape[-1], 3] = 255
    slice_orig, slice_orig_valid, slice_reorg = slices_to_final_len([slice_orig, slice_orig_valid, slice_reorg], formula)
    for _slice_orig, _slice_orig_valid, _slice_reorg in zip(slice_orig, slice_orig_valid, slice_reorg):
        orig_base_h, orig_base_w, orig_end_h_valid, orig_end_w_valid = _slice_orig_valid
        _, _, orig_end_h, orig_end_w = _slice_orig
        reorg_base_h, reorg_base_w, reorg_end_h, reorg_end_w = _slice_reorg
        reorg_base_h += offset_reorg_h
        reorg_end_h += offset_reorg_h
        reorg_base_w += offset_reorg_w
        reorg_end_w += offset_reorg_w

        # draw_bbox(img, [orig_base_h, orig_base_w, orig_end_h_valid, orig_end_w_valid], [0, 255, 0])
        draw_bbox(img, [orig_base_h, orig_base_w, orig_end_h, orig_end_w], [255, 0, 0, 255])
        draw_bbox(img, [reorg_base_h, reorg_base_w, reorg_end_h, reorg_end_w], [255, 0, 0, 255])
        # Connect two bboxes
        cv2.line(img, ((orig_base_w+orig_end_w)//2, (orig_base_h+orig_end_h)//2,),
                 ((reorg_base_w+reorg_end_w)//2, (reorg_base_h+reorg_end_h)//2), [0,0,255, 255], thickness=2)
    return img.astype(np.uint8)[:-10]

def interpolate(feature_map: torch.Tensor, M: list, last_h, last_w):
    """interpolate feature map according to matching corners

    Args:
        feature_map (torch.Tensor): _description_

    Returns:
        torch.Tensor: interpolated feature map that matches current img
    """
    # feature_map: [B, C, H, W]
    h, w = feature_map.shape[-2:]
    # [top-left, top-right, bottom-right, bottom-left]
    M = np.array(self.last_frame["M"])
    M[2] /= last_w / w
    M[5] /= last_h / h
    return F_t.perspective(feature_map, M, NEAREST.value, 0.)