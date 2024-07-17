import taichi as ti
import torch
import numpy as np

if torch.cuda.is_available():
    ti.init(arch=ti.gpu)
    device = torch.device("cuda:0")
else:
    ti.init(arch=ti.cpu)
    device = torch.device("cpu")

def multiple_slices_to_indices(slices: np.ndarray):
    slices_base = slices[:, [0,1]]
    slices_end = slices[:, [2,3]]
    slices_len = slices_end - slices_base
    spaces = np.prod(slices_len, axis=-1)
    total_num = spaces.sum()
    largest_space = spaces.max()
    index_base = np.roll(np.cumsum(spaces), 1)
    index_base[0] = 0
    all_indices = torch.zeros([total_num, 2], device=device, dtype=torch.int64)
    _multiple_slices_to_indices(all_indices, largest_space, len(slices), slices_len,
                                slices_base, index_base)
    ti.sync()
    return all_indices
    

@ti.kernel
def _multiple_slices_to_indices(
    all_indices: ti.types.ndarray(dtype=ti.int64),
    largest_space: ti.int32,
    slices_num: ti.int32,
    slices_h_w: ti.types.ndarray(dtype=ti.int64),
    slices_base: ti.types.ndarray(dtype=ti.int64),
    slices_indices_base: ti.types.ndarray(dtype=ti.int64)
):
    for i, j in ti.ndrange(slices_num, largest_space):
        h = slices_h_w[i, 0]
        w = slices_h_w[i, 1]
        if j < h*w:
            idx = j + slices_indices_base[i]
            all_indices[idx, 0] = slices_base[i, 0] + j // w
            all_indices[idx, 1] = slices_base[i, 1] + j % w

def merge_tensor(src: torch.Tensor, dst: torch.Tensor, slices_src, slices_dst):
    src_slice_base = slices_src[:, [0,1]]
    dst_slice_base = slices_dst[:, [0,1]]
    dst_slice_end = slices_dst[:, [2,3]]
    h_w = dst_slice_end - dst_slice_base
    largest_space = np.prod(h_w, axis=-1).max()
    if len(src.shape) == 3:
        func = _merge_image_2D_no_batch
    elif len(src.shape) == 4: # [B,C,H,W]
        func = _merge_image_2D
    elif len(src.shape) == 5: # [B,C,D,H,W]
        func = _merge_image_3D
    func(src, dst, largest_space, len(slices_src), h_w, src_slice_base, dst_slice_base)
    ti.sync()

@ti.kernel
def _merge_image_2D_no_batch(
    src_img: ti.types.ndarray(dtype=ti.float32),
    dst_img: ti.types.ndarray(dtype=ti.float32),
    largest_space: ti.int32,
    slices_num: ti.int32,

    slices_h_w: ti.types.ndarray(dtype=ti.int64),
    src_slices_base: ti.types.ndarray(dtype=ti.int64),
    dst_slices_base: ti.types.ndarray(dtype=ti.int64),
):
    channel_num = src_img.shape[0]
    for c, i, _j in ti.ndrange(channel_num, slices_num, largest_space//4):
        h = slices_h_w[i, 0]
        w = slices_h_w[i, 1]
        for __j in ti.static(range(4)):
            j = _j * 4 + __j
            if j < h*w:
                i_h = j // w
                i_w = j % w
                dst_img[c, dst_slices_base[i, 0]+i_h, dst_slices_base[i, 1]+i_w] = \
                    src_img[c, src_slices_base[i, 0]+i_h, src_slices_base[i, 1]+i_w]

@ti.kernel
def _merge_image_2D(
    src_img: ti.types.ndarray(dtype=ti.float32),
    dst_img: ti.types.ndarray(dtype=ti.float32),
    largest_space: ti.int32,
    slices_num: ti.int32,

    slices_h_w: ti.types.ndarray(dtype=ti.int64),
    src_slices_base: ti.types.ndarray(dtype=ti.int64),
    dst_slices_base: ti.types.ndarray(dtype=ti.int64),
):
    b_num = src_img.shape[0]
    channel_num = src_img.shape[1]
    for b, c, i, j in ti.ndrange(b_num, channel_num, slices_num, largest_space):
        h = slices_h_w[i, 0]
        w = slices_h_w[i, 1]
        if j < h*w:
            i_h = j // w
            i_w = j % w
            dst_img[b, c, dst_slices_base[i, 0]+i_h, dst_slices_base[i, 1]+i_w] = \
                src_img[b, c, src_slices_base[i, 0]+i_h, src_slices_base[i, 1]+i_w]

@ti.kernel
def _merge_image_3D(
    src_img: ti.types.ndarray(dtype=ti.float32),
    dst_img: ti.types.ndarray(dtype=ti.float32),
    largest_space: ti.int32,
    slices_num: ti.int32,
    slices_h_w: ti.types.ndarray(dtype=ti.int64),

    src_slices_base: ti.types.ndarray(dtype=ti.int64),
    dst_slices_base: ti.types.ndarray(dtype=ti.int64),
):
    b_num = src_img.shape[0]
    channel_num = src_img.shape[1]
    depth_num = src_img.shape[2]
    for b, c, d, i, j in ti.ndrange(b_num, channel_num, depth_num, slices_num, largest_space):
        h = slices_h_w[i, 0]
        w = slices_h_w[i, 1]
        if j < h*w:
            i_h = j // w
            i_w = j % w
            dst_img[b, c, d, dst_slices_base[i, 0]+i_h, dst_slices_base[i, 1]+i_w] = \
                src_img[b, c, d, src_slices_base[i, 0]+i_h, src_slices_base[i, 1]+i_w]

CLUSTER_NUM = 5

def assignment_to_bboxes(coord: torch.Tensor, assignment: np.ndarray):
    bboxes = np.zeros([CLUSTER_NUM, 4], dtype=np.int32)
    _assignment_to_bboxes(coord.type(torch.int32).contiguous(), assignment, bboxes)
    ti.sync()
    return bboxes

_bbox_f = ti.types.matrix(CLUSTER_NUM, 4, ti.i32)
@ti.kernel
def _assignment_to_bboxes(
    coord: ti.types.ndarray(dtype=ti.i32),
    assignment: ti.types.ndarray(dtype=ti.int32),
    bboxes: ti.types.ndarray(dtype=ti.i32)
):
    _bbox = _bbox_f([0xffffff, 0xffffff, 0, 0]*CLUSTER_NUM)
    for i in range(coord.shape[0]):
        idx = assignment[i]
        ti.atomic_min(_bbox[idx, 0], coord[i, 0])
        ti.atomic_min(_bbox[idx, 1], coord[i, 1])
        ti.atomic_max(_bbox[idx, 2], coord[i, 0]+1)
        ti.atomic_max(_bbox[idx, 3], coord[i, 1]+1)
    for b in ti.grouped(bboxes):
        bboxes[b] = _bbox[b]

if __name__ == "__main__":
    slices = np.array([[1,1, 3, 3],
                        [2,2, 5, 5],
                        [4, 3, 6, 8]])
    indices = multiple_slices_to_indices(slices)
    true_indices = []
    for h_start, w_start, h_end, w_end in slices:
        _indices = torch.stack(torch.meshgrid([
            torch.arange(h_start, h_end, device=device),
            torch.arange(w_start, w_end, device=device)], indexing="ij"), dim=-1).reshape(-1, 2)
        true_indices.append(_indices)
    true_indices = torch.vstack(true_indices)
    assert torch.allclose(indices, true_indices)