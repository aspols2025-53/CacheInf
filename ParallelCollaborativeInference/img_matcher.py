# Attempt to implement feature map warp
from typing import List
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.functional import F_t
import matplotlib.pyplot as plt
NEAREST = InterpolationMode.NEAREST
BILINEAR = InterpolationMode.BILINEAR

import cv2
import numpy as np
import torch
from torch import nan
from libKMCUDA import kmeans_cuda
import ctypes
import cupy as cp
import time
import warnings
warnings.filterwarnings("ignore")
from reorganize_input import reorganize_plan, reorganize_img, recover_img, compare_origin_and_reorg
from taichi_ops.multiple_slices_to_index import assignment_to_bboxes, CLUSTER_NUM
from _utils import log_dur


# install via
# CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda CUDA_ARCH=75 pip install git+https://github.com/src-d/kmcuda.git#subdirectory=src

MIN_NUM_GOOD_MATCHES = 20
MAX_NUM_GOOD_MATCHES = 30
CLUSTER_DTYPE = torch.float32
CUPY_DTYPE = cp.float32
CLUSTER_BYTES_NUM = 4

def to_data_ptr(t: torch.Tensor, shape):
    return t.data_ptr(), 0, shape

def ptr_to_tensor(device_ptr: int, nbytes: int, shape: tuple, dtype):
    mem = cp.cuda.UnownedMemory(device_ptr, nbytes, owner=None)
    memptr = cp.cuda.MemoryPointer(mem, offset=0)
    arr = cp.ndarray(shape, dtype=dtype, memptr=memptr)
    return torch.as_tensor(arr, device="cuda")

class ImageMatcher:
    '''https://blog.csdn.net/qq_45832961/article/details/122776322
    '''
    def __init__(self, debug=False) -> None:
        self.debug = debug
        self.detector = cv2.AKAZE_create()
        # 获取flann匹配器
        FLANN_INDEX_KDTREE = 0
        # 参数1：indexParams
        #    对于SIFT和SURF，可以传入参数index_params=dict(algorithm=FLANN_INDEX_KDTREE, trees=5)。
        #    对于ORB，可以传入参数index_params=dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12）。
        indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # 参数2：searchParams 指定递归遍历的次数，值越高结果越准确，但是消耗的时间也越多。
        searchParams = dict(checks=50)

        # 使用FlannBasedMatcher 寻找最近邻近似匹配
        self.matcher = cv2.FlannBasedMatcher(indexParams, searchParams)

        self.reorganize_info = {}
        
        self.adaptive_avg_layer: torch.nn.AdaptiveAvgPool2d = None
        self.bbox_cof: tuple = None

    def match(self, feature_map_src: torch.Tensor,
              feature_map_dst: torch.Tensor, min_diff=100):
        # [B,C,H,W]
        h, w = feature_map_src.shape[-2:]
        if self.adaptive_avg_layer is None:
            CLUSTER_SAMPLE_SIZE_H = int(np.ceil(h/40))
            CLUSTER_SAMPLE_SIZE_W = int(np.ceil(w/40))
            self.adaptive_avg_layer = torch.nn.AdaptiveAvgPool2d([CLUSTER_SAMPLE_SIZE_H, CLUSTER_SAMPLE_SIZE_W])
            self.bbox_cof = (h/CLUSTER_SAMPLE_SIZE_H, w/CLUSTER_SAMPLE_SIZE_W)
        all_min = min(feature_map_src.min(), feature_map_dst.min())
        all_max = max(feature_map_src.max(), feature_map_dst.max())
        aligned_src_torch = ((feature_map_src - all_min) / all_max * 255)
        aligned_dst_torch = ((feature_map_dst - all_min) / all_max * 255)
        aligned_src = aligned_src_torch.type(
            torch.uint8)[0, :3].permute(1,2,0).cpu().numpy()  # [H,W,C]
        aligned_dst = aligned_dst_torch.type(
            torch.uint8)[0, :3].permute(1,2,0).cpu().numpy()  # [H,W,C]

        kp_src, des_src = self.detector.detectAndCompute(aligned_src, None)
        kp_dst, des_dst = self.detector.detectAndCompute(aligned_dst, None)
        matches = self.matcher.knnMatch(des_src.astype(np.float32),
                                        des_dst.astype(np.float32), k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good_matches) >= MIN_NUM_GOOD_MATCHES:
            src_pts = np.float32(
                [kp_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp_dst[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            M = np.linalg.inv(M).ravel()[:-1]
            # M = M.ravel()[:-1]

            feature_map_src_inter = self.interpolate(feature_map_src, M, h, w)
            feature_map_src_inter = ((feature_map_src_inter - all_min) / all_max * 255)
            diff = torch.abs(self.adaptive_avg_layer(aligned_dst_torch
                                                     - feature_map_src_inter)).mean(1)[0]
            plt_torch(diff, save="diff.jpg", min_val=0, max_val=255)
            different_points: torch.Tensor = torch.nonzero(diff>min_diff) # [N,2]
            if self.debug:
                interpolated_dst = feature_map_src_inter.type(
                    torch.uint8)[0, :3].permute(1,2,0).cpu().numpy()
                plt.imsave("origin.jpg", aligned_src)
                plt.imsave("dst.jpg", aligned_dst)
                plt.imsave("interpolated_dst.jpg", interpolated_dst)
                all_img = np.hstack([aligned_src, aligned_dst, interpolated_dst])
                plt.imshow(all_img)
                plt.show()
            return {"M": M, "h": h, "w": w}, different_points
        return None, None

    def interpolate(self, feature_map: torch.Tensor, M, h, w, **_):
        """interpolate feature map according to matching corners

        Args:
            feature_map (torch.Tensor): _description_

        Returns:
            torch.Tensor: interpolated feature map that matches current img
        """
        # feature_map: [B, C, H, W]
        c_h, c_w = feature_map.shape[-2:]
        # [top-left, top-right, bottom-right, bottom-left]
        M = np.array(M)
        M[2] /= w / c_w
        M[5] /= h / c_h
        return F_t.perspective(feature_map, M, NEAREST.value, 0.)

    def reorganize_uncached_zones(self, 
            last_feature_map: torch.Tensor,
            current_feature_map: torch.Tensor, min_diff=50, offset=0, formula=[1., 0.], max_ratio=0.6):
        """interpolate last feature map; compare it with the current;
         reorganize the unmatched zones of the current into a new input

        Args:
            last_feature_map (torch.Tensor): _description_
            current_feature_map (torch.Tensor): _description_
            min_diff (float): different larger than min_diff will be considered unmatched;
                    its value depends on different input

        Returns:
            torch.Tensor: _description_
        """
        # feature_map: [B, C, H, W]
        match_info, different_points = self.match(last_feature_map, current_feature_map,
                                                  min_diff=min_diff)
        orig_size = np.prod(current_feature_map.shape[-2:])
        # last_feature_map = self.interpolate(last_feature_map)
        # diff = torch.abs(self.adaptive_avg_layer(
        #     last_feature_map[0]) - self.adaptive_avg_layer(current_feature_map[0])).max(-3)[0]
        # different_points: torch.Tensor = torch.nonzero(diff>min_diff) # [N,2]
        if match_info is None or len(different_points) >= max_ratio * orig_size:
            self.reorganize_info = None
            return current_feature_map, None
        if (l:=len(different_points)) > 0:
            if l > CLUSTER_NUM:
                # different_points = different_points[l%2:]   # Number must be even
                _different_points = different_points.type(CLUSTER_DTYPE).ravel().contiguous()
                centroids, assignments = kmeans_cuda(
                    to_data_ptr(_different_points, different_points.shape), CLUSTER_NUM)
                centroids = ptr_to_tensor(centroids, CLUSTER_NUM*2*CLUSTER_BYTES_NUM,
                                        (CLUSTER_NUM, 2), dtype=CUPY_DTYPE)
                assignments = ptr_to_tensor(assignments, len(different_points)*4,
                                            len(different_points), dtype=cp.int32)
                if self.debug:
                    from matplotlib import pyplot
                    fig = plt.figure()
                    pyplot.scatter(different_points[:, 1].cpu().numpy(),
                                different_points[:, 0].cpu().numpy(), c=assignments.cpu().numpy())
                    pyplot.scatter(centroids[:, 1].cpu().numpy(), centroids[:, 0].cpu().numpy(),    c="white", s=150)
                    pyplot.show()
            else:
                assignments = torch.arange(len(different_points), device=different_points.device, dtype=torch.int32)
            bboxes = assignment_to_bboxes(different_points, assignments)
            bboxes[:, 0::2] = np.around(bboxes[:, 0::2] * self.bbox_cof[0]).astype(int)
            bboxes[:, 1::2] = np.around(bboxes[:, 1::2] * self.bbox_cof[1]).astype(int)
            bboxes = np.unique(bboxes, axis=0)
            slice_orig, slice_orig_valid, slice_reorg, reorg_shape = reorganize_plan(
                bboxes, np.array(current_feature_map.shape[-2:]), offset, formula[0], formula[1],
                dim_len=max(last_feature_map.shape[-2:]))
            self.reorganize_info = {"slice_orig": slice_orig, 
                                    "slice_orig_valid": slice_orig_valid,
                                    "slice_reorg": slice_reorg,
                                    "reorg_shape": reorg_shape,
                                    "offset": offset, 
                                    "M": match_info["M"],
                                    "h": match_info["h"], "w": match_info["w"]}
            return reorganize_img(current_feature_map, slice_orig, slice_reorg, reorg_shape),\
                self.reorganize_info
        else:
            self.reorganize_info = None
            return current_feature_map[:0], None

    def recover_feature_map(self, last_feature_map: torch.Tensor,
                            reorg_result: torch.Tensor, formula: List[float],
                            min_val=None, max_val=None):
        if self.reorganize_info:
            curr_feature_map = self.interpolate(last_feature_map, **self.reorganize_info)
            plt_torch(curr_feature_map, "interpolated_last_inferred.jpg", min_val, max_val)
            return recover_img(reorg_result, curr_feature_map, formula=formula, **self.reorganize_info)
        return last_feature_map

@torch.no_grad()
def test():
    import torchvision
    import os
    debug = True
    work = os.environ["work"]
    path = f"/home/guanxiux/project/ParallelCollaborativeInference/data/DAVIS/JPEGImages/480p/goat/00000.jpg"
    whole_img = cv2.imread(path)[..., [2,1,0]][None]
    print(whole_img.shape)
    img_h = 300
    img_w = 400
    offset_h = 60
    offset_w = 30
    matcher = ImageMatcher(debug=debug)
    img = whole_img
    last_feature_map_0 = torch.from_numpy(img).moveaxis(-1, -3).type(torch.float32).to("cuda:0").contiguous()
    import copy
    path2 = "/home/guanxiux/project/ParallelCollaborativeInference/data/DAVIS/JPEGImages/480p/goat/00003.jpg"
    new_img = cv2.imread(path2)[..., [2,1,0]][None]
    current_feature_map_0 = torch.from_numpy(new_img).moveaxis(-1, -3).type(torch.float32).to("cuda:0").contiguous()
    with log_dur(prefix="match"):
        reorg_feature_map, reorg_info = matcher.reorganize_uncached_zones(last_feature_map_0, current_feature_map_0, offset=37)
    img = compare_origin_and_reorg(current_feature_map_0, reorg_feature_map, **matcher.reorganize_info)
    if debug:
        plt.figure()
        plt.imshow(img)
        plt.show()
    recovered_current_feature_map_0 = matcher.recover_feature_map(last_feature_map_0, reorg_feature_map, [1., 0.])
    assert torch.allclose(recovered_current_feature_map_0, current_feature_map_0, atol=0.5)

    conv = torch.nn.Conv2d(3, 10, 3, stride=2, device=last_feature_map_0.device)
    conv2 = torch.nn.Conv2d(10, 10, 7, stride=2, device=last_feature_map_0.device)
    last_feature_map_1 = conv(last_feature_map_0)
    current_feature_map_1 = conv(current_feature_map_0)
    
    warmup = 10
    repeat = 20
    for i in range(warmup + repeat):
        if i == warmup:
            stime = time.time()
        with log_dur(prefix="reorg"):
            reorg_feature_map_0 = matcher.reorganize_uncached_zones(last_feature_map_0, current_feature_map_0, offset=2, formula=[0.5, -2.])
        with log_dur(prefix="conv"):
            reorg_feature_map_1 = conv(reorg_feature_map_0)
        with log_dur(prefix="recover"):
            recovered_current_feature_map_1 = matcher.recover_feature_map(last_feature_map_1, reorg_feature_map_1, formula=[0.5, -2.])
    with log_dur(prefix="torch.cuda.synchronize"):
        torch.cuda.synchronize()
    print(f"avg dur {(time.time() - stime)/repeat:.4f}s")
    print(f"Max diff {torch.abs(current_feature_map_1-recovered_current_feature_map_1).max():.4f}")
    print(f"Max val {torch.abs(current_feature_map_1).max():.4f}")
    
    if debug:
        img = compare_origin_and_reorg(current_feature_map_1, recovered_current_feature_map_1, formula=[0.5, -2], **matcher.reorganize_info)
        plt.imshow(img)
        plt.show()
    for i in range(warmup+repeat):
        if i > warmup:
            stime = time.time()
        matcher.match(new_img)
    print(f"Matching avg takes {(time.time()-stime)/repeat:.4f}s")

def plt_torch(img: torch.Tensor, save="", min_val=None, max_val=None):
    shape = img.shape
    if shape[0] == 1 and len(shape) > 3:
        img = img[0].moveaxis(-3,-1)[..., :3]
    if min_val is None:
        min_val = img.min()
    if max_val is None:
        max_val = img.max()
    img = ((img - min_val) / max_val) * 255
    plt.figure()
    img = img.type(torch.uint8).cpu().numpy()
    if save:
        plt.imsave(save, img)
    plt.imshow(img)
    plt.show()

@torch.no_grad()
def test2():
    import torchvision
    import os
    debug = True
    work = os.environ["work"]
    path = f"{work}/third_parties/kapao/res/crowdpose_100024.jpg"
    whole_img = cv2.imread(path)[..., [2,1,0]][None]
    print(whole_img.shape)
    img_h = 300
    img_w = 400
    offset_h = 60
    offset_w = 30
    img_1 = whole_img[..., :img_h, :img_w, :]
    img_2 = whole_img[..., offset_h:offset_h+img_h, offset_w:offset_w+img_w, :]
    matcher = ImageMatcher(debug=True)
    img = whole_img
    last_feature_map_0 = torch.from_numpy(img_1).moveaxis(-1, -3).type(torch.float32).to("cuda:0").contiguous()
    current_feature_map_0 = torch.from_numpy(img_2).moveaxis(-1, -3).type(torch.float32).to("cuda:0").contiguous()

    conv1 = torch.nn.Conv2d(3, 10, 7, stride=1, device=last_feature_map_0.device)
    conv2 = torch.nn.Conv2d(10, 10, 3, stride=1, device=last_feature_map_0.device)
    conv3 = torch.nn.Conv2d(10, 3, 3, stride=2, device=last_feature_map_0.device)
    formula = [1., 0.]
    formula[0] /= 2* 1*1
    formula[1] = (-3 + 2) / 2 + formula[1] / 2
    formula[1] = (-7 + 1) / 1 + formula[1] / 1
    formula[1] = (-3 + 1) / 1 + formula[1] / 1
    formula[1] = 0.
    conv = torch.nn.Sequential(conv1, conv2, conv3)
    last_feature_map_1 = conv(last_feature_map_0)
    current_feature_map_1 = conv(current_feature_map_0)
    all_min = min(last_feature_map_1.min(), current_feature_map_1.min())
    all_max = max(last_feature_map_1.max(), current_feature_map_1.max())
    plt_torch(torch.cat([last_feature_map_1, current_feature_map_1], dim=-1),
              min_val=all_min, max_val=all_max)
    plt.show()

    with log_dur(prefix="reorg"):
        reorg_feature_map_0, _ = matcher.reorganize_uncached_zones(last_feature_map_0, current_feature_map_0, offset=10, formula=formula)
        plt_torch(reorg_feature_map_0, "reorg.jpg")
        img = compare_origin_and_reorg(current_feature_map_0, reorg_feature_map_0, **matcher.reorganize_info)
        if debug:
            plt.figure()
            plt.imshow(img)
            plt.imsave("compare_reorganize.jpg", img)
            plt.show()
        with log_dur(prefix="conv"):
            plt_torch(reorg_feature_map_0)
            reorg_feature_map_1 = conv(reorg_feature_map_0)
            plt_torch(reorg_feature_map_1, "reorg_inferred.jpg", min_val=all_min, max_val=all_max)
        with log_dur(prefix="recover"):
            recovered_current_feature_map_1 = matcher.recover_feature_map(last_feature_map_1, reorg_feature_map_1, formula=formula, min_val=all_min, max_val=all_max)
    plt_torch(last_feature_map_1, "last_inferred.jpg", min_val=all_min, max_val=all_max)
    plt_torch(current_feature_map_1, "current_inferred.jpg", min_val=all_min, max_val=all_max)
    plt_torch(recovered_current_feature_map_1, "recovered.jpg", min_val=all_min, max_val=all_max)
    plt.show()


@torch.no_grad()
def test3():
    import torchvision
    import os
    debug = True
    work = os.environ["work"]
    img_1 = cv2.imread(f"/home/guanxiux/project/ParallelCollaborativeInference/data/DAVIS/JPEGImages/480p/goat/00000.jpg")[..., [2,1,0]][None]
    img_2 = cv2.imread(f"/home/guanxiux/project/ParallelCollaborativeInference/data/DAVIS/JPEGImages/480p/goat/00004.jpg")[..., [2,1,0]][None]
    print(img_1.shape)
    matcher = ImageMatcher(debug=True)
    last_feature_map_0 = torch.from_numpy(img_1).moveaxis(-1, -3).type(torch.float32).to("cuda:0").contiguous()
    current_feature_map_0 = torch.from_numpy(img_2).moveaxis(-1, -3).type(torch.float32).to("cuda:0").contiguous()
    
    conv1 = torch.nn.Conv2d(3, 64, 1, stride=1, device=last_feature_map_0.device)

    conv1 = torch.nn.Conv2d(3, 10, 7, stride=1, device=last_feature_map_0.device)
    conv2 = torch.nn.Conv2d(10, 10, 3, stride=1, device=last_feature_map_0.device)
    conv3 = torch.nn.Conv2d(10, 3, 3, stride=2, device=last_feature_map_0.device)
    conv = torch.nn.Sequential(conv1, conv2, conv3)
    formula = [1., 0.]
    formula[0] /= 2* 1*1
    formula[1] = (-3 + 2) / 2 + formula[1] / 2
    formula[1] = (-7 + 1) / 1 + formula[1] / 1
    formula[1] = (-3 + 1) / 1 + formula[1] / 1
    formula[1] = 0.

    last_feature_map_1 = conv(last_feature_map_0)
    current_feature_map_1 = conv(current_feature_map_0)
    all_min = min(last_feature_map_1[:, :3].min(), current_feature_map_1[:, :3].min())
    all_max = max(last_feature_map_1[:, :3].max(), current_feature_map_1[:, :3].max())
    plt_torch(torch.cat([last_feature_map_1, current_feature_map_1], dim=-1),
              min_val=all_min, max_val=all_max)
    plt.show()

    with log_dur(prefix="reorg"):
        reorg_feature_map_0, _ = matcher.reorganize_uncached_zones(last_feature_map_0, current_feature_map_0, offset=10, formula=formula, min_diff=60)
        img = compare_origin_and_reorg(current_feature_map_0, reorg_feature_map_0, **matcher.reorganize_info)
        if debug:
            plt.figure()
            plt.imshow(img)
            plt.imsave("compare_reorganize.jpg", img)
            plt.show()
        with log_dur(prefix="conv"):
            plt_torch(reorg_feature_map_0)
            reorg_feature_map_1 = conv(reorg_feature_map_0)
            plt_torch(reorg_feature_map_1, "reorg_inferred.jpg", min_val=all_min, max_val=all_max)
        with log_dur(prefix="recover"):
            recovered_current_feature_map_1 = matcher.recover_feature_map(last_feature_map_1, reorg_feature_map_1, formula=formula, min_val=all_min, max_val=all_max)
    plt_torch(last_feature_map_1, "last_inferred.jpg", min_val=all_min, max_val=all_max)
    plt_torch(current_feature_map_1, "current_inferred.jpg", min_val=all_min, max_val=all_max)
    plt_torch(recovered_current_feature_map_1, "recovered.jpg", min_val=all_min, max_val=all_max)
    plt.show()


if __name__ == "__main__":
    test2()
