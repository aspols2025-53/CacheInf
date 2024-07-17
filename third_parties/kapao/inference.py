import sys
import time
from pathlib import Path

from torch import nn

import torch
import argparse
import yaml
from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords
from utils.datasets import LoadImages
from models.experimental import attempt_load
from val import run_nms, post_process_batch
import cv2
import os.path as osp
import numpy as np
from ParallelCollaborativeInference import ParallelCollaborativeInference
import os
os.chdir(osp.dirname(osp.abspath(__file__)))

device = torch.device("cuda")
# device = torch.device("cpu")

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.count += n
        self.val = val
        self.sum += val * n
        self.avg = self.sum / self.count

def parse_kapoa_argv():
    import argparse, yaml
    old_sys_argv = sys.argv[1:]
    sys.argv = sys.argv[:1] + ["--bbox", "--pose"]
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--img-path', default='res/crowdpose_100024.jpg', help='path to image')

    # plotting options
    parser.add_argument('--bbox', action='store_true')
    parser.add_argument('--kp-bbox', action='store_true')
    parser.add_argument('--pose', action='store_true')
    parser.add_argument('--face', action='store_true')
    parser.add_argument('--color-pose', type=int, nargs='+', default=[255, 0, 255], help='pose object color')
    parser.add_argument('--color-kp', type=int, nargs='+', default=[0, 255, 255], help='keypoint object color')
    parser.add_argument('--line-thick', type=int, default=2, help='line thickness')
    parser.add_argument('--kp-size', type=int, default=1, help='keypoint circle size')
    parser.add_argument('--kp-thick', type=int, default=2, help='keypoint circle thickness')

    # model options
    parser.add_argument('--data', type=str, default='data/coco-kp.yaml')
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--weights', default='kapao_l_coco.pt')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--no-kp-dets', action='store_true', help='do not use keypoint objects')
    parser.add_argument('--conf-thres-kp', type=float, default=0.5)
    parser.add_argument('--conf-thres-kp-person', type=float, default=0.2)
    parser.add_argument('--iou-thres-kp', type=float, default=0.45)
    parser.add_argument('--overwrite-tol', type=int, default=25)
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--flips', type=int, nargs='+', default=[-1])

    args = parser.parse_args()
    sys.argv = [sys.argv[0]] + old_sys_argv
    
    with open(args.data) as f:
        data = yaml.safe_load(f)  # load data dict

    # add inference settings to data dict
    data['imgsz'] = args.imgsz
    data['conf_thres'] = args.conf_thres
    data['iou_thres'] = args.iou_thres
    data['use_kp_dets'] = not args.no_kp_dets
    data['conf_thres_kp'] = args.conf_thres_kp
    data['iou_thres_kp'] = args.iou_thres_kp
    data['conf_thres_kp_person'] = args.conf_thres_kp_person
    data['overwrite_tol'] = args.overwrite_tol
    data['scales'] = args.scales
    data['flips'] = [None if f == -1 else f for f in args.flips]
    data['count_fused'] = False
    return args, data

if __name__ == '__main__':
    import os.path as osp
    kapao_dir = osp.dirname(osp.abspath(__file__))
    with open("/workspace/third_parties/kapao/data/coco-kp.yaml", "r") as f:
        data = yaml.safe_load(f)
    _, data = parse_kapoa_argv()


    print('KAPAO Using device: {}'.format(device))

    model = attempt_load("/workspace/data/kapao/kapao_l_coco.pt", map_location=device)
    for m in model.modules():
        if isinstance(m, nn.Upsample):
            m.recompute_scale_factor = None

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(1280, s=stride)  # check image size
    dataset = LoadImages("/workspace/third_parties/kapao/res/crowdpose_100024.jpg", img_size=imgsz, stride=stride, auto=True)
    (_, img, im0, _) = next(iter(dataset))
    img = torch.from_numpy(img).to(device)
    # img = img / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    inference_time=AverageMeter()
    i = 0
    PCI = ParallelCollaborativeInference(parallel_approach="select",ip="192.168.50.11")
    PCI.start_client(model=model, init_forward_count=1)
    # Init forward
    model(img, data, augment=True, kp_flip=data['kp_flip'], scales=data['scales'], flips=data['flips'])
    while i < 10:
        start = time.time()
        out = model(img, data, augment=True, kp_flip=data['kp_flip'], scales=data['scales'], flips=data['flips'])
        end = time.time()
        i += 1
        inference_time.update((end-start)*1000)
        print(f"inference time: {inference_time.val:.3f} ms, average {inference_time.avg:.3f} ms")