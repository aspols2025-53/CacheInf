import argparse
import os
import os.path as osp
import time
from inspect import getmembers, isclass

import sys
project_dir = osp.abspath(osp.join(*([osp.abspath(__file__)] + [os.pardir] * 5)))
sys.path.insert(0, project_dir)
from threading import Event
import rospy
from std_srvs.srv import Empty, EmptyResponse

import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from torchvision import models, datasets

tasks = ["all", "classification", "detection", "segmentation", "video"]
parser = argparse.ArgumentParser(description='torch vision inference')
parser.add_argument('-t', '--task', default='classification',
                    choices=tasks,
                    help='task: ' +
                    ' | '.join(tasks) +
                    ' (default: classification)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet101',
                    help='model architecture')
parser.add_argument('-d', '--dataset',
                    default=f'{os.environ["work"]}/data/DAVIS/JPEGImages/1080p',
                    help='sequences dir of video datasets')
parser.add_argument('-p', '--parallel', default='select')
parser.add_argument('--no-offload', action='store_true',
                    help='no offload')

def get_model_weights(model_arch: str, module):
    all_class = getmembers(module, isclass)
    model_arch = model_arch.upper()
    cls = None
    for cls_name, cls in all_class:
        if cls_name.upper().startswith(model_arch):
            break
    assert cls is not None, f"{model_arch} not found."
    return cls

def start_service(_):
    start.set()
    return EmptyResponse()

if __name__ == "__main__":
    rospy.init_node("run_torchvision_video")
    args = parser.parse_args()
    task = args.task
    model_arch: str = args.arch
    parallel_approach: str = args.parallel
    model_arch = model_arch.lower()
    dataset_name: str = args.dataset

    data_dir = osp.join(os.environ["work"], "data", dataset_name)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    rospy.loginfo(f"Using data dir: {data_dir};")
    rospy.loginfo(f"device {device}; task {task}; model_arch {model_arch}; dataset {dataset_name}")

    weights = get_model_weights(model_arch, models).DEFAULT
    model: torch.nn.Module = getattr(models, model_arch)(weights=weights)
    preprocess = weights.transforms()
    model.eval()
    model = model.to(device)
    if dataset_name == "ImageNet":
        kwargs = {"split": "val"}
    elif "CIFAR" in dataset_name:
        kwargs = {"download": True, "train": False}

    from ParallelCollaborativeInference import ParallelCollaborativeInference, VideoFrameDataset
    dataset = VideoFrameDataset(args.dataset, log=rospy.loginfo, frame_rate=1e-8, play_dur=1e8)
    offload_mode = rospy.get_param("/offload_mode", "flex")
    offload_method = rospy.get_param("/offload_method", "cached")
    PCI = ParallelCollaborativeInference(offload= not args.no_offload,
                                         parallel_approach=offload_method,
                                        #  ip="192.168.10.8", port=12345,
                                        constraint_latency=offload_mode=="fix",
                                        log=rospy.loginfo)
    PCI.start_client(model=model, init_forward_count=1)

    start = Event()
    correct_count = 0
    for i, img in enumerate(dataset):
        short_shape = min(img.shape[:2])
        img = img[:short_shape, :short_shape]
        if rospy.is_shutdown():
            break
        inp = preprocess(torch.from_numpy(img).permute(2,0,1).unsqueeze(0))
        pred = model(inp.to(device), img=img)[0]
        result = weights.meta['categories'][torch.argmax(pred)]
        rospy.loginfo(f"Current prediction result: {result}")

        torch.cuda.synchronize()
        if not start.is_set():
            rospy.Service("/Start", Empty, start_service)
            rospy.logwarn("Waiting for /Start called.")
            while not rospy.is_shutdown() and not start.wait(2):
                continue

# model candidates
# name; param size; acc@1
# DenseNet121; 8.0M; 74.434
# MobileNet_V3_Small; 2.5M; 67.668  (to many shape 1 layers)
# RegNet_X_3_2GF; 15.3M; 81.196
# ResNet101; 44.5M; 81.886
# ConvNeXt_Base; 88.6M; 84.062
# VGG19_BN; 143.7M; 74.218
# ConvNeXt_Large; 197.8M; 84.414



