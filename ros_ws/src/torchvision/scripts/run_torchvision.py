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
parser.add_argument('-d', '--dataset', default='CIFAR10',
                    help='dataset')
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
    rospy.init_node("run_torchvision")
    args = parser.parse_args()
    task = args.task
    model_arch: str = args.arch
    parallel_approach: str = args.parallel
    # if parallel_approach == "tp":
    #     assert model_arch.lower().startswith("vgg"),\
    #         f"Unsupported model arch {model_arch} with parallel approach {parallel_approach}"
    model_arch = model_arch.lower()
    dataset_name: str = args.dataset

    data_dir = osp.join(os.environ["work"], "data", dataset_name)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    rospy.loginfo(f"Using data dir: {data_dir};")
    rospy.loginfo(f"device {device}; task {task}; model_arch {model_arch}; dataset {dataset_name}")

    if False: # TODO support iterate through all these models and datasets
        if task == "all":
            all_models = models.list_models()
        if task == "classification":
            all_models = models.list_models(module=models)
        else:
            all_models = models.list_models(module=getattr(models, task))

        datasets = {
            "classification": ["CIFAR10", "CIFAR100", "MNIST"],
            "detection": ["CocoDetection", "Kitti", "VOCDetection"],
            "segmentation": ["Cityscapes", "VOCSegmentation"],
            "video": ["HMDB51", "Cityscapes"]
        }
    weights = get_model_weights(model_arch, models).DEFAULT
    model: torch.nn.Module = getattr(models, model_arch)(weights=weights)
    preprocess = weights.transforms()
    model.eval()
    model = model.to(device)
    if dataset_name == "ImageNet":
        kwargs = {"split": "val"}
    elif "CIFAR" in dataset_name:
        kwargs = {"download": True, "train": False}
    dataset: datasets.DatasetFolder = getattr(datasets, dataset_name)(
        data_dir, transform=preprocess, **kwargs)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)

    from ParallelCollaborativeInference import ParallelCollaborativeInference
    offload_mode = rospy.get_param("/offload_mode", "flex")
    offload_method = rospy.get_param("/offload_method", "select")
    PCI = ParallelCollaborativeInference(offload= not args.no_offload,
                                         parallel_approach=offload_method,
                                         ip="192.168.10.8", port=12345,
                                        constraint_latency=offload_mode=="fix",
                                        log=rospy.loginfo)
    PCI.start_client(model=model, init_forward_count=1)

    start = Event()
    correct_count = 0
    for i, (inp, target) in enumerate(dataloader):
        if rospy.is_shutdown():
            break
        pred = model(inp.to(device))[0]
        result = weights.meta['categories'][torch.argmax(pred)]
        if result in dataset.classes[target]:
            correct_count += 1

        torch.cuda.synchronize()
        if not start.is_set():
            rospy.Service("/Start", Empty, start_service)
            rospy.logwarn("Waiting for /Start called.")
            while not rospy.is_shutdown() and not start.wait(2):
                continue
        if task == "classification" and i % 100 == 0:
            rospy.loginfo(f"Result: pred: {result}; gt: {dataset.classes[target]}.")
            rospy.loginfo(f"Accuracy@1 {correct_count / (i+1)*100:.4f}%")

# model candidates
# name; param size; acc@1
# DenseNet121; 8.0M; 74.434
# MobileNet_V3_Small; 2.5M; 67.668  (to many shape 1 layers)
# RegNet_X_3_2GF; 15.3M; 81.196
# ResNet101; 44.5M; 81.886
# ConvNeXt_Base; 88.6M; 84.062
# VGG19_BN; 143.7M; 74.218
# ConvNeXt_Large; 197.8M; 84.414
# RegNet_Y_128GF; 644.8M; 88.228    # (Abnormal with swap involved.)



