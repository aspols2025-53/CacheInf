#!/usr/bin/env python3
# -*-coding:utf-8 -*-
import os
import os.path as osp
import sys
from threading import Event

project_dir = osp.abspath(osp.join(*([osp.abspath(__file__)] + [os.pardir] * 5)))
argnav_dir = osp.join(project_dir, "third_parties", "AGRNav")
perception_dir = osp.join(argnav_dir, "src", "perception")
perception_sconet_dir = osp.join(perception_dir, "SCONet")
sys.path.insert(0, osp.abspath(osp.join(osp.abspath(__file__), *[osp.pardir]*5)))
from ParallelCollaborativeInference import ParallelCollaborativeInference
sys.path.insert(0, perception_dir)
sys.path.insert(0, perception_sconet_dir)
import torch
import numpy as np
import rospy
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import Float64MultiArray, Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import time
from SCONet import SCONet
# from dataset import get_dataset
from network.common.dataset import get_dataset
from network.common.seed import seed_all
from network.common.config import CFG
from network.common.model import get_model
from network.common.logger import get_logger
from network.common.io_tools import dict_to, _create_directory
import network.common.checkpoint as checkpoint
import network.data.io_data as SemanticKittiIO

start = Event()
def start_handle(_):
    start.set()
    return EmptyResponse()

def get_model(_cfg, dataset):

  nbr_classes = dataset.nbr_classes
  grid_dimensions = dataset.grid_dimensions
  class_frequencies = dataset.class_frequencies

  model = SCONet(class_num=nbr_classes, input_dimensions=grid_dimensions, class_frequencies=class_frequencies)
  # ------------------------------------------------------------------------------------------------------------------

  return model

def publish_coordinates(coordinates, publisher, scores):

    coordinates = coordinates[:, [0, 2, 1]]
    
    voxel_size = 0.1
    voxel_origin = np.array([-10, -10, -0.1])
    pc_coordinates = coordinates * voxel_size + voxel_origin
    header = Header()
    header.frame_id = "world"
    header.stamp = rospy.Time.now()
    fields = [
        PointField(name="x", offset=0, datatype=7, count=1),
        PointField(name="y", offset=4, datatype=7, count=1),
        PointField(name="z", offset=8, datatype=7, count=1),
        PointField(name="intensity", offset=12, datatype=7, count=1),]
    pc_msg = pc2.create_cloud(header, fields, np.concatenate([pc_coordinates, scores[:, None]], -1))
    # pc_msg = pc2.create_cloud_xyz32(header, pc_coordinates)
    publisher.publish(pc_msg)

    # coordinates_msg = Float64MultiArray()
    # for coordinate in coordinates:
    #     # print(f"coordinate : {coordinate}")
    #     coordinates_msg.data.extend(coordinate)

    # publisher.publish(coordinates_msg)

last_input_file_name = [""]
# os.chdir(perception_sconet_dir)

def test(model, dset, _cfg, logger, out_path_root, coordinates_publisher):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32
    # ori_voxels_path = "/home/melodic/jetsonNX/Aerial-Walker/src/oc_navigation/plan_manage/raw_data/ori_voxels"
    logger.info('=> Passing the network on the test set...')
    inv_remap_lut = dset.dataset.get_inv_remap_lut()

    start_time = time.time()
    inference_time = []
   
    with torch.no_grad():
        for t, (data, indices) in enumerate(dset):
            data = dict_to(data, device, torch.uint8)

            # Record the inference start time
            inference_start_time = time.time()
            
            data['3D_OCCUPANCY'] = data['3D_OCCUPANCY'].type(torch.float32)[..., :, ::2]

            waiting = False
            while True:
                scores = model(data)
                while not start.wait(timeout=2) and not rospy.is_shutdown():
                    if not waiting:
                        rospy.Service("Start", Empty, start_handle)
                        waiting = True
                    rospy.logwarn("Waiting for /Start call.")
                    continue

                torch.cuda.synchronize()
                inference_end_time = time.time()

                # Record the inference end time

                # Log the inference time of each sample
                inference_time.append(inference_end_time - inference_start_time)

                curr_index = 0
                for score in scores['pred_semantic_1_1']:
                    # voxel occupancy file
                    input_filename = dset.dataset.filepaths['3D_OCCUPANCY'][indices[curr_index]]
                    print(input_filename)

                    voxel_occupancy = torch.moveaxis(data['3D_OCCUPANCY'][0][0], 2, 1).ravel().reshape(256, 32, 128)
                    # assert np.allclose(voxel_occupancy, _voxel_occupancy.cpu().numpy())


                    # Create a mask for occupied voxels
                    voxel_mask = voxel_occupancy.ravel() == 1


                    # Create a mask for occupied voxels in scores
                    score_mask = score.ravel() > 0


                    # Compute the intersection of occupied voxels in both score and voxel_occupancy
                    # intersection = torch.logical_and(voxel_mask, score_mask)


                    # Compute the non-intersected occupied voxels coordinates in voxel_occupancy
                    non_intersection = torch.logical_and(score_mask, torch.logical_not(voxel_mask))

                    # Get the non-intersected occupied voxel coordinates
                    non_intersection_coordinates = torch.nonzero(non_intersection.reshape(256, 32, 128))
                    non_intersection_score = score[torch.nonzero(non_intersection.reshape(256, 32, 128), as_tuple=True)]

                    # score = torch.moveaxis(score, 2, 1).reshape(-1).cpu().numpy().astype(np.uint16)
                    # score = inv_remap_lut[score].astype(np.uint16)

                    publish_coordinates(non_intersection_coordinates.cpu().numpy(), coordinates_publisher, non_intersection_score.cpu().numpy())

                    # filename, extension = os.path.splitext(os.path.basename(input_filename))
                    # out_filename = os.path.join(out_path_root, 'predictions', filename + '.label')
                    # _create_directory(os.path.dirname(out_filename))
                    # score.tofile(out_filename)
                    #shutil.copy(input_filename, ori_voxels_path)
                    # if len(os.listdir(osp.dirname(input_filename))) > 2:
                    #     os.remove(input_filename)
                    # curr_index += 1


    return inference_time


def main():
    rospy.init_node("inference_node")
    #Create the publisher using a specific ROS message type and topic
    # coordinates_publisher = rospy.Publisher('/non_intersection_coordinates', Float64MultiArray, queue_size=1) 
    coordinates_publisher = rospy.Publisher('sconet_pred', PointCloud2, queue_size=1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    torch.backends.cudnn.enabled = True
    seed_all(0)
    weights_f = rospy.get_param(
        '~weights_file', osp.join(
            perception_dir, "SCONet", "network", "weights", "weights_epoch_037.pth"))
    dataset_f = rospy.get_param('~dataset_root', osp.join(perception_dir, "raw_data", "voxels"))
    out_path_root = rospy.get_param('~output_path', osp.join(perception_dir, "SCONet", "network", "output"))
    assert os.path.isfile(weights_f), f'=> No file found at {weights_f}'
    checkpoint_path = torch.load(weights_f)
    config_dict = checkpoint_path.pop('config_dict')
    config_dict["DATALOADER"]["NUM_WORKERS"] = 1
    config_dict["TRAIN"]["BATCH_SIZE"] = 1
    config_dict["VAL"]["BATCH_SIZE"] = 1
    config_dict["TRAIN"]["BATCH_SIZE"] = 1
    config_dict['DATASET']['ROOT_DIR'] = dataset_f
    _cfg = CFG()
    _cfg.from_dict(config_dict)
    logger = get_logger(out_path_root, 'logs_test.log')
    logger.info(f'============ Test weights: "{weights_f}" ============\n')
    wait_time = 1  # Seconds to wait before checking the dataset folder again
    train_batch_size = 1  # Set your desired batch_size here
    model = None
    dataset = get_dataset(_cfg)['test']
    while not rospy.is_shutdown():

        # Check if the dataset folder has sufficient data (files) for the batch size
        # while model is None and len(os.listdir(dataset_f)) < train_batch_size + 1:
        #     rospy.loginfo("Waiting for dataset folder to accumulate sufficient files.")
        #     rospy.sleep(wait_time)
        if model is None:
            logger.info('=> Loading network architecture...')
            model = get_model(_cfg, dataset.dataset)
            logger.info('=> Loading network weights...')
            model = model.to(device=device)
            model = checkpoint.load_model(model, weights_f, logger)
            model.eval()
            offload = rospy.get_param("/offload", True)
            offload_mode = rospy.get_param("/offload_mode", "flex")
            offload_method = rospy.get_param("/offload_method", "select")
            ip = rospy.get_param("~offload_ip", "192.168.10.8")
            port = rospy.get_param("~port", 12345)
            rospy.loginfo(f"offload_mode {offload_mode}")
            pci = ParallelCollaborativeInference(offload, offload_method, ip=ip, port=port,
                                                constraint_latency=offload_mode=="fix",
                                                log=rospy.loginfo)
            pci.start_client(model, init_forward_count=0)
        inference_time = test(model, dataset, _cfg, logger, out_path_root, coordinates_publisher)  
        logger.info('=> ============ Network Test Done ============')
        logger.info('Inference time per frame is %.6f seconds\n' % (np.sum(inference_time) / 1.0))

if __name__ == '__main__':
    main()
