#!/usr/bin/env python
import os
import os.path as osp
import sys

import time
from threading import Thread, Event
import rospy
import numpy as np
from numpy import ma
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from std_srvs.srv import Empty, EmptyResponse
from tf2_ros import TransformListener, Buffer
import cv2
from cv_bridge import CvBridge
import message_filters
from message_filters import ApproximateTimeSynchronizer
import tcod
from scipy.spatial.transform import Rotation

from projection_utils import (unproject_depth, homogeneous, tf_convert,
                              yaw_to_quat)
from offload_utils import (LatestQueue, LogState, Kalman2DTracker,
                           CostmapMaintain, assign_around_2d, los_end)

project_dir = osp.abspath(osp.join(*([osp.abspath(__file__)] + [os.pardir] * 5)))
kapao_path = osp.join(project_dir, "third_parties", "kapao")
sys.path.insert(0, project_dir)
sys.path.insert(0, kapao_path)
os.chdir(kapao_path)


from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords
# from kapao.utils.datasets import LoadImages
from models.experimental import attempt_load
from utils.augmentations import letterbox
from val import run_nms, post_process_batch


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
    parser.add_argument('--imgsz', type=int, default=960)
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


class CompNavigation():
    def __init__(self):
        self.ready1 = Event()
        self.ready2 = Event()
        self.start = Event()
        self.data_queue = LatestQueue(10)
        self.inferred_queue = LatestQueue(1)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        self.log_tf = LogState(
            "TF message found.",
            "TF message error."
        )

        while not rospy.is_shutdown():
            try:
                if not self.tf_buffer.can_transform('map', "camera_link",
                    rospy.Time(), timeout=rospy.Duration(1)):
                    rospy.loginfo("Waiting for transform from map to camera to become available.")
                    continue
            except:
                continue
            break

        self.bridge = CvBridge()
        self.costmap = CostmapMaintain()
        rgb_sub = message_filters.Subscriber("camera/rgb/image_raw", Image, queue_size=500)
        depth_sub = message_filters.Subscriber("camera/depth/image_raw", Image, queue_size=500)
        approx_time_synchronizer = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], 10, 0.1, reset=True)
        approx_time_synchronizer.registerCallback(self.store_data)
        rospy.loginfo("Created subscribtion for camera/rgb/image_raw")
        rospy.loginfo("Created subscribtion for camera/depth/image_raw")

        Thread(target=self.run_nms, name="run_nms", daemon=True).start()
        Thread(target=self.comp_goal, name="comp_goal", daemon=True).start()
        while not rospy.is_shutdown() and not (self.ready1.wait(2) and self.ready2.wait(2)):
            continue
        rospy.Service("Start", Empty, self.start_handler)
        rospy.logwarn("Waiting for service call: /Start")

    def start_handler(self, _):
        if not self.ready1.is_set() or not self.ready2.is_set():
            rospy.logerr("Not ready yet.")
            return EmptyResponse()
        self.start.set()
        return EmptyResponse()

    def store_data(self, rgb: Image, depth: Image):
        stamp = rgb.header.stamp
        try:
            depth_optical_tf = tf_convert(self.tf_buffer.lookup_transform(
                "map", "camera_link",
                stamp, rospy.Duration.from_sec(0.5)), True)
            baselink_tf = tf_convert(self.tf_buffer.lookup_transform(
                "map", "base_link", stamp,
                rospy.Duration.from_sec(0.5)), True)
            self.log_tf(True)
        except:
            depth_optical_tf = baselink_tf = None
            self.log_tf(False)
        self.data_queue.put([
            self.bridge.imgmsg_to_cv2(rgb),
            self.bridge.imgmsg_to_cv2(depth),
            stamp,
            depth_optical_tf,
            baselink_tf
        ])

    def run_nms(self):
        rospy.loginfo("Initializing model.")
        args, data = parse_kapoa_argv()
        img_pub = rospy.Publisher("kapao_pose", Image, queue_size=100)
        offload = rospy.get_param("/offload", True)
        offload_mode = rospy.get_param("/offload_mode", "flex")

        debug = rospy.get_param("/debug", True)
        window = rospy.get_param("/window", 10)
        rospy.loginfo(f"Offload {offload}; window {window}s; debug {debug}")
        import torch
        device = select_device(args.device, batch_size=1)
        model = attempt_load(args.weights, map_location=device)
        model.eval()
        for m in model.modules():
            if type(m) in [torch.nn.Upsample]:
                m.recompute_scale_factor = None

        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(args.imgsz, s=stride)  # check image size
        rospy.loginfo('Using device: {}'.format(device))
        rospy.loginfo("Model initialization finished.")

        from ParallelCollaborativeInference import ParallelCollaborativeInference
        offload_method = rospy.get_param("/offload_method", "select")
        ip = rospy.get_param("~offload_ip", "192.168.10.8")
        port = rospy.get_param("~port", 12345)
        pci = ParallelCollaborativeInference(offload, offload_method, ip, port,
                                                constraint_latency=offload_mode=="fix",
                                                log=rospy.loginfo)
        pci.start_client(model, init_forward_count=1)

        rospy.logwarn("Started.")

        stime = time.time()
        count = 0
        depth = None
        stamp = None
        depth_optical_tf = None
        baselink_tf = None
        im0 = cv2.imread(args.img_path)[..., [2,1,0]]
        im0 = np.ascontiguousarray(im0)
        while not rospy.is_shutdown():
            if debug:
                if depth is None:
                    _, depth, stamp, depth_optical_tf, baselink_tf = self.data_queue.get()
            else:
                im0, depth, stamp, depth_optical_tf, baselink_tf = self.data_queue.get()

            im0 = cv2.resize(im0, (640, 480))
            img = letterbox(im0, imgsz, stride=stride, auto=True)[0]

            img = img.transpose((2, 0, 1))  # HWC to CHW
            img = np.ascontiguousarray(img)
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            img = torch.from_numpy(img).to(device)

            person_dets, kp_dets = model(img, data, augment=False, kp_flip=data['kp_flip'], scales=data['scales'], flips=data['flips'])

            if not self.ready1.is_set():
                self.ready1.set()
            while not self.start.wait(timeout=2) and not rospy.is_shutdown():
                continue

            num_persons = len(person_dets[0])
            person_keypoints = []
            if num_persons > 0:
                if args.bbox and len(person_dets[0]) > 0:
                    bboxes = scale_coords(img.shape[-2:], person_dets[0][:, :4], im0.shape[:2]).round().cpu().numpy()
                    for i, (x1, y1, x2, y2) in enumerate(bboxes):
                        cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), args.color_pose, thickness=args.line_thick)

                _, poses, _, _, _ = post_process_batch(data, img, [], [[im0.shape[:2]]], person_dets, kp_dets)

                if args.pose:
                    for i, pose in enumerate(poses):
                        if args.face:
                            for x, y, c in pose[data['kp_face']]:
                                cv2.circle(im0, (int(x), int(y)), args.kp_size, args.color_pose, args.kp_thick)
                        for seg in data['segments'].values():
                            pt1 = (int(pose[seg[0], 0]), int(pose[seg[0], 1]))
                            pt2 = (int(pose[seg[1], 0]), int(pose[seg[1], 1]))
                            cv2.line(im0, pt1, pt2, args.color_pose, args.line_thick)
                        if i == 0:
                            person_keypoints = pose[..., :2].astype(int)
                        if data['use_kp_dets']:
                            for x, y, c in pose:
                                if c:
                                    cv2.circle(im0, (int(x), int(y)), args.kp_size, args.color_kp, args.kp_thick)
                if args.kp_bbox:
                    bboxes = scale_coords(img.shape[2:], kp_dets[0][:, :4], im0.shape[:2]).round().cpu().numpy()
                    for x1, y1, x2, y2 in bboxes:
                        cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), args.color_kp, thickness=args.line_thick)

            img_pub.publish(self.bridge.cv2_to_imgmsg(im0, encoding="rgb8"))
            if not (depth_optical_tf is None or baselink_tf is None):
                self.inferred_queue.put([depth, person_keypoints, stamp,
                                         depth_optical_tf, baselink_tf])
            count += 1
            dur = time.time() - stime
            if dur > window:
                rospy.loginfo(f"Inference freq: {count / dur:.4f} FPS; dur: {dur:.4f} s")
                stime = time.time()
                count = 0

    def comp_goal(self):
        camera_info = rospy.wait_for_message("camera/depth/camera_info", CameraInfo)
        rospy.loginfo("Got camera info.")
        debug = rospy.get_param("/debug", True)
        window = rospy.get_param("/window", 10)
        stationary = rospy.get_param("/stationary", True)
        rospy.loginfo(f"Stationary: {stationary}")
        if stationary:
            goal_pub = rospy.Publisher("test", PoseStamped, queue_size=1)
        else:
            goal_pub = rospy.Publisher("move_base_simple/goal", PoseStamped, queue_size=1)
        person_point_pub = rospy.Publisher("person_point", PointStamped, queue_size=100)

        kf = Kalman2DTracker(init_len=5)
        last_person_idx = [-100, -100]
        last_target_idx = [-100, -100]
        last_predicted_idx = [None, None]
        predicted = [0]

        log_valid_data = LogState(
            "Start/resume processing image.",
            "Sensor data not yet ready.",)
        log_people_in_view = LogState(
            "Marking and tracking pose of people in view.",
            "No people in view. Predicting position.")
        log_not_enough_observation = LogState(
            "Not enough observation for Kalman Filter prediction.",
            "Predicting with Kalman Filter.")
        log_stationary= LogState("Skipping small perturbation.",
                                "Tracking people for evident movement.")
        count = 0
        stime = time.time()
        clear_time = time.time()

        self.ready2.set()
        while not rospy.is_shutdown():
            depth, person_keypoints, stamp, depth_optical_tf, baselink_tf = self.inferred_queue.get()
            costmap = self.costmap
            if costmap.map is None:
                log_valid_data(False)
                continue
            log_valid_data(True)
            if time.time() - clear_time > 10:
                try:
                    rospy.ServiceProxy("/move_base/clear_costmaps", Empty)()
                except:
                    pass
                clear_time = time.time()

            def determine_people_coord():
                if len(person_keypoints) == 0:
                    return []
                # cv2 image indexing different from numpy
                _person_keypoints = np.clip(person_keypoints, [0,0], np.array(depth.shape) - 1)
                _person_depths = depth.T[tuple(_person_keypoints.T)] / 1000.
                if not debug:
                    mask = _person_depths > 0
                    _person_depths = _person_depths[mask]
                    _person_keypoints = _person_keypoints[mask]
                if len(_person_depths) == 0:
                    return []
                camera_matrix = np.array(camera_info.K).reshape(3,3)
                # unproject 2d point to camera frame
                coord_cam = unproject_depth(_person_keypoints, _person_depths, camera_matrix)
                coord_world = (depth_optical_tf @ homogeneous(coord_cam).T).T[...,:3]

                return np.median(coord_world, axis=0)
            person_coord = determine_people_coord()

            if debug and len(person_coord) == 0:
                person_coord = baselink_tf[:3, -1] + np.array([1.,0,0])

            def track_pose():
                map = costmap.map
                origin = costmap.origin[:2]
                resolution = costmap.resolution
                map[(map>=0) & (map <= 33)] = 0
                if debug and stationary:
                    map[map<0] = 0
                else:
                    map[map<0] = 100
                map[map>0] = 100
                trans = baselink_tf[:2, -1]
                # x,y in coordinates corresponds to width, height in map 
                baselink_idx = ((trans - origin) / resolution).astype(int)[[1,0]]

                distance = 2.5    # naively setting 4 meters
                def navigate_to_towards(x,y, tx, ty):
                    target_yaw = np.arctan2(ty - y,
                                            tx - x)
                    target_quat = yaw_to_quat(target_yaw)
                    goal = PoseStamped()
                    goal.header.frame_id = "map"
                    (goal.pose.position.x,
                        goal.pose.position.y,
                        goal.pose.position.z) = (x, y, baselink_tf[2,-1])
                    (goal.pose.orientation.x,
                        goal.pose.orientation.y,
                        goal.pose.orientation.z,
                        goal.pose.orientation.w,) = target_quat
                    # Publish goal
                    goal_pub.publish(goal)
                safety_length = 0.5 // resolution

                if log_people_in_view(len(person_coord) > 0):
                    predicted[0] = 0
                    
                    assign_around_2d(map, baselink_idx, 0, 2)

                    # Heading towards people if people in view
                    obs_person_idx = ((person_coord[:2] - origin) / resolution).astype(int)[[1,0]]

                    kf_person_idx = kf.observed(obs_person_idx)

                    filtered_person_coord = kf_person_idx[[1,0]] * resolution + origin

                    person_point = PointStamped()
                    person_point.header.frame_id = "map"
                    (person_point.point.x,
                        person_point.point.y,
                        person_point.point.z) = (filtered_person_coord[0],
                                                 filtered_person_coord[1], 0.)
                    person_point_pub.publish(person_point)

                    dir = baselink_idx - kf_person_idx
                    current_dist = np.linalg.norm(dir) * resolution
                    remain_dist = np.abs(distance - current_dist) / resolution
                    if current_dist > distance:
                        dir = dir * -1
                    target_idx = los_end(baselink_idx, baselink_idx + dir,
                                        remain_dist, safety_length, map, 0.2)


                    target_coord = target_idx[[1,0]] * resolution + origin
                    
                    target_yaw = np.arctan2(filtered_person_coord[1] - target_coord[1],
                                            filtered_person_coord[0] - target_coord[0])
                    current_yaw = Rotation.from_matrix(baselink_tf[:3,:3]).as_euler("xyz")[-1]
                    if np.abs(target_yaw - current_yaw) > 0.3:
                        navigate_to_towards(baselink_tf[0,-1],
                                        baselink_tf[1,-1],
                                        filtered_person_coord[0],
                                        filtered_person_coord[1])
                        log_stationary(False)
                        return kf_person_idx, baselink_idx
                    if np.linalg.norm(last_person_idx - kf_person_idx) * resolution < 0.8 and\
                        np.linalg.norm(last_target_idx -  target_idx) * resolution < 0.2:
                        log_stationary(True)
                        return None

                    navigate_to_towards(target_coord[0],
                                        target_coord[1],
                                        filtered_person_coord[0],
                                        filtered_person_coord[1])
                    log_stationary(False)
                    return kf_person_idx, target_idx
                else:
                    if count % 2 != 0:
                        return None
                    log_people_in_view(False)
                    if predicted[0] > 3:
                        kf_person_idx = np.array(last_predicted_idx)
                    else:
                        kf_person_idx = kf.observed(ma.masked)
                    if log_not_enough_observation(kf_person_idx[0] is None):
                        return None

                    kf_person_coord = kf_person_idx[[1,0]] * resolution + origin
                    person_point = PointStamped()
                    person_point.header.frame_id = "map"
                    (person_point.point.x,
                        person_point.point.y,
                        person_point.point.z) = (kf_person_coord[0],
                                                 kf_person_coord[1], 0.)
                    person_point_pub.publish(person_point)

                    length = np.linalg.norm(last_person_idx - kf_person_idx)
                    if log_stationary(length * resolution < 0.8):
                        return None
                    def get_closest():
                        _map = np.zeros_like(map)
                        _map[map > 0] = -1
                        _map[map == 0] = 1
                        assign_around_2d(_map, last_person_idx, 1, 10)
                        assign_around_2d(_map, baselink_idx, 1, 4)
                        graph = tcod.path.SimpleGraph(cost=_map, cardinal=1, diagonal=2)
                        pf = tcod.path.Pathfinder(graph)
                        pf.add_root(baselink_idx)
                        pf.resolve()
                        distance = pf.distance
                        mask = distance <= 2 / resolution
                        indices = np.argwhere(mask)

                        graph = tcod.path.SimpleGraph(cost=_map, cardinal=1, diagonal=2)
                        pf = tcod.path.Pathfinder(graph)
                        pf.add_root(last_person_idx)
                        pf.resolve()
                        dist2person = pf.distance
                        dist = dist2person[tuple(indices.T)]
                        idx = np.argmin(dist)
                        if dist[idx] < 10e5:
                            return indices[idx]
                        dist2last_person_idx = np.linalg.norm(indices - last_person_idx, axis=-1)
                        target_indice = np.argmin(dist2last_person_idx)
                        return indices[target_indice]

                    trace_person_idx = get_closest()

                    dir = trace_person_idx - kf_person_idx
                    current_dist = np.linalg.norm(dir) * resolution
                    remain_dist = np.abs(distance - current_dist) / resolution
                    if current_dist > distance:
                        dir = dir * -1
                    target_idx = los_end(trace_person_idx,
                                         kf_person_idx,
                                         remain_dist, safety_length, map, 0.2)
                    target_coord = target_idx[[1,0]] * resolution + origin

                    navigate_to_towards(target_coord[0],
                                        target_coord[1],
                                        kf_person_coord[0],
                                        kf_person_coord[1])
                    predicted[0] += 1
                    last_predicted_idx[0] = kf_person_idx[0]
                    last_predicted_idx[1] = kf_person_idx[1]
                    return None
            ret = track_pose()
            if ret is not None:
                last_person_idx, last_target_idx = ret[0].astype(int), ret[1].astype(int)

            count += 1
            dur = time.time() - stime
            if dur > window:
                stime = time.time()
                rospy.loginfo(f"Comp freq: {count/dur:.4f} FPS; dur: {dur:.4f}s")
                count = 0

if __name__ == "__main__":
    rospy.init_node("pose_follower")
    rospy.loginfo("Starting pose_follower pipeline.")
    inference = CompNavigation()
    rospy.loginfo("Started pose_follower pipeline.")
    rospy.spin()
    rospy.loginfo("Stopped pose_follower pipeline.")
