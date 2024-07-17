import numpy as np
import torch
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation

def homogeneous(src):
    if isinstance(src, np.ndarray):
        ones = np.ones_like(src[..., [0]])
        homo = np.hstack([src, ones])
    elif isinstance(src, torch.Tensor):
        ones = torch.ones_like(src[..., [0]])
        homo = torch.hstack([src, ones])
    return homo

def move_device(src, *args):
    if isinstance(src, np.ndarray):
        for i, arg in enumerate(args):
            if not isinstance(arg, np.ndarray):
                args[i] = arg.cpu().numpy()
    elif isinstance(src, torch.Tensor):
        for i, arg in enumerate(args):
            if not isinstance(arg, torch.Tensor):
                args[i] = torch.tensor(arg).to(src.device)
    else:
        raise NotImplementedError(f"type(src) not supported")
    return [src] + list(args)


def unproject_depth(points_2d, depths, camera_matrix):
    points_2d, depths, camera_matrix = move_device(points_2d, depths, camera_matrix)
    u = points_2d[..., 0]
    v = points_2d[..., 1]
    fx = camera_matrix[..., 0, 0]
    fy = camera_matrix[..., 1, 1]
    cx = camera_matrix[..., 0, 2]
    cy = camera_matrix[..., 1, 2]
    x_coord = (u - cx) / fx
    y_coord = (v - cy) / fy
    if isinstance(u, np.ndarray):
        ones = np.ones_like(x_coord)
        xyz = np.vstack([x_coord, y_coord, ones]) * depths
    elif isinstance(u, torch.Tensor):
        ones = torch.ones_like(x_coord)
        xyz = torch.vstack([x_coord, y_coord, ones]) * depths
    return xyz.T

def tf_convert(tf: TransformStamped, to_matrix=False):
    """Convert tf msg to transform, quaternine or matrix

    Args:
        tf (TransformStamped): _description_
        to_matrix (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    trans = np.array([
        tf.transform.translation.x,
        tf.transform.translation.y,
        tf.transform.translation.z,
    ])
    quat = np.array([
        tf.transform.rotation.x,
        tf.transform.rotation.y,
        tf.transform.rotation.z,
        tf.transform.rotation.w,
    ])
    if to_matrix:
        mat = np.eye(4)
        mat[:3, -1] = trans
        mat[:3, :3] = Rotation.from_quat(quat).as_matrix()
        return mat
    else:
        return trans, quat

def yaw_to_quat(yaw):
    return Rotation.from_euler("z", yaw).as_quat()
    