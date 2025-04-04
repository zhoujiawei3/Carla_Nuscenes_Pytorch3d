import hashlib
import numpy as np
from pyquaternion import Quaternion
import json

def transform_timestamp(timestamp):
    return int(timestamp*10e6)

def generate_token(key,data):
    obj = hashlib.md5(str(key).encode('utf-8'))
    obj.update(str(data).encode('utf-8'))
    result = obj.hexdigest()
    return result

def dump(data,path):
    with open(path, "w") as filedata:
        json.dump(data, filedata, indent=0, separators=(',',':'))

def load(path):
    with open(path, "r") as filedata:
        return json.load(filedata)

def get_intrinsic(fov, image_size_x,image_size_y):
    focal = image_size_x / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = image_size_x / 2.0
    K[1, 2] = image_size_y / 2.0
    return K

def get_nuscenes_rt(transform,mode=None):
    translation = [transform.location.x,
                -transform.location.y,
                transform.location.z]
    if mode == "zxy":
        rotation_matrix1 = np.array([#原来的 z 轴变成了新的 x 轴原来的 x 轴变成了新的 y 轴。原来的 y 轴变成了新的 z 轴。
            [0,0,1],
            [1,0,0],
            [0,1,0]
            # [0,1,0],
            # [0,0,1],
            # [1,0,0]
        ])@np.array([
            [1,0,0],
            [0,-1,0],
            [0,0,1]
        ])
    else:
        rotation_matrix1 = np.array([
            [1,0,0],
            [0,-1,0],
            [0,0,1]
        ])

    rotation_matrix2 = np.array(transform.get_matrix())[:3,:3]
    rotation_matrix3 = np.array([
            [1,0,0],
            [0,-1,0],
            [0,0,1]
        ])
    rotation_matrix = rotation_matrix1@rotation_matrix2@rotation_matrix3
    quat = Quaternion(matrix=rotation_matrix,rtol=1, atol=1).elements.tolist()
    return quat,translation
def get_nuscenes_rt_origin(transform,mode=None):
    translation = [transform.location.x,
                transform.location.y,
                transform.location.z]
    if mode == "zxy":
        rotation_matrix1 = np.array([
            [0,0,1],
            [1,0,0],
            [0,1,0]
        ])
        rotation_matrix3 = np.array([
            [1,0,0],
            [0,-1,0],
            [0,0,1]
        ])
        rotation_matrix = rotation_matrix1@rotation_matrix3
    else:
        rotation_matrix = np.array([
            [1,0,0],
            [0,-1,0],
            [0,0,1]
        ])

    rotation_matrix2 = np.array(transform.get_matrix())[:3,:3]
    # print(f"original matrix{rotation_matrix2}\n")
    rotation_matrix = rotation_matrix2
    # print(f"rotation_matrix{rotation_matrix}\n")
    
    quat = Quaternion(matrix=rotation_matrix,rtol=1, atol=1).elements.tolist()
    return quat,translation
def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))