import glob
import os
import shutil

import numpy as np
import open3d as o3d
import utils
import pandas
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import numpy as np
import json
from scipy.linalg import pinv
from pyquaternion import Quaternion
import cv2 as cv

"""
# Function to create a dictionary from input data
# Input data in the specified format
input_data = 
center: 1,2,3
dimension: 5,2,1.8
rotation: 0,-1.5,0
"""


def copy_file(source_path, destination_path):
    """
    Copy a file from source_path to destination_path.

    Parameters:
    - source_path: The path to the source file.
    - destination_path: The path where the file will be copied.
    """
    shutil.copyfile(source_path, destination_path)


def create_dictionary(input_data):
    # Split the input data into lines
    lines = input_data.split('\n')

    # Initialize an empty dictionary
    data = {}

    # Process each line
    for line in lines:
        # Split the line into key and value
        key, values = line.split(':')

        # Split the values into x, y, z components
        components = [float(val.strip()) for val in values.split(',')]

        # Update the dictionary with the key and components
        data[key.strip()] = {"x": components[0], "y": components[1], "z": components[2]}

    return data


def write_json(input_list):
    # Convert the input list to a list of dictionaries
    output_list = input_list

    # Save the list of dictionaries to a JSON file
    json_file_path = "kitty_open3d/output.json"
    with open(json_file_path, "w") as json_file:
        json.dump(output_list, json_file, indent=4)


def dbscan(points):
    epsilon = 0.5
    min_samples = 4
    # Compute DBSCAN
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(points)
    labels = db.labels_

    no_clusters = len(np.unique(labels))
    no_noise = np.sum(np.array(labels) == -1, axis=0)

    clusters = []
    for j in np.unique(labels):
        if j != -1:
            cluster = []
            for i in range(len(labels)):
                if labels[i] == j:
                    cluster.append(points[i])
            clusters.append(cluster)
    return clusters
    # print('Estimated no. of clusters: %d' % no_clusters)
    # print('Estimated no. of noise points: %d' % no_noise)


def threshold_cluster(Data_set, threshold):
    # 统一格式化数据为一维数组
    stand_array = np.asarray(Data_set).ravel('C')
    stand_Data = pandas.Series(stand_array)
    index_list, class_k = [], []
    while stand_Data.any():
        if len(stand_Data) == 1:
            index_list.append(list(stand_Data.index))
            class_k.append(list(stand_Data))
            stand_Data = stand_Data.drop(stand_Data.index)
        else:
            class_data_index = stand_Data.index[0]
            class_data = stand_Data[class_data_index]
            stand_Data = stand_Data.drop(class_data_index)
            if (abs(stand_Data - class_data) <= threshold).any():
                args_data = stand_Data[abs(stand_Data - class_data) <= threshold]
                stand_Data = stand_Data.drop(args_data.index)
                index_list.append([class_data_index] + list(args_data.index))
                class_k.append([class_data] + list(args_data))
            else:
                index_list.append([class_data_index])
                class_k.append([class_data])
    return index_list, class_k


def get_3d_box_fromjson(center, dimensions, rotation):
    l, w, h = dimensions["length"], dimensions["width"], dimensions["height"]

    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners = np.vstack([x_corners, y_corners, z_corners])
    # 转换为弧度
    rx, ry, rz = np.radians(rotation["x"]), np.radians(rotation["y"]), np.radians(rotation["z"])

    # 旋转矩阵
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

    R = Rz * Ry * Rx
    corners_rotated = np.dot(R, corners).T

    # 平移到中心点
    if isinstance(center, list):
        corners_rotated += np.array(center)
    else:
        corners_rotated += np.array([center["x"], center["y"], center["z"]])

    return corners_rotated


def get_3d_box_fromarray(center, dimensions, rotation):
    l, w, h = dimensions

    x_corners = [-l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2]
    y_corners = [-w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2]
    z_corners = [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]

    corners = np.vstack([x_corners, y_corners, z_corners])
    # 转换为弧度
    rx, ry, rz = rotation

    # 旋转矩阵
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

    R = np.dot(Rx, np.dot(Rz, Ry))
    corners_rotated = np.dot(R, corners).T

    # 平移到中心点
    if isinstance(center, list):
        corners_rotated += np.array(center)
    else:
        corners_rotated += np.array([center[0], center[1], center[2]])

    return corners_rotated


def read_json(filepath):
    with open(filepath, "rb") as file:
        json_data = json.load(file)
    return json_data


def find_border(pointcloud):
    epsilon_x = 0.02
    epsilon_y = 0.01
    min_samples = 300
    border = []
    temp = [pointcloud[0]]
    for i in range(1, len(pointcloud)):
        temp.append(pointcloud[i])
        if (abs(pointcloud[i][0] - pointcloud[i - 1][0]) > epsilon_x or
                abs(pointcloud[i][1] - pointcloud[i - 1][1]) > epsilon_y):
            border.append(np.array(temp))
            temp = []
    return np.array(border)


def get_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points represented as dictionaries.

    Parameters:
    - dict1: Dictionary representing the coordinates of the first point.
    - dict2: Dictionary representing the coordinates of the second point.

    Returns:
    - distance: Euclidean distance between the two points.
    """
    # Convert dictionary values to NumPy arrays
    # point1 = np.array(list(dict1.values()))
    # point2 = np.array(list(dict2.values()))

    # Calculate Euclidean distance
    # distance = np.linalg.norm(point2 - point1)
    distance = abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    return distance


def compute_2D_box(dimension, center):
    '''
    Return:3Xn in cam2 coordinate
    '''
    w, h = dimension
    x, y = center

    # 计算4个顶点坐标
    x_corners = [w / 2, w / 2, -w / 2, -w / 2]
    y_corners = [h / 2, -h / 2, -h / 2, h / 2]
    # 使用旋转矩阵变换坐标
    corners_2d = np.vstack([x_corners, y_corners])
    # 最后在加上中心点
    corners_2d += np.vstack([x, y])
    corners_2d = corners_2d.astype(int)
    return corners_2d.T


"""
    Parameters:
    -extrinsic: camera extrinsic matrix
    -intrinsic: camera intrinsic matrix
    -lidar:lidar extrinsic matrix
"""


def transform_3d_to_2d_point_cloud(point_cloud_3d, extrinsic, intrinsic, lidar, projection_matrix):
    # Transform 3D point to camera coordinate system
    point_cloud_3d_homogeneous = np.hstack((point_cloud_3d, np.ones(1)))
    point_cloud_3d_homogeneous = np.array([point_cloud_3d_homogeneous])

    point_cloud_lidar = np.dot(pinv(lidar), point_cloud_3d_homogeneous.T)

    point_cloud_lidar_homogeneous = np.vstack((point_cloud_lidar, np.ones((1, 1))))

    point_cloud_cam = np.dot(pinv(extrinsic), point_cloud_lidar_homogeneous)

    # Apply projection
    point_2d_homogeneous = np.dot(intrinsic, point_cloud_cam)

    # Perform perspective division to get pixel coordinates
    point_2d = point_2d_homogeneous[:2] / point_2d_homogeneous[2]

    return point_2d


def compute_2d_camera(box_3d):
    box_3d = box_3d.T
    # Given quaternion components
    x = 0.00135415
    y = -5.48873e-05
    z = 0.00328209
    w = 0.999994

    # Normalize the quaternion
    # 从lidar frame转换到ego vehicle frame
    quaternion = Quaternion([w, x, y, z])

    lidar_translation = np.array([1.8432562987670762, -0.0030671341903952447, 1.7010926506264883])
    lidar_extrinsic = quaternion.rotation_matrix

    cam_intrinsic = np.array([[1908.77423, 0, 1866.30384],
                              [0, 1909.10843, 1096.59674],
                              [0, 0, 1]])

    # box_3d = np.dot(lidar_extrinsic, box_3d)
    # for i in range(box_3d.shape[1]):
    #     box_3d[:, i] += lidar_translation

    # compute camera extrinsic matrix
    x = 0.50287847704854882
    y = -0.48973278755948968
    z = 0.50355094531918976
    w = -0.50369780581087953

    quaternion = Quaternion([w, x, y, z])

    cam_translation = np.array([0.3302634214315821, -0.14661258754289822, -0.061445358344051154])
    cam_extrinsic = quaternion.rotation_matrix.T

    for i in range(box_3d.shape[1]):
        box_3d[:, i] -= cam_translation
    box_3d = np.dot(cam_extrinsic, box_3d)

    points = np.dot(cam_intrinsic, box_3d)
    points /= points[2, :]
    points = points[:2, :]
    points = points.astype(int)

    box_2d = points

    #  projection matrix
    #
    # projection_matrix = np.array(
    #     [[1703.44751, 0, 943.25427300000001, 0], [0, 1896.693115, 444.13498499999997, 0], [0, 0, 1, 0]
    #      ])
    # points = np.vstack((box_3d, np.ones((1, box_3d.shape[1]))))
    # print(points)
    # points = np.dot(projection_matrix, points)
    # points /= points[2, :]
    # points = points[:2, :]
    # points = points.astype(int)
    # print(points)
    # box_2d = points

    return box_2d


def compute_2d_camera_test(box_3d):
    # Given quaternion components
    x = -0.004915791273042756
    y = -0.01656425824108485
    z = 0.0
    w = 0.9998507190301373

    # Normalize the quaternion
    # 从lidar frame转换到ego vehicle frame
    quaternion = Quaternion([w, x, y, z])

    lidar_translation = np.array([1.819821526976407, 0.0036075834256731427, 1.7944077639129257])
    lidar_extrinsic = quaternion.rotation_matrix

    # 先做个转置
    RT = np.transpose(lidar_extrinsic)

    # 再求罗德里德斯变换
    lidar_extrinsic = cv.Rodrigues(RT)[0]

    lidar_translation = np.float64([-lidar_translation[2], lidar_translation[0], lidar_translation[1]])

    camera_matrix = np.array([[1908.77423, 0, 1866.30384],
                              [0, 1909.10843, 1096.59674],
                              [0, 0, 1]])

    # compute camera extrinsic matrix
    x = -0.51878351271986445
    y = 0.49881635954671755
    z = -0.4769199304129057
    w = 0.50457237969745006

    quaternion = Quaternion([w, x, y, z])

    cam_translation = np.array([0.31663917188438107, -0.13415676223430298, -0.079411571533359496])
    cam_extrinsic = quaternion.rotation_matrix.T

    # 先做个转置
    RT = np.transpose(cam_extrinsic)

    # 再求罗德里德斯变换
    rvec = cv.Rodrigues(RT)[0]

    tvec = np.float64([-cam_translation[2], cam_translation[0], cam_translation[1]])

    # 相机形变矩阵
    distCoeffs = np.float64([])

    box_2d, _ = cv.projectPoints(box_3d, rvec, tvec, camera_matrix, distCoeffs)

    #  projection matrix
    #
    # projection_matrix = np.array(
    #     [[1703.44751, 0, 943.25427300000001, 0], [0, 1896.693115, 444.13498499999997, 0], [0, 0, 1, 0]
    #      ])
    # points = np.vstack((box_3d, np.ones((1, box_3d.shape[1]))))
    # print(points)
    # points = np.dot(projection_matrix, points)
    # points /= points[2, :]
    # points = points[:2, :]
    # points = points.astype(int)
    # print(points)
    # box_2d = points

    return np.array(box_2d[0])


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set


def filter_tooNear(objects):
    pre_point = None
    distance = None
    removeList = []
    for i in range(0, len(objects)):
        for j in range(0, len(objects)):
            if j != i:
                if abs(objects[j][0] - objects[i][0]) <= 5 and abs(
                        objects[j][0] - objects[i][0]) <= 2 and abs(objects[j][0] - \
                                                                    objects[i][0]) <= 1.8:
                    if objects[j][0] < objects[i][0]:
                        removeList.append(i)
                    else:
                        removeList.append(j)
    removeList = list(set(removeList))
    removeList.sort(reverse=True)
    for index in removeList:
        objects = np.delete(objects, index, axis=0)
    return objects


def get_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points represented as dictionaries.

    Parameters:
    - dict1: Dictionary representing the coordinates of the first point.
    - dict2: Dictionary representing the coordinates of the second point.

    Returns:
    - distance: Euclidean distance between the two points.
    """
    # Convert dictionary values to NumPy arrays
    # point1 = np.array(list(dict1.values()))
    # point2 = np.array(list(dict2.values()))

    # Calculate Euclidean distance
    # distance = np.linalg.norm(point2 - point1)
    distance = abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    return distance
