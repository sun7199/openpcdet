import copy
import math
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import utils
from utils import *


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


def fit_bev(cluster, bbox):
    center_x, center_y, center_z, width_x, width_y, width_z, r = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], \
        -bbox[6]

    R_z = np.array([
        [np.cos(r), -np.sin(r), 0],
        [np.sin(r), np.cos(r), 0],
        [0, 0, 1]
    ])

    plt.figure()
    plt.scatter(cluster[:, 0], cluster[:, 1], s=20, c='b', marker='o')
    plt.title('car BEV')
    cluster = cluster[:, :2]
    y = cluster[:, 1]
    boarder = None

    # plot_pca_direction(cluster)
    # if y.min() > 0:
    #     boarder = cluster[(y > y.min()) & (y < y.min() + 0.2)]
    # elif y.max() < 0:
    #     boarder = cluster[(y < y.max()) & (y > y.max() - 0.2)]

    # line, r = fitLineRansac(cluster)
    #

    # Parameters
    y_interval = 0.1
    x_interval = 0.2
    # Step 1: Find the min and max y values
    y_min = np.min(cluster[:, 1])
    y_max = np.max(cluster[:, 1])
    x_min = np.min(cluster[:, 0])
    x_max = np.max(cluster[:, 0])

    # Step 2: Initialize variables for clustering
    max_cluster = []
    y = y_min

    if center_y - width_y / 2 > 0:
        max_cluster = cluster[(cluster[:, 1] >= y_min) & (cluster[:, 1] < y_min + y_interval)]
    elif center_y + width_y / 2 < 0:
        max_cluster = cluster[(cluster[:, 1] <= y_max) & (cluster[:, 1] > y_max - y_interval)]
    else:
        if center_x - width_x / 2 > 0:
            max_cluster = cluster[(cluster[:, 0] >= x_min) & (cluster[:, 0] < x_min + x_interval)]
        elif center_x + width_x / 2 < 0:
            max_cluster = cluster[(cluster[:, 0] <= x_max) & (cluster[:, 0] > x_max - x_interval)]

    # Step 3: Iterate over the y-coordinates with the specified interval
    # while y < y_max:
    #     # Get points within the current y-interval
    #     current_cluster = cluster[(cluster[:, 1] >= y) & (cluster[:, 1] < y + y_interval)]
    #
    #     # Compare with the maximum cluster found so far
    #     if len(current_cluster) > len(max_cluster):
    #         max_cluster = current_cluster
    #
    #     # Increment y by the interval
    #     y += y_interval

    # Step 4: Apply RANSAC to fit a line to the largest cluster

    # Create a RANSAC Regressor
    if len(max_cluster) > 2:
        line, r = fitLineRansac(max_cluster[:, :2])
    else:
        r = 0
    # if r> np.pi /2:
    #     r= r - np.pi/2
    # elif r<-np.pi /2:
    #     r= r + np.pi/2

    R = np.array([
        [np.cos(-r), -np.sin(-r)],
        [np.sin(-r), np.cos(-r)]
    ])
    cluster = np.dot(R, cluster.T).T

    plt.figure(figsize=(8, 6))
    plt.scatter(cluster[:, 0], cluster[:, 1], alpha=0.6, label='Original Data')
    plt.title('rotate BEV')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

    x = cluster[:, 0]
    y = cluster[:, 1]
    xmax = np.max(x)
    xmin = np.min(x)
    ymax = np.max(y)
    ymin = np.min(y)

    down_cluster_y = cluster[(cluster[:, 1] <= 0)]
    up_cluster_y = cluster[(cluster[:, 1] > 0)]
    down_cluster_x = cluster[(cluster[:, 0] <= 0)]
    up_cluster_x = cluster[(cluster[:, 0] > 0)]

    if xmax - xmin < 3.5:
        if len(down_cluster_x) > len(up_cluster_x):
            xmax = xmin + 3.5
        else:
            xmin = xmax - 3.5

    dimension_x = xmax - xmin
    center_x = (xmax + xmin) / 2
    if ymax - ymin < 1.5:

        if len(down_cluster_y) > len(up_cluster_y):
            ymax = ymin + 1.5
        else:
            ymin = ymax - 1.5

    dimension_y = ymax - ymin
    center_y = (ymax + ymin) / 2

    return center_x, center_y, dimension_x, dimension_y, bbox[6] + r


def fit_side(cluster):
    """
    Fit and rotate the side view of the point cloud, and calculate dimensions and center.

    Args:
        cluster (np.array): Nx3 array representing the point cloud.

    Returns:
        tuple: (rotation_angle, center_z, dimension_z)
    """
    # Scatter plot for original side view (x-z plane)

    plt.figure()

    plt.scatter(cluster[:, 0], cluster[:, 2], s=20, c='b', marker='o')
    plt.title('Car Side View')
    plt.xlabel('X-axis')
    plt.ylabel('Z-axis')

    # Consider only x and z for fitting the side view
    x = cluster[:, 0]
    z = cluster[:, 2]

    # Detect points outside the original x area but within the extended x range, with z near zero
    outside_points = cluster[
        (
                ((x > x.min()) & (x <= x.min() + 2)) |
                ((x < x.max()) & (x >= x.max() - 2))
        ) &
        (z >= -0.2) & (z <= 0.2)
        ]

    # Ensure outside_points is a 2D array with x and z columns
    outside_points_2d = outside_points[:, [0, 2]]

    # Fit a line to the outside points using RANSAC
    r = 0
    if len(outside_points_2d) > 0:
        line, r = fitLineRansac(outside_points_2d)

    # Rotation matrix based on the angle found
    R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])

    # Rotate the cluster
    rotated_cluster = np.dot(R, cluster[:, [0, 2]].T).T

    # Scatter plot after rotation
    plt.figure(figsize=(8, 6))

    plt.scatter(rotated_cluster[:, 0], rotated_cluster[:, 1], alpha=0.6, label='Rotated Data')
    plt.title('Rotated Side View')
    plt.xlabel('Rotated X-axis')
    plt.ylabel('Rotated Z-axis')
    plt.show()

    # Recalculate z based on the rotated cluster
    z_rotated = rotated_cluster[:, 1]

    # Calculate dimensions and center for z
    if z_rotated.max() - z_rotated.min() < 1.5:
        dimension_z = 2
        center_z = z_rotated.min() + 1
    else:
        dimension_z = z_rotated.max() - z_rotated.min()
        center_z = (z_rotated.max() + z_rotated.min()) / 2

    return -r, center_z, dimension_z


def fit_front(cluster):
    """
    Fit and rotate the front view of the point cloud, and calculate the rotation angle.

    Args:
        cluster (np.array): Nx3 array representing the point cloud.

    Returns:
        float: Rotation angle in radians.
    """
    # Scatter plot for original front view (y-z plane)
    plt.figure()
    plt.scatter(cluster[:, 1], cluster[:, 2], s=20, c='b', marker='o')
    plt.title('Car Front View')
    plt.xlabel('Y-axis')
    plt.ylabel('Z-axis')

    # Consider only y and z for fitting the front view
    y = cluster[:, 1]
    z = cluster[:, 2]

    # Detect points outside the original x area but within the extended x range, with y and z near zero
    outside_points = cluster[
        (
                ((y > y.min()) & (y <= y.min() + 1)) |
                ((y < y.max()) & (y >= y.max() - 1))
        ) &
        (z >= -0.2) & (z <= 0.2)
        ]

    # Ensure outside_points is a 2D array with y and z columns
    outside_points_2d = outside_points[:, [1, 2]]

    # Fit a line to the outside points using RANSAC
    r = 0
    if len(outside_points_2d) > 0:
        line, r = fitLineRansac(outside_points_2d)

    # Rotation matrix based on the angle found
    R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])

    # Rotate the cluster
    rotated_cluster = np.dot(R, cluster[:, [1, 2]].T).T

    # Scatter plot after rotation
    plt.figure(figsize=(8, 6))
    plt.scatter(rotated_cluster[:, 0], rotated_cluster[:, 1], alpha=0.6, label='Rotated Data')
    plt.title('Rotated Front View')
    plt.xlabel('Rotated Y-axis')
    plt.ylabel('Rotated Z-axis')
    plt.show()

    return r


def fit_bounding(center, dimension, rotation):
    l, w, h = dimension

    x_corners = [-l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2]
    y_corners = [-h / 2, -h / 2, h / 2, h / 2, -h / 2, -h / 2, h / 2, h / 2]
    z_corners = [-w / 2, -w / 2, -w / 2, -w / 2, w / 2, w / 2, w / 2, w / 2]

    corners = np.vstack([x_corners, y_corners, z_corners])
    # 转换为弧度
    rx, ry, rz = rotation
    rx = -rx
    rz = -rz
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


def plot_pca_direction(point_cloud):
    # Step 1: Apply PCA
    pca = PCA(n_components=2)  # Compute the first two principal components
    pca.fit(point_cloud)
    principal_components = pca.components_  # Principal directions
    mean = pca.mean_  # Mean of the point cloud

    # Step 2: Get the main direction (first principal component)
    main_direction = principal_components[0]  # The first principal component direction

    # Step 3: Normalize the direction vector
    direction_unit = main_direction / np.linalg.norm(
        main_direction)  # Unit vector in the direction of the principal component

    # Step 4: Define the length for visualization based on the extent of the point cloud
    # Find the maximum distance from the center to any point in the direction of the principal component
    distances = np.dot(point_cloud - mean, direction_unit)
    max_distance = np.max(np.abs(distances))  # Maximum distance along the principal direction

    # Define length for visualization (shorter than the maximum distance to ensure it stays within the cluster)
    scale = 0.8  # Adjust this scale to ensure the line doesn't exceed the point cloud
    line_length = scale * max_distance

    # Compute the end point of the direction line
    direction_start = mean
    direction_end = mean + line_length * direction_unit

    # Step 5: Plot the point cloud and the direction line
    fig, ax = plt.subplots()
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], color='blue', label='Points')

    # Plot the direction line
    ax.plot([direction_start[0], direction_end[0]], [direction_start[1], direction_end[1]], color='red', linewidth=2,
            label='Main Direction')

    # Add labels and adjust plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Main Direction of 2D Point Cloud')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')  # Equal scaling of x and y axes for better visualization

    plt.show()


def fitLineRansac(points, iterations=1000, sigma=0.1, k_min=-7, k_max=7):
    """
    RANSAC 拟合2D 直线
    :param points:输入点集,numpy [points_num,1,2],np.float32
    :param iterations:迭代次数
    :param sigma:数据和模型之间可接受的差值,车道线像素宽带一般为10左右
                （Parameter use to compute the fitting score）
    :param k_min:
    :param k_max:k_min/k_max--拟合的直线斜率的取值范围.
                考虑到左右车道线在图像中的斜率位于一定范围内，
                添加此参数，同时可以避免检测垂线和水平线
    :return:拟合的直线参数,It is a vector of 4 floats
                (vx, vy, x0, y0) where (vx, vy) is a normalized
                vector collinear to the line and (x0, y0) is some
                point on the line.
    """
    max_density = -1
    max_density_point = None
    for point in points:
        x = point[0]
        y = point[1]
        filtered_points = points[(x - sigma < points[:, 0]) & (points[:, 0] < x + sigma) &
                                 (y - sigma < points[:, 1]) & (points[:, 1] < y + sigma)]

        # Getting the density (number of points)
        density = filtered_points.shape[0]

        if density > max_density:
            max_density = density
            max_density_point = point

    line = [0, 0, 0, 0]
    points_num = points.shape[0]

    if points_num < 2:
        return line

    bestScore = -1
    # for k in range(iterations):
    #     i1, i2 = random.sample(range(points_num), 2)
    #     p1 = points[i1]
    #     p2 = points[i2]

    for i in range(points_num):
        p1 = points[i]

        p2 = max_density_point

        dp = p1 - p2  # 直线的方向向量

        # dp *= 1. / np.linalg.norm(dp)  # 除以模长，进行归一化

        score = 0

        if dp[1] != 0 and dp[0] != 0:
            for i in range(points_num):
                v = points[i] - p2
                # 向量a与b叉乘/向量b的摸.||b||=1./norm(dp) |p x ab|是 p 和 ab 形成的四边面的面积，那么除以 底边|ab| 就是高，即 p 到 ab 的距离
                dis = v[1] * dp[0] - v[0] * dp[1]
                dis *= 1. / np.linalg.norm(dp)
                if math.fabs(dis) < sigma:
                    score += 1

            if score > bestScore:
                line = [dp[0], dp[1], p1, p2]
                bestScore = score

    slot = line[1] / line[0]
    rad = math.atan(slot)

    if rad < 0:
        rad += 2 * np.pi
    # 判断转角偏向x轴还是y轴
    if np.abs(rad) < np.pi / 4:
        r = rad
    elif np.pi / 4 < rad < 3 * np.pi / 4:
        r = rad - np.pi / 2
    elif 3 * np.pi / 4 < rad < 5 * np.pi / 4:
        r = rad - np.pi
    elif 5 * np.pi / 4 < rad < 7 * np.pi / 4:
        r = rad - 3 * np.pi / 2
    else:
        r = 0
    return line, r


def get_bbox_extents(bbox):
    """Calculate the extents of the bounding box."""
    min_x = bbox[0] - bbox[3] / 2
    max_x = bbox[0] + bbox[3] / 2
    min_y = bbox[1] - bbox[4] / 2
    max_y = bbox[1] + bbox[4] / 2
    min_z = bbox[2] - bbox[5] / 2
    max_z = bbox[2] + bbox[5] / 2
    return (min_x, max_x, min_y, max_y, min_z, max_z)


def is_overlap(bbox1, bbox2):
    """Check if two bounding boxes overlap."""
    ext1 = get_bbox_extents(bbox1)
    ext2 = get_bbox_extents(bbox2)

    return not (
            ext1[1] < ext2[0] or  # bbox1 is completely left of bbox2
            ext1[0] > ext2[1] or  # bbox1 is completely right of bbox2
            ext1[3] < ext2[2] or  # bbox1 is completely below bbox2
            ext1[2] > ext2[3] or  # bbox1 is completely above bbox2
            ext1[5] < ext2[4] or  # bbox1 is completely behind bbox2
            ext1[4] > ext2[5]  # bbox1 is completely in front of bbox2
    )


def points_in_bbox(bbox, pointcloud):
    """Calculate how many points are within a bounding box."""
    ext = get_bbox_extents(bbox)
    # Select points within the bbox extents
    within_bbox = (
            (pointcloud[:, 0] >= ext[0]) & (pointcloud[:, 0] <= ext[1]) &
            (pointcloud[:, 1] >= ext[2]) & (pointcloud[:, 1] <= ext[3]) &
            (pointcloud[:, 2] >= ext[4]) & (pointcloud[:, 2] <= ext[5])
    )
    return np.sum(within_bbox)


def filter_bboxes(bboxes, labels, pointcloud):
    """Filter bounding boxes based on overlap, keeping the bbox with the most points."""
    filtered_bboxes = []
    filtered_labels = []

    for i, bbox in enumerate(bboxes):
        if bbox is None:
            continue  # Skip if the bbox has already been discarded

        max_points = points_in_bbox(bbox, pointcloud)
        keep_bbox = True

        for j in range(i + 1, len(bboxes)):
            if bboxes[j] is None:  # Skip if the bbox has been marked as None
                continue

            if is_overlap(bbox, bboxes[j]):
                points_in_j = points_in_bbox(bboxes[j], pointcloud)

                if points_in_j > max_points:
                    # bbox j has more points, so discard the current bbox
                    keep_bbox = False
                    break
                else:
                    # bbox i has more points, so discard bbox j
                    bboxes[j] = None  # Mark for deletion

        if keep_bbox:
            filtered_bboxes.append(bbox)
            filtered_labels.append(labels[i])

    return np.array(filtered_bboxes), np.array(filtered_labels)


def adjust_bbox(bbox, label, point_cloud):
    """
    Adjust the bounding box based on the point cloud data.

    Args:
        bbox (list): The bounding box [center_x, center_y, center_z, width_x, width_y, width_z].
        point_cloud (np.array): The point cloud as an Nx3 array with columns [x, y, z].

    Returns:
        list: The adjusted bounding box.
    """

    translation_x, translation_y, translation_z, dimension_x, dimension_y, dimension_z, r = bbox[0], bbox[1], bbox[2], \
    bbox[3], bbox[
        4], bbox[5], \
        -bbox[6]

    # Calculate the extents of the bbox
    min_x = - dimension_x / 2
    max_x = dimension_x / 2
    min_y = -dimension_y / 2
    max_y = dimension_y / 2

    R_z = np.array([
        [np.cos(r), -np.sin(r), 0],
        [np.sin(r), np.cos(r), 0],
        [0, 0, 1]
    ])

    cluster = copy.deepcopy(point_cloud)
    cluster -= np.array([translation_x, translation_y, 0])
    cluster = np.dot(R_z, cluster.T).T

    if translation_x < 50:
        points_in_bbox = cluster[
            (cluster[:, 0] >= min_x) & (cluster[:, 0] <= max_x) &
            (cluster[:, 1] >= min_y) & (cluster[:, 1] <= max_y) & (cluster[:, 2] >= -0.1) & (
                    cluster[:, 2] <= 4)]

        points_in_bbox_bev = points_in_bbox[(points_in_bbox[:, 2] >= 0.2)]

    else:
        points_in_bbox = cluster[
            (cluster[:, 0] >= min_x) & (cluster[:, 0] <= max_x) &
            (cluster[:, 1] >= min_y) & (cluster[:, 1] <= max_y) & (cluster[:, 2] <= 4)]
        points_in_bbox_bev = points_in_bbox
    # Filter points within the bbox in the x and y dimensions

    if points_in_bbox_bev.size <= 4:
        return None  # No points found in bbox area

    center_x, center_y, dimension_x, dimension_y, r_z = fit_bev(points_in_bbox_bev, bbox)

    R_z = np.array([
        [np.cos(r_z), -np.sin(r_z), 0],
        [np.sin(r_z), np.cos(r_z), 0],
        [0, 0, 1]
    ])

    # new_min_x = np.min(points_in_bbox_bev[:, 0])
    # new_max_x = np.max(points_in_bbox_bev[:, 0])
    # new_min_y = np.min(points_in_bbox_bev[:, 1])
    # new_max_y = np.max(points_in_bbox_bev[:, 1])
    # new_min_z = np.min(points_in_bbox[:, 2])
    # new_max_z = np.max(points_in_bbox[:, 2])
    #
    # translation_x = 0
    # translation_y = 0
    #
    # if new_max_z - new_min_z > 1.5:
    #     center_z = (new_min_z + new_max_z) / 2
    #     dimension_z = new_max_z - new_min_z
    # else:
    #     new_max_z = new_min_z + 1.5
    #     center_z = (new_min_z + new_max_z) / 2
    #     dimension_z = new_max_z - new_min_z
    #
    # if abs(bbox[6]) < 1 or abs(bbox[6]) > 3:
    #     if label == 3:
    #         center_x = (new_min_x + new_max_x) / 2
    #         dimension_x = new_max_x - new_min_x
    #         center_y = (new_min_y + new_max_y) / 2
    #         dimension_y = new_max_y - new_min_y
    #     else:
    #
    #         if new_max_x - new_min_x > 3.5:
    #             dimension_x = new_max_x - new_min_x
    #             center_x = (new_max_x + new_min_x) / 2
    #
    #         if new_max_y - new_min_y > 1.5:
    #             dimension_y = new_max_y - new_min_y
    #             center_y = (new_max_y + new_min_y) / 2

    box_cors = get_3d_box_fromarray([center_x, center_y, bbox[2]], [dimension_x, dimension_y, bbox[5]], [0,0,0])

    box_cors=np.dot(R_z, box_cors.T).T
    box_cors+=np.array([translation_x, translation_y, 0])

    center_x=(np.min(box_cors[:, 0])+np.max(box_cors[:, 0]))/2
    center_y=(np.min(box_cors[:, 1])+np.max(box_cors[:, 1]))/2
    bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[
        5], bbox[
        6] = center_x, center_y, bbox[2], dimension_x, dimension_y, bbox[5], r_z

    # Return the adjusted bounding box
    return bbox


def adjust_all_bboxes(bboxes, labels, point_cloud):
    """
    Adjust all bounding boxes in the list based on the point cloud data.

    Args:
        bboxes (list): A list of bounding boxes.
        point_cloud (np.array): The point cloud as an Nx3 array with columns [x, y, z].

    Returns:
        list: The list of adjusted bounding boxes.
    """
    adjusted_bboxes = []
    for bbox, label in zip(bboxes, labels):
        adjusted_bbox = adjust_bbox(bbox, label, point_cloud)
        if adjusted_bbox is not None:
            adjusted_bboxes.append(adjusted_bbox)
    return adjusted_bboxes
