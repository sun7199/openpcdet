import math
import random
from collections import defaultdict
from itertools import combinations

import cv2
import open3d as o3d
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from scipy import stats
import json
import queue


# 加载点云
def load_point_cloud(filename):
    return o3d.io.read_point_cloud(filename)


def visualize_point_cloud(points):
    vis = o3d.visualization.Visualizer()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    vis.add_geometry(point_cloud)
    vis.run()


def fitLineRansac(points, iterations=1000, sigma=0.1, k_min=-7, k_max=7):
    """
    RANSAC to fit a 2D line.

    :param points: Input points as a numpy array [points_num, 2] of np.float32.
    :param iterations: Number of RANSAC iterations.
    :param sigma: Acceptable distance threshold between points and the line.
    :param k_min: Minimum acceptable slope.
    :param k_max: Maximum acceptable slope.

    :return: The slope and intercept of the best fitting line.
    """
    best_score = -1
    best_line = None
    points_num = points.shape[0]

    if points_num < 2:
        return 0, 0  # Not enough points to fit a line

    for _ in range(iterations):
        i1, i2 = random.sample(range(points_num), 2)
        p1 = points[i1]
        p2 = points[i2]

        dp = p1 - p2  # Direction vector of the line

        if np.linalg.norm(dp) == 0:
            continue  # Skip if the points are identical

        # Avoid fitting vertical lines by checking for zero x-difference
        if dp[0] == 0:
            continue

        # Calculate slope (k) and intercept (b)
        k = dp[1] / dp[0]
        if k < k_min or k > k_max:
            continue  # Skip lines with slopes outside the specified range

        b = p1[1] - k * p1[0]

        # Calculate the distance of all points to the line
        distances = np.abs(k * points[:, 0] - points[:, 1] + b) / np.sqrt(k ** 2 + 1)

        # Count inliers (points within the threshold distance)
        inliers = np.sum(distances < sigma)

        if inliers > best_score:
            best_score = inliers
            best_line = (k, b)

    if best_line:
        return best_line
    else:
        return 0, 0  # No valid line found


def euclidean_cluster(points, x_threshold=0.01, y_threshold=0.01,
                      min_points=100):
    current_label = 0
    labels = np.full(points.shape[0], -1, dtype=int)  # Initialize all labels to -1

    for i in range(len(points)):
        if labels[i] == -1:
            temp_labels = []
            current_cluster_index = []
            for j in range(len(points)):
                x_diff = abs(points[i, 0] - points[j, 0])
                y_diff = abs(points[i, 1] - points[j, 1])
                # Use np.divide with where parameter to handle division by zero
                # slot_diff = np.divide(y_diff, x_diff, out=np.zeros_like(y_diff, dtype=float), where=x_diff != 0)

                # Take the absolute value
                # slot_diff = np.abs(y_diff / (x_diff + 0.00001))
                if x_diff < x_threshold and y_diff < y_threshold:
                    current_cluster_index.append(j)
                    if labels[j] != -1:
                        temp_labels.append(labels[j])
            if len(temp_labels) == 0:
                for k in current_cluster_index:
                    labels[k] = current_label
                current_label += 1  # Increment the label only if a new cluster is created
            else:
                temp_labels = np.asarray(temp_labels)
                min_label = np.min(temp_labels)
                for k in current_cluster_index:
                    if labels[k] > min_label:
                        old_label = labels[k]
                        labels[labels == old_label] = min_label
                    labels[k] = min_label  # Ensure current point is labeled

    # Extract unique labels
    unique_labels = np.unique(labels)
    point_clusters = []

    # Create clusters based on labels
    for label in unique_labels:
        if label != -1 and np.sum(labels == label) >= min_points:
            point_cluster = points[labels == label]
            point_clusters.append(point_cluster)

    return point_clusters


# 初步道路边沿检测
def filter_road_edge_points(point_cloud, z_min=-1.0, z_max=1.0):
    # Convert point cloud to a numpy array
    points = np.asarray(point_cloud.points)

    # Filter points based on z values
    points = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]

    # Sort points based on x values
    points = points[points[:, 0].argsort()]

    # Initialize variables
    edge_points = []  # This variable is declared but not used in your original code

    pointClusters = euclidean_cluster(points, x_threshold=1, y_threshold=1,
                                      min_points=100)

    return pointClusters


# 寻找密集的点
def point_with_max_neighbors(points, radius=0.1):
    # Convert the list of points to a numpy array for efficient computation
    points = np.array(points)

    # Initialize a list to hold the count of points within the radius for each point
    counts = []

    # Iterate over each point in the list
    for point in points:
        # Calculate the Euclidean distance from the current point to all other points
        distances = np.linalg.norm(points - point, axis=1)

        # Count the number of points within the specified radius (including the point itself)
        count = np.sum(distances <= radius)

        # Append the count to the list
        counts.append(count)

    # Find the index of the point with the maximum count
    max_index = np.argmax(counts)

    # Return the point with the maximum neighbors and the count
    return points[max_index], counts[max_index]


# 分段寻找最密集的点并连线
def split_points_and_find_points(point_cloud, z_min, z_max, y, interval=10, x_min=-50, x_max=50, radius=0.1):
    """
    Split points by x value into intervals and find the most intense point in each interval.

    :param points: List of points where each point is a list or tuple of coordinates [x, y, z].
    :param interval: Interval size for splitting points by x value.
    :param x_min: Minimum x value for splitting.
    :param x_max: Maximum x value for splitting.
    :param radius: Radius to determine the neighborhood for intensity calculation.
    :return: List of the most intense points in each interval.
    """
    # Convert the list of points to a numpy array
    points = np.asarray(point_cloud.points)
    if y == 'down':
        points = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max) & (points[:, 1] <= 0)]
    if y == 'up':
        points = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max) & (points[:, 1] >= 0)]
    # Split points into bins based on the x-value
    bins = defaultdict(list)
    for point in points:
        x = point[0]
        bin_index = int((x - x_min) // interval)
        bins[bin_index].append(point)

    # Find the most intense point in each bin
    most_intense_points = []
    for bin_points in bins.values():
        if bin_points:
            max_point, max_count = point_with_max_neighbors(bin_points, radius)
            most_intense_points.append(max_point)

    # Sort the most intense points by x value
    most_intense_points.sort(key=lambda point: point[0])
    return most_intense_points


def add_sequential_lines(visualizer, points):
    """
    Add a sequential line connecting points to an Open3D visualizer.

    :param visualizer: The Open3D visualizer instance to add geometries to.
    :param points: List of points where each point is a list or tuple of coordinates [x, y, z].
    """
    # Convert the list of points to a numpy array
    points = np.array(points)

    # Create a PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Create lines connecting each point to the next one in sequence
    lines = [[i, i + 1] for i in range(len(points) - 1)]

    # Create a LineSet object
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Add the point cloud and line set to the visualizer
    visualizer.add_geometry(point_cloud)
    visualizer.add_geometry(line_set)


def point_cloud_to_black_white_image(points, width=500):
    """
    Convert a 2D point cloud to a black and white image.

    :param points: A numpy array of shape (N, 2) where each row is a point (x, y).
    :param width: Width of the output image.
    :param height: Height of the output image.
    :return: Black and white image as a numpy array.
    """
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

    # Calculate the aspect ratio
    aspect_ratio = (y_max - y_min) / (x_max - x_min)

    # Calculate the corresponding y width
    height = int(width * aspect_ratio)
    # Normalize x and y coordinates
    points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min) * width
    points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min) * height

    # Create an empty black and white image
    image = np.zeros((height, width), dtype=np.uint8)

    # Normalize point coordinates to image coordinates
    x = points[:, 0]
    y = height - points[:, 1]

    # Normalize x and y to be within [0, width) and [0, height) respectively
    x = np.clip(x, 0, width - 1).astype(int)
    y = np.clip(y, 0, height - 1).astype(int)

    # Set the corresponding pixels to white (255)
    image[y, x] = 255

    return image


def black_white_image_to_point_cloud(points, image_points, width=500):
    """
    Convert a black-and-white image back to a 2D point cloud.

    :param image_points: A single edge line with two points.

    :return: A numpy array of shape (N, 2) where each row is a point (x, y).
    """

    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    # Calculate the aspect ratio
    aspect_ratio = (y_max - y_min) / (x_max - x_min)

    # Calculate the corresponding height
    height = int(width * aspect_ratio)

    lidar_points = []
    x1, y1, x2, y2 = image_points[0]

    x_indices = np.array([x1, x2])
    y_indices = np.array([y1, y2])
    # Convert the indices back to original point cloud coordinates
    x = x_indices / width * (x_max - x_min) + x_min
    y = (height - y_indices) / height * (y_max - y_min) + y_min

    # Add z-coordinate with value 0
    z = np.zeros_like(x)
    # Combine x and y into a single array of points
    points = np.vstack((x, y, z)).T
    lidar_points.extend(points)

    return lidar_points


def display_ground_by_image(point_cloud, z_min, z_max):
    points = np.asarray(point_cloud.points)

    points = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]
    bw_image = point_cloud_to_black_white_image(points)

    # Display the image
    plt.imshow(bw_image, cmap='gray')
    plt.axis('off')
    plt.show()
    return bw_image


def sobel_filters(image):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = cv2.filter2D(image, -1, Kx)
    Iy = cv2.filter2D(image, -1, Ky)

    G = np.hypot(Ix, Iy)  # Gradient magnitude
    G = G / G.max() * 255  # Normalize to 0-255

    theta = np.arctan2(Iy, Ix)  # Gradient direction

    return G, theta


def non_maximum_suppression(gradient_magnitude, gradient_direction):
    M, N = gradient_magnitude.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = gradient_direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # Angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = gradient_magnitude[i, j + 1]
                    r = gradient_magnitude[i, j - 1]
                # Angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = gradient_magnitude[i + 1, j - 1]
                    r = gradient_magnitude[i - 1, j + 1]
                # Angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = gradient_magnitude[i + 1, j]
                    r = gradient_magnitude[i - 1, j]
                # Angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = gradient_magnitude[i - 1, j - 1]
                    r = gradient_magnitude[i + 1, j + 1]

                if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                    Z[i, j] = gradient_magnitude[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def threshold(image, lowThresholdRatio=0.05, highThresholdRatio=0.15):
    highThreshold = image.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = image.shape
    res = np.zeros((M, N), dtype=np.int32)

    strong_i, strong_j = np.where(image >= highThreshold)
    zeros_i, zeros_j = np.where(image < lowThreshold)

    weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))

    res[strong_i, strong_j] = 255
    res[weak_i, weak_j] = 75

    return res


def hysteresis(image):
    M, N = image.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if image[i, j] == 75:
                if 255 in [image[i + 1, j - 1], image[i + 1, j], image[i + 1, j + 1], image[i, j - 1], image[i, j + 1],
                           image[i - 1, j - 1], image[i - 1, j], image[i - 1, j + 1]]:
                    image[i, j] = 255
                else:
                    image[i, j] = 0
    return image


def detect_lines(image):
    if image.size == 0:
        raise ValueError("Empty image provided for line detection.")

    threshold = 100
    up_edge = None
    down_edge = None
    while up_edge is None or up_edge[0][1] > 130:
        lines = cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=threshold, minLineLength=50, maxLineGap=10)

        # Step 3: Create an image to draw lines on
        line_image = np.zeros_like(image)

        min_y_value = float('inf')
        max_y_value = float('-inf')
        # Step 4: Draw the detected lines
        if lines is not None:

            for line in lines:
                x1, y1, x2, y2 = line[0]  # Extract the coordinates of the line

                # Consider only y-values that are greater than 0
                if y1 > 0 and y1 < min_y_value:
                    min_y_value = y1
                    up_edge = line

                if y1 > 0 and y1 > max_y_value:
                    max_y_value = y1
                    down_edge = line

                if y2 > 0 and y2 < min_y_value:
                    min_y_value = y2
                    up_edge = line

                if y2 > 0 and y2 > max_y_value:
                    max_y_value = y2
                    down_edge = line

            threshold -= 10

    print(up_edge)
    x1, y1, x2, y2 = up_edge[0]
    cv2.line(line_image, (x1, y1), (x2, y2), 255, 1)
    x1, y1, x2, y2 = down_edge[0]
    cv2.line(line_image, (x1, y1), (x2, y2), 255, 1)
    # Step 5: Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(line_image, cmap='gray')
    plt.title('Detected Lines')
    plt.axis('off')
    plt.show()
    return up_edge, down_edge


def get_min_max_y_groups(points):
    """
    Generate all possible pairs of points and select the groups with the minimum
    and maximum y-coordinate.

    Args:
        points (np.array): An Nx2 or Nx3 array where each row is a point [x, y] or [x, y, z].

    Returns:
        tuple: (group_min_y, group_max_y)
            - group_min_y: Group with the minimum y-coordinate.
            - group_max_y: Group with the maximum y-coordinate.
    """
    groups = list(combinations(points, 2))  # Generate all pairs of points

    group_min_y = None
    group_max_y = None
    min_y_value = float('inf')
    max_y_value = float('-inf')

    for group in groups:
        y_values = [p[1] for p in group]  # Extract y-coordinates from the group
        group_min = min(y_values)
        group_max = max(y_values)

        if group_min < min_y_value:
            min_y_value = group_min
            group_min_y = group

        if group_max > max_y_value:
            max_y_value = group_max
            group_max_y = group

    return group_min_y, group_max_y


def calculate_slope_and_intercept(group):
    """
    Calculate the slope (m) and y-intercept (b) of the line formed by two points.

    Args:
        group (tuple): A pair of points [(x1, y1), (x2, y2)].

    Returns:
        tuple: (slope, y_intercept)
            - slope (float): The slope of the line.
            - y_intercept (float): The y-intercept of the line.
    """

    p1, p2 = group
    x1, y1, _ = p1
    x2, y2, _ = p2
    if x2 != x1:
        slope = (y2 - y1) / (x2 - x1)
        y_intercept = y1 - slope * x1
    else:
        slope = float('inf')  # Vertical line
        y_intercept = None  # Undefined for vertical lines
    return slope, y_intercept


def get_roadBoarder(pcd):
    # 道路边沿检测
    x_threshold = 0.5  # 10.0
    y_threshold = 0.05  # 10.0
    z_min = -0.5
    z_max = 0.5
    min_points = 10  # 2

    image = display_ground_by_image(pcd, z_min, z_max)
    up_line, down_line = detect_lines(image)
    points = np.asarray(pcd.points)

    points = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]

    up_points = black_white_image_to_point_cloud(points, up_line)
    up_points = np.asarray(up_points)

    down_points = black_white_image_to_point_cloud(points, down_line)
    down_points = np.asarray(down_points)

    slot, b = calculate_slope_and_intercept(up_points)
    up_edge = np.array([[-50, -50 * slot + b, 0], [50, 50 * slot + b, 0]])

    slot, b = calculate_slope_and_intercept(down_points)

    down_edge = np.array([[-50, 50 * slot + b, 0], [50, 50 * slot + b, 0]])

    return up_edge, down_edge


# 主程序
if __name__ == "__main__":
    # 加载点云
    # pcd = load_point_cloud('data\\report数据集\\102_lidar_33-1\ext\ext_pcd_ego\车多\scene_006\\003-transfered\\1688782475.889312.pcd')
    pcd = o3d.io.read_point_cloud("/home/yueming/Downloads/1V7-PVB/batch_1/lidar/2024-03-05-10-18-40-926.pcd")

    vis = o3d.visualization.Visualizer()
    # 创建窗口，设置窗口标题
    vis.create_window(window_name="point_cloud")
    # 设置点云渲染参数
    opt = vis.get_render_option()
    # 设置背景色（这里为白色）
    opt.background_color = np.array([255, 255, 255])
    # 设置渲染点的大小
    opt.point_size = 3.0
    # 添加点云
    vis.add_geometry(pcd)

    up_edge, down_edge = get_roadBoarder(pcd)

    add_sequential_lines(vis, up_edge)
    add_sequential_lines(vis, down_edge)

    vis.run()
