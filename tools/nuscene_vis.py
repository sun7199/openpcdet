import open3d as o3d
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
import os

# Initialize NuScenes dataset
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/yueming/github projects/OpenPCDet/data/nuscenes/v1.0-trainval', verbose=True)

# Load a sample
sample = nusc.sample[0]  # Get the first sample (you can change this for different samples)

# Load LIDAR point cloud data (LIDAR_TOP)
lidar_token = sample['data']['LIDAR_TOP']
lidar_data = nusc.get('sample_data', lidar_token)
nusc.render_sample_data(lidar_data['token'])

# lidar_filepath = os.path.join(nusc.dataroot, lidar_data['filename'])
#
# # Load the point cloud
# point_cloud = LidarPointCloud.from_file(lidar_filepath)
#
# # Convert the point cloud to Open3D format
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(point_cloud.points[:3, :].T)  # Use only x, y, z coordinates
#
#
# # Create a list of Open3D geometries (point cloud + bounding boxes)
# geometries = [pcd]
#
# # Get ground truth annotations (bounding boxes)
# for ann_token in sample['anns']:
#     ann = nusc.get('sample_annotation', ann_token)
#     instance_token = ann['instance_token']
#
#
#     # Get the bounding box for the annotation
#     box = nusc.get_box(ann['token'])
#
#     # Convert the Box to Open3D geometry (8 corners of the 3D bounding box)
#     corners = box.corners()
#
#     print(corners)
#     # Create Open3D lines for the bounding box
#     lines = [
#         [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
#         [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
#         [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical lines
#     ]
#
#     # Use Open3D's LineSet to draw the bounding box
#     line_set = o3d.geometry.LineSet(
#         points=o3d.utility.Vector3dVector(corners.T),  # Transpose to get (N, 3) shape
#         lines=o3d.utility.Vector2iVector(lines)
#     )
#
#     # Set a random color for each box (or you can set specific colors per category)
#     line_set.colors = o3d.utility.Vector3dVector(np.random.rand(len(lines), 3))
#
#     # Add the box to the list of geometries
#     geometries.append(line_set)
#
# # Visualize the point cloud with bounding boxes using Open3D
# o3d.visualization.draw_geometries(geometries)
