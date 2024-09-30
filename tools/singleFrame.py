import cv2 as cv
import torch
import torch.nn as nn
import prcnn
import numpy as np
import open3d as o3d
import os
import ssl
from utils import *
from utils import get_3d_box_fromarray
import fit_bbox
from road_boarder import get_roadBoarder
from fit_bbox import filter_bboxes, adjust_all_bboxes


# 分别显示2d，3d的ground truth
def singleFrame(folderpath, pcdpath, frame):

    pcd = o3d.io.read_point_cloud(pcdpath)
    # 创建窗口对象
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
    # vis.add_geometry(pcd)

    points = np.array(pcd.points)

    up_edge, down_edge = get_roadBoarder(pcd)

    # points = points[(points[:, 1] > down_edge[0][1]) & (points[:, 1] < up_edge[0][1])]
    # if up_edge[0][1] < 0:
    #     up_edge[:, 1] = 20

    num_points = points.shape[0]
    # add the fourth dimension spare value to ensure the custom data compatible with kitti
    new_points = np.zeros((num_points, 5))

    # Copy the x, y, z coordinates to the first three columns
    new_points[:, :3] = points
    # new_points[:, 2] -= 1.6
    # Set the fourth dimension to 0
    new_points[:, 3] = 0
    new_points[:, 4] = 0

    np.save('output.npy', new_points)

    preds = prcnn.main(
        '/home/yueming/OpenPCDet/tools/cfgs/nuscenes_models/transfusion_lidar.yaml',
        'output.npy',
        '/home/yueming/OpenPCDet/tools/models/nuscene/cbgs_transfusion_lidar.pth')

    labels = preds['pred_labels'].cpu().numpy()
    scores = preds['pred_scores'].cpu().numpy()
    bboxs = preds['pred_boxes'].cpu().numpy()
    candidate_boxs = []
    candidate_labels = []
    filter_boxs = []
    objects = []
    ID = 1
    down_edge[0][1] = -9
    up_edge[0][1] = 20
    points=points[(points[:,1]<up_edge[0][1]) & (points[:,1]>down_edge[0][1])]
    point_cloud=o3d.geometry.PointCloud()
    point_cloud.points=o3d.utility.Vector3dVector(points)
    vis.add_geometry(point_cloud)
    for bbox, label, score in zip(bboxs, labels, scores):
        if label == 1 or label == 2 or label == 3:

            if down_edge[0][1] < bbox[1] < up_edge[0][1]:
                candidate_boxs.append(bbox)
                candidate_labels.append(label)
                origin_bbox = bbox
                rz = -bbox[6]

    filter_boxs, filter_labels = filter_bboxes(candidate_boxs, candidate_labels,points)

    adjusted_boxs = adjust_all_bboxes(filter_boxs, filter_labels, points)
    for bbox, label in zip(adjusted_boxs, filter_labels):

        box_cors = get_3d_box_fromarray([bbox[0], bbox[1], bbox[2]], [bbox[3], bbox[4], bbox[5]],
                                        [0, 0, bbox[6]])

        x = box_cors[:, 0]
        y = box_cors[:, 1]
        z = box_cors[:, 2]
        cluster = []

        for point in points:
            if x.min() < point[0] < x.max() and y.min() < point[1] < y.max() and 0 < point[
                2] < z.max():
                cluster.append(point)
        cluster = np.array(cluster)
        rx = 0
        ry = 0
        lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                              [0, 4], [1, 5], [2, 6], [3, 7]])
        # 设置点与点之间线段的颜色
        colors = np.array([[0, 1, 0] for j in range(len(lines_box))])
        # 创建Bbox候选框对象
        line_set = o3d.geometry.LineSet()
        # 将八个顶点连接次序的信息转换成o3d可以使用的数据类型
        line_set.lines = o3d.utility.Vector2iVector(lines_box)
        # 设置每条线段的颜色
        line_set.colors = o3d.utility.Vector3dVector(colors)
        # 把八个顶点的空间信息转换成o3d可以使用的数据类型
        line_set.points = o3d.utility.Vector3dVector(box_cors)
        # 将矩形框加入到窗口中
        vis.add_geometry(line_set)

        obj = {"shapeType": "cube", "static": False, "box2d": {}, "box3d": {
            "generateMode": 1,
            "center": {
                "x": 124.87528916911066,
                "y": 14.046987558348778,
                "z": 3.178713188295765
            },
            "rotation": {
                "x": 0,
                "y": 0,
                "z": 3.141592653589793
            },
            "isExist": True,
            "isMove": True,
            "content": {
                "Motion": [
                    "static"
                ],
                "ID-2": "",
                "occulsion": "0",
                "subclass": "normal",
                "Lane": "on_Lane/001/L1",
                "CIPO": "no",
                "truncation": "0"
            },
            "dimensions": {
                "length": 5,
                "width": 1.8,
                "height": 1.5
            },
            "quality": {
                "errorType": {
                    "attributeError": [],
                    "targetError": [],
                    "otherError": ""
                },
                "changes": {
                    "remark": "",
                    "attribute": [],
                    "target": []
                },
                "qualityStatus": "unqualified"
            }
        }, "label": "car", "objectId": 28}

        # Modify the values for "center," "dimensions," and "rotation" for each object

        obj['box3d']['center']['x'] = float(bbox[0])  # Modify x-coordinate of center
        obj['box3d']['center']['y'] = float(bbox[1])  # Modify y-coordinate of center
        obj['box3d']['center']['z'] = float(bbox[2])  # Modify z-coordinate of center

        obj['box3d']['dimensions']['length'] = float(bbox[3])  # Modify length
        obj['box3d']['dimensions']['width'] = float(bbox[4])  # Modify width
        obj['box3d']['dimensions']['height'] = float(bbox[5])  # Modify height

        obj['box3d']['rotation']['x'] = float(0)  # Modify x-rotation
        obj['box3d']['rotation']['y'] = float(0)  # Modify y-rotation
        obj['box3d']['rotation']['z'] = float(bbox[6])  # Modify z-rotation

        if bbox[5] > 2:  # if height >2
            obj['label'] = 'trunk'
        if label == 3:
            obj['label'] = 'motorcycle'
        obj['objectId'] = ID
        ID += 1
        objects.append(obj)

    border_points = [up_edge, down_edge]

    for boarder_point in border_points:
        # Define a line between the points
        lines = np.array([[0, 1]])

        # Create a LineSet object
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(boarder_point)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # Optionally, set the color of the line
        colors = [[0, 1, 0]]  # Red color for the line
        line_set.colors = o3d.utility.Vector3dVector(colors)

        # Add the LineSet to the visualizer
        # vis.add_geometry(line_set)
    # write road border to json
    for border_point in border_points:
        obj = {
            "shapeType": "line",
            "static": False,
            "box2d": {},
            "box3d": {
                "generateMode": 1,
                "coordinates": [
                    [
                        [
                            1.8565764156245046,
                            2.3099910998639093,
                            0
                        ],
                        [
                            55.43836510450844,
                            2.9140614945973184,
                            0
                        ]
                    ]
                ],
                "isExist": True,
                "content": {},
                "quality": {
                    "errorType": {
                        "attributeError": [],
                        "targetError": [],
                        "otherError": ""
                    },
                    "changes": {
                        "remark": "",
                        "attribute": [],
                        "target": []
                    },
                    "qualityStatus": "unqualified"
                }
            },
            "label": "Road-boundary",
            "objectId": 22
        }

        # Modify the values for "center," "dimensions," and "rotation" for each object
        obj['box3d']["coordinates"] = [border_point.tolist()]  # Modify x-coordinate of center

        obj['objectId'] = ID
        ID += 1
        objects.append(obj)

    # vis.run()
    with open("/home/yueming/Downloads/sample.json", 'r', encoding='gbk', errors='replace') as file:

        data = json.load(file)
    data['objects'] = objects

    data["frameId"] = frame

    folderpath = "/home/yueming/Downloads/output{}".format(folderpath)

    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    # pcdname = os.path.basename(pcd_path)
    jsonpath = pcdpath.replace(".pcd", ".json")
    print(folderpath)
    jsonpath = '/home/yueming/Downloads/output{}'.format(jsonpath)
    print(jsonpath)
    with open(jsonpath, 'w') as file:
        json.dump(data, file, indent=2)

    copy_file(pcdpath, '/home/yueming/Downloads/output{}'.format(pcdpath))


if __name__ == "__main__":
    folder_path = "/home/yueming/Downloads/1V7-PVB/batch_1"

    img_folder = "/home/yueming/Downloads/1V7-PVB/batch_1"
    img_files = os.listdir(img_folder)

    frame = 1

    for root, dirs, files in os.walk(folder_path):

        if files:
            files = sorted(files)
            if files[0].endswith("pcd"):
                for i in range(0, len(files)):
                    file1 = files[i]
                    # file2 = files[i + 1] if i + 1 < len(files) else None
                    # if i == 0:
                    #     files3 = img_files[0]
                    # else:
                    #     index = int(i / 2)
                    #     files3 = img_files[index]s
                    pcd_path = os.path.join(root, file1) if file1 else None
                    pcd_path = pcd_path.replace("\\", "/")
                    # json_path = os.path.join(root, file1)
                    # json_path = json_path.replace("\\", "/")
                    # img_path = os.path.join(img_folder, files3)
                    # img_path = img_path.replace("\\", "/")

                    root = root.replace("\\", "/")

                    singleFrame(root, pcd_path, frame)
                    print("frame:", frame)
                    frame += 1
