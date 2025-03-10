#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import copy
import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], recon_type='undefined', is_little_object=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.recon_type = recon_type
        self.is_little_object = is_little_object

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        if self.is_little_object:
            print("Fit the plane")
            self.getPlaneModel(scene_info.point_cloud.points, scene_info.point_cloud.tracks)

    def save(self, iteration, center_mode=None):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

        if center_mode is not None:
            train_cameras = self.train_cameras[1.0]
            camera_centers = []
            for idx, camera in enumerate(train_cameras):
                pose = np.array(camera.camera_center.cpu())
                camera_centers.append(pose)
            camera_centers = np.array(camera_centers)

            max_x = camera_centers[:, 0].max()
            min_x = camera_centers[:, 0].min()
            max_y = camera_centers[:, 1].max()
            min_y = camera_centers[:, 1].min()
            max_z = camera_centers[:, 2].max()
            min_z = camera_centers[:, 2].min()

            bbox = np.array([max_x, min_x, max_y, min_y, max_z, min_z])

            if self.is_center_crop:
                self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), bbox)
            if self.is_little_object:
                centerObj_gaussians = copy.deepcopy(self.gaussians)
                centerObj_gaussians.save_centerObj_ply(os.path.join(point_cloud_path, "point_cloud.ply"), bbox)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getPlaneModel(self, points3D, tracks):
        # 设置被多个相机观察到的阈值
        min_observations = 5

        # 过滤点云数据，只保留被多个相机观察到的点
        filtered_points = []
        for point_id in range(points3D.shape[0]):
            if tracks[point_id] >= min_observations:
                filtered_points.append(points3D[point_id])
        if len(filtered_points) > 10:
            self.gaussians.colmap_plane_model = self.gaussians.fit_plane(np.array(filtered_points))
            print('colmap_plane_model is :', self.gaussians.colmap_plane_model[0],
                                              self.gaussians.colmap_plane_model[1],
                                              self.gaussians.colmap_plane_model[2],
                                              self.gaussians.colmap_plane_model[3])

        # 根据相机位姿，拟合一个圆形
        def fit_sphere(x, y, z):
            # 定义目标函数，求解球心和半径
            def calc_R(c):
                Ri = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2)
                return Ri - Ri.mean()

            # 初始猜测球心
            x_m = np.mean(x)
            y_m = np.mean(y)
            z_m = np.mean(z)
            center_estimate = x_m, y_m, z_m
            result = least_squares(calc_R, center_estimate)
            center = result.x
            radius = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2).mean()
            return center, radius

        # 提取 X, Y, Z 坐标
        x = camera_centers[:, 0]
        y = camera_centers[:, 1]
        z = camera_centers[:, 2]

        # 拟合球
        center, radius = fit_sphere(x, y, z)
        self.gaussians.center = center
        self.gaussians.radius = radius
