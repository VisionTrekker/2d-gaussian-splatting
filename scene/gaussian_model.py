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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import math
from scipy.spatial.transform import Rotation

from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
import open3d as o3d
from scipy.spatial import ConvexHull


def project_points_to_plane(points, plane_normal, plane_point):
    # 计算点到平面的投影
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    points_from_plane = points - plane_point
    projected_points = points - np.dot(points_from_plane, plane_normal[:, None]) * plane_normal
    return projected_points

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        # 边界框
        self.bbox_min = None
        self.bbox_max = None
        self._plane_normal = None
        self._plane_d = None
        self.initial_rotation = None

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        # 以稀疏点云的中心位置作为3D高斯的中心
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        # # 计算点云边界框并拟合平面方程
        # # 计算边界框
        # self.bbox_min = fused_point_cloud.min(dim=0).values
        # self.bbox_max = fused_point_cloud.max(dim=0).values
        # print("bbox_min: {}, {}, {}".format(self.bbox_min[0], self.bbox_min[1], self.bbox_min[2]))
        # print("bbox_max: {}, {}, {}".format(self.bbox_max[0], self.bbox_max[1], self.bbox_max[2]))
        # # 拟合平面方程 ax + by + cz + d = 0 (a,b,c对应法向量)
        # # 方法1：协方差矩阵 拟合平面
        # plane_normal, plane_d = self.fit_plane_by_cov(fused_point_cloud)
        # # 方法2：协方差矩阵 拟合多个平面，选出最优的（未验证）
        # # plane_normal, plane_d = self.fit_plane_by_cov_best(fused_point_cloud)
        # # 方法3：RANSAC 拟合平面（未验证）
        # # plane_normal, plane_d = self.fit_plane_by_ransac(fused_point_cloud.cpu().numpy())
        # # 添加参数：平面方程的法向量和常数项（不可训练）
        # self._plane_normal = nn.Parameter(plane_normal.requires_grad_(False))
        # self._plane_d = nn.Parameter(plane_d.requires_grad_(False))
        #
        # # 将fused_point_cloud投影到平面上
        # distance_to_plane = (torch.sum(fused_point_cloud * plane_normal, dim=1) + plane_d) / torch.norm(plane_normal)
        # fused_point_cloud = fused_point_cloud - distance_to_plane.unsqueeze(1) * plane_normal / torch.norm(plane_normal)
        # fused_point_cloud = torch.clamp(fused_point_cloud, min=self.bbox_min, max=self.bbox_max)
        # # 点云稀疏采样
        # downsampled_indices = self.downsample_points(fused_point_cloud.cpu().numpy(), num_points=6)
        # fused_point_cloud = fused_point_cloud[downsampled_indices].cuda()
        #
        # # 将平面normal转换成四元数，设置给高斯
        # plane_quat = self.quaternion_from_normal_by_axis(plane_normal)
        # rots = plane_quat.repeat(fused_point_cloud.shape[0], 1)
        # # self.initial_rotation = rots.clone().detach()
        # print("rots[0]: {}".format(rots[0]))

        origin_colors = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        # downsampled_colors = origin_colors[downsampled_indices]
        fused_color = RGB2SH(origin_colors)   # (N, 3)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # (N, 3, (最大球谐阶数 + 1)²)
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)    # 初始化2D高斯在两个方向上的缩放因子为 该点与其K个最近邻点的平均距离 (N, 2)
        # 默认的：初始化2D高斯的旋转四元数 为[0,1]的均匀分布，3DGS是无旋转的单位四元数 (N, 4)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
        # 初始化各2D高斯的 不透明度为0.1 (N, 1)
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 将以上需计算的参数设置为模型的可训练参数
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))    # 各2D高斯的中心位置，(N, 3)
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))   # 球谐函数直流分量的系数，(N, 1, 3)
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))  # 球谐函数高阶分量的系数，(N, (最大球谐阶数 + 1)² - 1, 3)
        self._scaling = nn.Parameter(scales.requires_grad_(True))   # 缩放因子，(N, 2)
        self._rotation = nn.Parameter(rots.requires_grad_(True))    # 0,1均匀分布的旋转四元数，(N, 4)
        self._opacity = nn.Parameter(opacities.requires_grad_(True))    # 不透明度，(N, 1)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")  # 投影到所有2D图像平面上的最大半径，初始化为0，(N, )

    def fit_plane_by_cov(self, points):
        centroid = torch.mean(points, dim=0)     # 计算点云的质心
        centered_points = points - centroid      # 将点云中心化
        cov = torch.mm(centered_points.t(), centered_points)# 计算协方差矩阵
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # 对协方差矩阵进行特征值分解
        normal = eigenvectors[:, 0]                   # 最小特征值对应的特征向量就是平面法向量
        d = -torch.dot(normal, centroid)        # 计算平面方程的d值
        return normal / torch.norm(normal), d

    def fit_plane_by_cov_best(self, points, num_planes=3):
        best_normal = None
        best_d = None
        min_error = float("inf")

        for _ in range(num_planes):
            normal, d = self.fit_plane_by_cov(points)
            error = torch.abs(torch.sum(points * normal, dim=1) + d).mean()
            if error < min_error:
                min_error = error
                best_normal = normal
                best_d = d
        return best_normal, best_d

    def fit_plane_by_ransac(self, points):
        from sklearn.linear_model import RANSACRegressor
        # 使用RANSAC拟合平面
        ransac = RANSACRegressor(residual_threshold=0.01)
        ransac.fit(points[:, :2], points[:, :2])
        # 获取平面参数
        a, b = ransac.estimator_.coef_
        c = -1
        d = ransac.estimator_.intercept_

        normal = torch.tensor([a, b, c]).float().cuda()
        normal = normal / torch.norm(normal)
        return normal, torch.sensor(d).float().cuda()

    def quaternion_from_normal_by_Rotation(self, normal):
        r = Rotation.from_rotvec(math.pi / 2 * normal / np.linalg.norm(normal)) # 将法向量为轴，旋转90°，因此计算的旋转四元数 不对，从平行于y轴到平行于x轴
        fix_rotation = torch.from_numpy(r.as_quat()[[3, 0, 1, 2]]).float().cuda()   # 四元数格式调整为 (qw, qx, qy, qz)
        return fix_rotation

    def quaternion_from_normal_by_axis(self, normal):
        z_axis = torch.tensor([0.0, 0.0, 1.0]).float().cuda()
        rotation_axis = torch.cross(z_axis, normal)     # 计算旋转轴
        if torch.allclose(rotation_axis, torch.zeros(3, device=normal.device)):
            rotation_axis = torch.tensor([1.0, 0.0, 0.0]).float().cuda()  # 选择任意正交轴
        rotation_axis = rotation_axis / torch.norm(rotation_axis)

        # 计算旋转角度
        angle = torch.acos(torch.dot(normal, z_axis))
        # qw, qx, qy, qz
        quat = torch.tensor([
            torch.cos(angle / 2),
            torch.sin(angle / 2) * rotation_axis[0],
            torch.sin(angle / 2) * rotation_axis[1],
            torch.sin(angle / 2) * rotation_axis[2]
        ], device=normal.device)
        return quat / torch.norm(quat)

    def downsample_points(self, points, num_points):
        from sklearn.cluster import KMeans
        # 均匀下采样点云
        if len(points) <= num_points:
            return np.arange(len(points))
        # 使用K-means聚类来获取均匀分布的点
        kmeans = KMeans(n_clusters=num_points, random_state=0)
        kmeans.fit(points)

        # 对于每个聚类，选择最接近中心的点
        closest_indices = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(points - center, axis=1)
            closest_index = np.argmin(distances)
            closest_indices.append(closest_index)

        return np.array(closest_indices)

    def force_points_to_plane(self):
        # 允许平面参数微调时开启
        # self._plane_normal.data = self._plane_normal.data / torch.norm(self._plane_normal.data)
        distance_to_plane = (torch.sum(self._xyz * self._plane_normal, dim=1) + self._plane_d) / torch.norm(self._plane_normal)
        self._xyz.data = self._xyz - distance_to_plane.unsqueeze(1) * self._plane_normal / torch.norm(self._plane_normal)
        self._xyz.data = torch.clamp(self._xyz.data, min=self.bbox_min, max=self.bbox_max)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def fit_plane(self, points):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

        # 使用统计滤波去除离群点
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)

        # 使用RANSAC算法拟合平面
        planes = []
        remaining_cloud = pcd

        inliers_counts = []
        areas = []
        best_area = -10
        best_inliers_counts = -1
        best_id = -1
        idx = 0

        # 迭代10次，拟合最多10个平面
        for _ in range(10):
            if len(remaining_cloud.points) < 10:
                break
            # 拟合一个平面,并得到平面模型参数 plane_model 和内点索引 inliers
            plane_model, inliers = remaining_cloud.segment_plane(distance_threshold=0.02, ransac_n=3,
                                                                 num_iterations=1000)
            # 计算平面的面积
            # 提取内点
            inlier_cloud = remaining_cloud.select_by_index(inliers)
            inlier_points = np.asarray(inlier_cloud.points)
            area = 0
            if inlier_points.shape[0] > 3:
                plane_normal = plane_model[:3]  # 平面法向量
                plane_point = np.mean(inlier_points, axis=0)    # 平面中心点
                projected_points = project_points_to_plane(inlier_points, plane_normal, plane_point)    # 内点投影到拟合的平面上
                hull = ConvexHull(projected_points[:, :2])  # 投影到拟合平面后的xy平面
                area = hull.volume  # 凸包的面积
            if np.sum(inliers) > best_inliers_counts and area > best_area:
                best_id = idx
                best_inliers_counts = np.sum(inliers)
                best_area = area
            idx += 1
            planes.append((plane_model, inliers, area))
            remaining_cloud = remaining_cloud.select_by_index(inliers, invert=True)  # 移除当前平面的内点，在下一轮在剩余内点中拟合平面

        best_plane = planes[best_id][0]
        return best_plane

    def save_centerObj_ply(self, path, bbox=None):
        def filter_largest_cluster_dbscan_o3d(points, eps=0.13, min_samples=20):  # eps=0.2 min_samples=20
            # Convert points to Open3D PointCloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Apply DBSCAN clustering
            labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_samples))

            # Find the largest cluster
            unique_labels, counts = np.unique(labels, return_counts=True)
            largest_cluster_label = unique_labels[np.argmax(counts)]

            # Filter points belonging to the largest cluster
            inlier_mask = labels == largest_cluster_label
            filtered_points = np.asarray(pcd.points)[inlier_mask]

            return filtered_points, inlier_mask

        def compute_pca(cloud):
            from sklearn.decomposition import PCA

            # 计算点云的质心
            # centroid = np.mean(cloud, axis=0)
            #
            # # 将点云平移到质心
            # points_centered = cloud - centroid
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            final_points = np.asarray(cl.points)
            points = np.array(final_points)
            pca = PCA(n_components=3)
            pca.fit(points)
            return pca.components_

        obj_path = path.replace('point_cloud.ply', 'centerObj_point_cloud.ply')
        mkdir_p(os.path.dirname(obj_path))

        if bbox is not None:
            xyz = self._xyz.detach().cpu().numpy()
            # find PCA directions
            pca_components = compute_pca(xyz)

            # Calculate the center and radius of the bounding box
            center = np.array([(bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2,
                               (bbox[4] + bbox[5]) / 2
                               ])
            center = np.tile(center, (xyz.shape[0], 1))
            r = max(abs(bbox[0] - bbox[1]), abs(bbox[2] - bbox[3]), abs(bbox[4] - bbox[5])) / 2
            r = r * 1.5

            # Filter points within the circular region
            copy = xyz.copy()
            dis = copy[:, 0] ** 2 - 2 * copy[:, 0] * center[:, 0] + \
                  copy[:, 1] ** 2 - 2 * copy[:, 1] * center[:, 1] + \
                  copy[:, 2] ** 2 - 2 * copy[:, 2] * center[:, 2]

            indx = dis < r ** 2
            xyz = xyz[indx]

            # Fit a plane to the points within the circular region
            if np.sum(self.colmap_plane_model) == 0:
                # normal, d = fit_plane_o3d_PCA_interval(xyz, pca_components)
                # normal, d = fit_plane_o3d_PCA_without_interval(xyz)
                plane = self.fit_plane(xyz)

            else:
                plane = self.colmap_plane_model
            normal = plane[:3]
            d = plane[3]

            # Calculate the distance from each point to the plane
            distances = np.dot(xyz, normal) + d

            # Only keep points above the plane
            # plane_distance = 0.013
            plane_distance = 0.05
            indx_above_plane = distances > plane_distance if len(xyz[distances > plane_distance]) > len(
                xyz[distances < -1 * plane_distance]) else distances < -1 * plane_distance
            # indx_above_plane = distances > -100
            xyz = xyz[indx_above_plane]

            # Get remaining attributes for the filtered points
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()[indx][
                indx_above_plane]
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()[indx][
                indx_above_plane]
            opacities = self._opacity.detach().cpu().numpy()[indx][indx_above_plane]
            scale = self._scaling.detach().cpu().numpy()[indx][indx_above_plane]
            rotation = self._rotation.detach().cpu().numpy()[indx][indx_above_plane]

            # Filter out points from the largest cluster using DBSCAN
            xyz, cluster_mask = filter_largest_cluster_dbscan_o3d(xyz)

            # Filter other attributes based on the largest cluster mask
            f_dc = f_dc[cluster_mask]
            f_rest = f_rest[cluster_mask]
            opacities = opacities[cluster_mask]
            scale = scale[cluster_mask]
            rotation = rotation[cluster_mask]

            # Get remaining attributes for the filtered points
            normals = np.zeros_like(xyz)

            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
        print(f"\tnew ply is saved in {obj_path}")
        PlyData([el]).write(obj_path)

    def save_ply(self, path, bbox=None):
        mkdir_p(os.path.dirname(path))

        if bbox is not None:
            xyz = self._xyz.detach().cpu().numpy()
            center = np.array([(bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2])
            center = np.tile(center, (xyz.shape[0], 1))
            r = max(abs(bbox[0] - bbox[1]), abs(bbox[2] - bbox[3])) / 2
            r = r * 1.5
            copy = xyz.copy()
            dis = copy[:, 0] ** 2 - 2 * copy[:, 0] * center[:, 0] + copy[:, 1] ** 2 - 2 * copy[:, 1] * center[:, 1]
            indx = dis < r ** 2
            normals = np.zeros_like(xyz)[indx]
            xyz = xyz[indx]
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()[indx]
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()[indx]
            opacities = self._opacity.detach().cpu().numpy()[indx]
            scale = self._scaling.detach().cpu().numpy()[indx]
            rotation = self._rotation.detach().cpu().numpy()[indx]
            hist, bins = np.histogram(xyz[:, 2], bins=100)
            a = np.argmax(hist)
            hist = hist[a:]
            bins = bins[a:]
            if np.sum(hist < 100) > 0:
                indx = np.where(hist < 100)[0][0]
                # print(bins[indx])
                v = bins[indx]
                indx_re = xyz[:, 2] < v
                indx_temp = xyz[:, 2] > -0.6
                indx_re = np.logical_and(indx_re, indx_temp)
                normals = normals[indx_re]
                xyz = xyz[indx_re]
                f_dc = f_dc[indx_re]
                f_rest = f_rest[indx_re]
                opacities = opacities[indx_re]
                scale = scale[indx_re]
                rotation = rotation[indx_re]

                dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

                elements = np.empty(xyz.shape[0], dtype=dtype_full)
                attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
                elements[:] = list(map(tuple, attributes))
                el = PlyElement.describe(elements, 'vertex')
            else:
                xyz = self._xyz.detach().cpu().numpy()
                normals = np.zeros_like(xyz)
                f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
                f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
                opacities = self._opacity.detach().cpu().numpy()
                scale = self._scaling.detach().cpu().numpy()
                rotation = self._rotation.detach().cpu().numpy()

                dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

                elements = np.empty(xyz.shape[0], dtype=dtype_full)
                attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
                elements[:] = list(map(tuple, attributes))
                el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(path)
        else:
            xyz = self._xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()

            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(path)


    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # 计算点到平面的距离
        # distance_to_plane = torch.abs(torch.sum(self._xyz * self._plane_normal, dim=1) + self._plane_d) / torch.norm(self._plane_normal)
        # selected_pts_mask = torch.logical_and(selected_pts_mask, distance_to_plane <= 0.5 * (self.bbox_max[2] - self.bbox_min[2]))

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # 计算点到平面的距离
        # distance_to_plane = torch.abs(torch.sum(self._xyz * self._plane_normal, dim=1) + self._plane_d) / torch.norm(self._plane_normal)
        # selected_pts_mask = torch.logical_and(selected_pts_mask, distance_to_plane <= 0.5 * (self.bbox_max[2] - self.bbox_min[2]))
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # 再剔除掉bbox外的点
        # bbox_out_mask = torch.any(self._xyz < self.bbox_min, dim=1) | torch.any(self._xyz > self.bbox_max, dim=1)
        # prune_mask = torch.logical_or(prune_mask, bbox_out_mask)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1