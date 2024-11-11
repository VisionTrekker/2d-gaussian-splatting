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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 image_path=None, depth=None, normal=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid  # 每个相机的id不一样
        self.colmap_id = colmap_id  # 如果COLMAP设置的统一一个相机，则colmap_id只有一个值，一般为1
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path
        self.depth = None
        self.normal = None

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        Fx = fov2focal(FoVx, self.image_width)
        Fy = fov2focal(FoVy, self.image_height)
        Cx = 0.5 * self.image_width
        Cy = 0.5 * self.image_height
        # 从深度图计算法向量图时需要
        self.K = torch.tensor([[Fx, 0, Fy],
                               [0, Cx, Cy],
                               [0, 0, 1]]).to(self.data_device).to(torch.float32)

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        if depth is not None:
            self.depth = depth.to(self.data_device)     # 1 H W
            # print("depth min: {}, max: {}".format(torch.min(self.depth), torch.max(self.depth)))

            # if normal is None:
            #     # 从深度图计算世界坐标系下的法向量
            #     c2w = (self.world_view_transform.T).inverse()  # C2W，cuda
            #     grid_x, grid_y = torch.meshgrid(torch.arange(self.image_width, device='cuda').float(), torch.arange(self.image_height, device='cuda').float(), indexing='xy')  # 生成1个二维网格，分别包含x轴和y轴的坐标，(H, W)
            #     points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)  # H*W个像素点在图像坐标系中的齐次坐标(H*W, 3)，cuda：torch.stack 将grid_x、grid_y和全1的张量按最后一个维度拼接，形状为(H, W, 3)，每个位置的值是像素坐标的齐次形式 (x, y, 1)
            #     # 每个像素点在世界坐标系下的坐标：pix @ pix2C @ C2W，cuda
            #     rays_d = points @ self.K.inverse().T.cuda() @ c2w[:3, :3].T
            #     # 相机光心在世界坐标系中的位置(射线起点)，cuda
            #     rays_o = c2w[:3, 3]
            #
            #     # 3D点云的世界坐标：深度值与射线方向相乘，并加上射线起点，cuda
            #     points = self.depth.cuda().reshape(-1, 1) * rays_d + rays_o # (H*W, 3)
            #     points = points.reshape(*depth.shape[1:], 3)  # 重新调整为 H W 3
            #
            #     output = torch.zeros_like(points)
            #     # 计算世界坐标系下3D点云在x、y方向上的梯度
            #     dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
            #     dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
            #     # 叉乘 得到法向量，然后归一化
            #     normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
            #     output[1:-1, 1:-1, :] = normal_map  # 中心填充
            #     self.normal = output.to(self.data_device).permute(2, 0, 1)  # 3 H W

        if normal is not None:
            self.normal = normal.to(self.data_device)   # 3 H W
            # print("normal min: {}, max: {}".format(torch.min(self.normal), torch.max(self.normal)))

        # 显示深度图
        # if depth is not None and self.normal is not None:
        #     import matplotlib.pyplot as plt
        #     # 将图像从 PyTorch 张量转换为 NumPy 数组
        #     image_np = self.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        #     depth_np = self.depth.detach().cpu().numpy().squeeze(0)
        #     normal_np = self.normal.detach().cpu().numpy().transpose(1, 2, 0)
        #
        #     H, W, C = image_np.shape
        #     new_image = np.zeros((H, W*3, C), dtype=np.uint8)
        #     new_image[:, :W, :] = (image_np * 255).astype(np.uint8)
        #     depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())    # 如果读取png不需要
        #     depth_np = np.repeat(depth_np[:, :, np.newaxis], 3, axis=2)
        #     new_image[:, W:2*W, :] = (depth_np * 255).astype(np.uint8)
        #     normal_np = (normal_np + 1.0) / 2.0
        #     new_image[:, 2*W:, :] = (normal_np * 255).astype(np.uint8)
        #     save_path = "/data2/liuzhi/3DGS_code/2d-gaussian-splatting/output/gm_Museum_0.1gtdepthnormal/image_test" + str(uid) + ".png"
        #     plt.imsave(save_path, new_image)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

