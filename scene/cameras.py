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
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

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

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

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
            self.depth = depth.to(self.data_device)
        if normal is not None:
            self.normal = normal.to(self.data_device)   # 3 H W
            # print("camera normal size: {}".format(self.normal.size()))

            # # 显示深度图
            # import matplotlib.pyplot as plt
            # # 将图像从 PyTorch 张量转换为 NumPy 数组
            # image_np = self.original_image.detach().cpu().numpy().transpose(1, 2, 0)
            # normal_np = self.normal.detach().cpu().numpy().transpose(1, 2, 0)
            #
            # H, W, C = image_np.shape
            # new_image = np.zeros((H, W*2, C), dtype=np.uint8)
            # new_image[:, :W, :] = (image_np * 255).astype(np.uint8)
            # normal_np = (normal_np + 1.0) / 2.0
            # new_image[:, W:, :] = (normal_np * 255).astype(np.uint8)
            # save_path = "/data2/liuzhi/3DGS_code/2d-gaussian-splatting/output/gm_Museum_0.05normal_gt_surfel/image_normal" + str(uid) + ".png"
            # plt.imsave(save_path, new_image)

            # fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            # # 显示原始图像
            # axes[0].imshow(image_np)
            # axes[0].set_title('Original Image')
            # # 显示法线图
            # normal_np = (normal_np + 1.0) / 2.0
            # axes[1].imshow(normal_np)
            # axes[1].set_title('Normal Map')
            # plt.show()

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

