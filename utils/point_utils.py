import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math

def depths_to_points(view, depthmap):
    """
    从深度图生成世界坐标系下的3D点云
    """
    c2w = (view.world_view_transform.T).inverse()

    W, H = view.image_width, view.image_height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    # 投影变换 C2NDC = C2W @ W2NDC，(4, 4)
    projection_matrix = c2w.T @ view.full_proj_transform
    # 相机坐标系2图像坐标系的旋转矩阵，即 内参矩阵 = 投影变换C2NDC @ 视口变换NDC2pix，(4, 4) @ (4, 3) => (4, 3) => (3, 3)
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T

    # 构建深度图的网格坐标
    # 生成1个二维网格，分别包含x轴和y轴的坐标，(H, W)
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    # H*W个像素点在图像坐标系中的齐次坐标(H*W, 3)：torch.stack 将grid_x、grid_y和全1的张量按最后一个维度拼接，形状为(H, W, 3)，每个位置的值是像素坐标的齐次形式 (x, y, 1)
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)

    # 每个像素点在世界坐标系中的坐标：pix @ pix2C @ C2W，(H*W, 3) @ (3, 3) @ (3, 3) = (H*W, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    # 相机光心在世界坐标系中的位置(射线起点)，(3,)
    rays_o = c2w[:3,3]

    # 3D点云 在世界坐标系中的坐标，(H*W, 1) * (H*W, 3) = (H*W, 3)
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(view, depth):
    """
    从相机坐标系下的伪表面深度图 计算 世界坐标系下的表面法向量 surf_normal（已归一化）
        view: 当前相机
        depth: 当前相机坐标系下的 伪表面深度图，(1, H, W)
    """
    # 从伪表面深度图生成世界坐标系下的3D点云
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3) # (H, W, 3)

    # 假设深度点分布于表面，使用它们生成伪表面法向量
    output = torch.zeros_like(points)
    # 计算世界坐标系下3D点云在x、y方向上的梯度
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)   # (H-2, W-2, 3)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    # 使用叉乘计算法向量，然后归一化
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1) # (H-2, W-2, 3)
    output[1:-1, 1:-1, :] = normal_map  # 中心填充到输出中，边缘值为1，(H, W, 3)
    return output