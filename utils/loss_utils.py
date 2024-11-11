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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def LogL1(pred_depth, gt_depth):
    return torch.log(1 + torch.abs(pred_depth - gt_depth))

def depth_EdgeAwareLogL1(pred_depth, gt_depth, gt_image, valid_mask):
    logl1 = LogL1(pred_depth, gt_depth) # 1 H W

    gt_image = gt_image.permute(1, 2, 0)    # H W 1
    grad_img_x = torch.mean(torch.abs(gt_image[:-1, :, :] - gt_image[1:, :, :]), -1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(gt_image[:, :-1, :] - gt_image[:, 1:, :]), -1, keepdim=True)
    lambda_x = torch.exp(-grad_img_x)   # H-1  W   1
    lambda_y = torch.exp(-grad_img_y)   #  H  W-1  1

    logl1_x = logl1[:, :-1, :].squeeze(0)   # H-1  W
    logl1_y = logl1[:, :, :-1].squeeze(0)   #  H  W-1
    loss_x = lambda_x * logl1_x.unsqueeze(-1)
    loss_y = lambda_y * logl1_y.unsqueeze(-1)

    if valid_mask is not None:
        assert valid_mask.shape == pred_depth.shape
        valid_mask = valid_mask.permute(1, 2, 0)
        loss_x = loss_x[valid_mask[:-1, :, :]].mean()
        loss_y = loss_y[valid_mask[:, :-1, :]].mean()
    return loss_x + loss_y


def depth_smooth_loss(depth_map, filter_mask):
    if depth_map.dim() == 3:
        depth_map = depth_map.squeeze(0)
    if filter_mask.dim() == 3:
        filter_mask = filter_mask.squeeze(0)

    # 计算深度图中相邻像素之间的差异
    diff_x = torch.abs(depth_map[:-1, :] - depth_map[1:, :])
    diff_y = torch.abs(depth_map[:, :-1] - depth_map[:, 1:])

    # 过滤掉异常值
    filter_mask_x = filter_mask[:-1, :] & filter_mask[1:, :]
    filter_mask_y = filter_mask[:, :-1] & filter_mask[:, 1:]

    # 计算总的平滑损失
    smooth_loss_x = diff_x[filter_mask_x].mean()
    smooth_loss_y = diff_y[filter_mask_y].mean()
    smooth_loss = smooth_loss_x + smooth_loss_y

    return smooth_loss

def normal_smooth_loss(normal_map,  filter_mask):
    # 将法线图从 (C, H, W) 变为 (H, W, C)
    normal_map = normal_map.permute(1, 2, 0)

    # 计算法线图中相邻像素之间的差异
    diff_x = torch.abs(normal_map[:-1, :, :] - normal_map[1:, :, :])
    diff_y = torch.abs(normal_map[:, :-1, :] - normal_map[:, 1:, :])

    # 过滤掉异常值
    filter_mask_x = filter_mask[:-1, :] & filter_mask[1:, :]
    filter_mask_y = filter_mask[:, :-1] & filter_mask[:, 1:]
    # 计算总的平滑损失
    smooth_loss_x = diff_x[filter_mask_x].mean()
    smooth_loss_y = diff_y[filter_mask_y].mean()
    smooth_loss = smooth_loss_x + smooth_loss_y

    return smooth_loss

def depth_align(gt_depth, pred_depth, valid_threshold=1e-3, outlier_percentile=95):
    """
    返回与gt深度图对齐后的 渲染深度图，以及缩放比率
        valid_threshold: 有效深度值的阈值
        outlier_percentile: 用于去除异常值的百分位数
    """
    if gt_depth.dim() == 3:
        gt_depth = gt_depth.squeeze(0)
    if pred_depth.dim() == 3:
        pred_depth = pred_depth.squeeze(0)

    # 创建有效深度的掩码
    valid_mask = (gt_depth > valid_threshold) & (pred_depth > valid_threshold)

    gt_valid = gt_depth[valid_mask]
    pred_valid = pred_depth[valid_mask]

    if gt_valid.numel() == 0:
        print("Warning: No valid depths found for alignment.")
        return pred_depth, 1.0

    # 使用中位数比率法计算初始尺度
    initial_scale = torch.median(gt_valid) / torch.median(pred_valid)
    # 应用初始尺度
    scaled_pred = pred_valid * initial_scale

    # 计算深度比率
    ratios = gt_valid / scaled_pred

    # 去除异常值
    # 先计算深度比率中分别在2.5%、97.5%处索引的值，符合要求的有效深度
    lower_bound = torch.quantile(ratios, 0.5 - outlier_percentile / 200)    # torch.quantile(a, q)：计算a数组在分位数q处索引对应的的值，如果q处的索引为小数则根据左右两个数差值得到
    upper_bound = torch.quantile(ratios, 0.5 + outlier_percentile / 200)
    inlier_mask = (ratios >= lower_bound) & (ratios <= upper_bound)

    gt_inliers = gt_valid[inlier_mask]
    pred_inliers = pred_valid[inlier_mask]

    # 使用鲁棒的最小二乘法计算最终尺度
    A = pred_inliers.unsqueeze(1)
    b = gt_inliers.unsqueeze(1)
    try:
        solution = torch.linalg.lstsq(A, b).solution
    except AttributeError:
        # 对于较旧的 PyTorch 版本
        solution, _ = torch.linalg.lstsq(A, b)
    scale = solution[0].item()

    # 与gt depth尺度对齐后的渲染深度图
    aligned_pred_depth = pred_depth * scale # H W

    return aligned_pred_depth.unsqueeze(0), scale


def save_depth_comparsion(gt_depth, pred_depth, aligned_pred_depth, save_path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plt.figure(figsize=(15, 5))

    # 计算全局的深度范围用于归一化
    all_depths = torch.cat([gt_depth.flatten(), pred_depth.flatten(), aligned_pred_depth.flatten()])
    vmin, vmax = all_depths.min().item(), all_depths.max().item()
    norm = Normalize(vmin=vmin, vmax=vmax)

    plt.subplot(131)
    plt.imshow(gt_depth.detach().cpu().numpy().squeeze(0), cmap='viridis', norm=norm)
    plt.title('GT Depth')
    plt.colorbar()

    plt.subplot(132)
    plt.imshow(pred_depth.detach().cpu().numpy().squeeze(0), cmap='viridis', norm=norm)
    plt.title('Original Pred Depth')
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(aligned_pred_depth.detach().cpu().numpy().squeeze(0), cmap='viridis', norm=norm)
    plt.title('Aligned Pred Depth')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    # print(f"Depth comparison image saved to {save_path}")