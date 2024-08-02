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
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # 视锥体外的或半径为0的高斯体 在当前相机视角下是不可见的，它们将被排除在增稠中的分裂条件中使用的值更新之外。
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }


    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # 渲染的法向量图，并将其从相机坐标系转换到世界坐标系中，transform normal from view space to world space
    render_normal = allmap[2:5] # 3 H W
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)    # H W 3 @ 3 3 = H W 3 ==> 3 H W

    # 获取渲染的深度图：根据场景选择是 中值深度图 or 期望深度图
    # (1) 渲染的中值深度图，get median depth map
    render_depth_median = allmap[5:6]   # 1 H W
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # (2) 渲染的期望深度图，使用不透明度归一化，get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # 最终的渲染深度图 (伪表面深度图 surf_depth)，由根据场景设置的depth_ratio决定：
    # 有界场景(TnT、DTU)，depth_ratio = 1，使用 中值深度
    # 无边界场景(MipNeRF360)，depth_ratio = 0，使用 期望深度，减少伪影
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median    # 1 H W
    
    # 从伪表面深度图计算表面法向量 surf_normal：假设深度点分布于表面，使用它们生成伪表面法向量，用于正则化
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth) # H W 3
    surf_normal = surf_normal.permute(2,0,1)
    # 记得乘以累加权重accum_alpha，因为render_normal未归一化
    surf_normal = surf_normal * (render_alpha).detach() # 3 H W

    # 获取 与光线相交的 2D高斯与2D高斯之间的距离，get depth distortion map
    render_dist = allmap[6:7]


    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,   # 与光线相交的2D高斯与2D高斯之间的距离
            'surf_depth': surf_depth,   # 伪表面深度图，1 H W
            'surf_normal': surf_normal, # 由伪表面深度图计算的世界坐标下的表面法向量，3 H W
    })

    return rets