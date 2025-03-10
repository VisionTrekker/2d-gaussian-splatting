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
import math
import numpy as np
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import depth_EdgeAwareLogL1, depth_smooth_loss, depth_align, save_depth_comparsion
from torchmetrics.functional.regression import pearson_corrcoef

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        # (1) image loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        # (2) depth loss: 最小化 与光线相交的2D高斯与2D高斯之间的距离，将2D高斯局部约束在表面上
        rend_dist = render_pkg["rend_dist"]
        rend_depth = render_pkg["surf_depth"]   # 相机坐标系下的渲染depth。1 H W
        if not dataset.load_depth:
            # print("use self depth regularization")
            dist_loss = lambda_dist * (rend_dist).mean()
        else:
            # print("use gt depth regularization")
            gt_depth = viewpoint_cam.depth.cuda()   # 1 H W
            rend_depth_aligned, scale_factor = depth_align(gt_depth, rend_depth)
            # print(f"Estimated scale factor: {scale_factor:.4f}")
            # print(f"GT depth range: {gt_depth.min():.4f}, {gt_depth.max():.4f}, size: {gt_depth.size()}")
            # print(f"Rend depth range: {rend_depth.min():.4f}, {rend_depth.max():.4f}")
            # print(f"Aligned rend depth range: {rend_depth_aligned.min():.4f}, {rend_depth_aligned.max():.4f}, size: {rend_depth_aligned.size()}")

            # if iteration % 300 == 0:
            #     save_depth_comparsion(gt_depth, rend_depth, rend_depth_aligned,
            #                           save_path="/data2/liuzhi/3DGS_code/2d-gaussian-splatting/output/7copybook_densify_noplanar_nofixnormal_0.05gtnormal_gtsurfel_0.1gtdepth/depth_align" + str(iteration) + ".png")

            filter_mask_depth = torch.logical_and(gt_depth>1e-3, gt_depth>1e-3)
            l_depth = depth_EdgeAwareLogL1(rend_depth_aligned, gt_depth, gt_image, filter_mask_depth)
            dist_loss = lambda_dist * l_depth

            # l_depth_smooth = depth_smooth_loss(rend_depth_aligned, filter_mask_depth)
            # dist_loss = lambda_dist * (l_depth + 0.5 * l_depth_smooth)
            # print(f"l_depth: {l_depth:.{4}f}, l_depth_smooth: {l_depth_smooth:.{4}f}, dist_loss: {dist_loss:.{4}f}")

            # 皮尔逊损失
            # rend_depth = rend_depth[0].reshape(-1, 1)
            # gt_depth = gt_depth[0].reshape(-1, 1)
            #
            # valid_gt_index = gt_depth > 0
            # valid_gt_data = gt_depth[valid_gt_index]
            # valid_gt_depth_th = np.percentile(valid_gt_data.cpu().numpy(), 10)
            #
            # valid_index = gt_depth < valid_gt_depth_th
            # gt_depth[valid_index] = 0.0
            # rend_depth[valid_index] = 0.0
            # depth_error = min((1 - pearson_corrcoef(-gt_depth, rend_depth)),
            #                 (1 - pearson_corrcoef(1 / (gt_depth + 200), rend_depth)))
            # dist_loss = lambda_dist * depth_error

        # (3) normal loss
        rend_normal = render_pkg['rend_normal']    # 世界坐标系下的渲染normal
        rend_normal_view = render_pkg['rend_normal_view']   # 相机坐标系下的渲染normal
        surf_normal = render_pkg['surf_normal']             # 世界坐标系下从渲染深度图计算的normal
        surf_normal_view = (surf_normal.permute(1, 2, 0) @ viewpoint_cam.world_view_transform[:3, :3]).permute(2, 0, 1)

        if not dataset.load_normal:
            # print("use self normal regularization")
            # 世界坐标系下 渲染的normal 与 从伪表面深度图计算的normal 的Loss
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
        else:
            # print("use gt normal regularization")
            gt_normal_view = viewpoint_cam.normal.cuda()
            gt_normal = (gt_normal_view.permute(1, 2, 0) @ (viewpoint_cam.world_view_transform[:3, :3].T)).permute(2, 0, 1)

            # gt normal 与 渲染的normal 的Loss
            normal_error = (1 - (gt_normal * rend_normal).sum(dim=0))[None]
            # GaussianPro中还计算了l1_normal：torch.abs(rendered_normal - normal_gt).sum(dim=0)[filter_mask].mean()
            normal_loss = lambda_normal * (normal_error).mean()

            # gt normal 与 从伪表面深度图计算的normal 的Loss
            normal_error = (1 - (gt_normal * surf_normal).sum(dim=0))[None]
            normal_loss += lambda_normal * (normal_error).mean()

        # gt_normal = viewpoint_cam.normal.cuda() # 从深度图计算的法向量是世界坐标系下的
        # # gt normal 与 渲染的normal 的Loss
        # normal_error = (1 - (gt_normal * rend_normal).sum(dim=0))[None]
        # # GaussianPro中还计算了l1_normal：torch.abs(rendered_normal - normal_gt).sum(dim=0)[filter_mask].mean()
        # normal_loss = lambda_normal * (normal_error).mean()
        # # gt normal 与 从伪表面深度图计算的normal 的Loss
        # normal_error = (1 - (gt_normal * surf_normal).sum(dim=0))[None]
        # normal_loss += lambda_normal * (normal_error).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            # 法向量约束
            # 方法1：将ratation的梯度置为0，真正起作用，rotation的值不变，且增稠的新高斯也有相同的ratation
            # gaussians._rotation.grad = None
            # gaussians._rotation.data.copy_(gaussians.initial_rotation)    # 值设为初始值（未验证）
            # 方法2：创建self._rotation时将requires_grad设为True/False，
            #       使用get_rotation.requires_grad查询的结果一直为False，使用_rotation.requires_grad查询的结果一直为True，但实际不起作用，ratation仍在计算梯度，其值一直在改变
            # 方法3：在优化器中删除_rotation，在不增稠时实际起作用，但是开启增稠因查询不到_ratation的动量而报错

            if iteration % 10 == 0:
                # 平面约束：将高斯强制拉到平面上
                # gaussians.force_points_to_plane()

                # 检查旋转的一致性
                # max_diff = torch.max(torch.sum((gaussians.get_rotation - gaussians.get_rotation[0]) ** 2, dim=1))

                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                    # "rots[-1]": [f"{x:.{4}f}" for x in gaussians.get_rotation[-1].tolist()],
                    # "Max_rotation_diff": f"{max_diff.item()}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")