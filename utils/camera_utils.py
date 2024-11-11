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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import sys
from PIL import Image

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    image = Image.open(cam_info.image_path)
    orig_w, orig_h = image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if len(image.split()) > 3:
        import torch
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb

    depth = None
    normal = None
    if cam_info.depth_path is not None:
        import cv2
        import torch
        # 读取png文件
        # depth = np.array(Image.open(depth_path), dtype=np.float32)  # H W 3
        # resized_depth = cv2.resize(cam_info.depth, resolution, interpolation=cv2.INTER_NEAREST) # H W 3
        # resized_depth = resized_depth[:, :, 0]
        # resized_depth = resized_depth.astype(np.float32) / 255.0
        # resized_depth = torch.from_numpy(resized_depth).unsqueeze(0)    # 1 H W
        # 读取npy文件
        depth = np.load(cam_info.depth_path).astype(np.float32)  # H W
        resized_depth = cv2.resize(depth, resolution, interpolation=cv2.INTER_NEAREST)  # H W
        resized_depth = torch.from_numpy(resized_depth).unsqueeze(0)  # 1 H W
    else:
        resized_depth = None

    if cam_info.normal_path:
        normal = np.load(cam_info.normal_path).astype(np.float32)  # H W 3
        import cv2
        import torch
        resized_normal = cv2.resize(normal, resolution, interpolation=cv2.INTER_NEAREST)
        resized_normal = torch.from_numpy(resized_normal).permute((2, 0, 1))  # 3 H W
    else:
        resized_normal = None

    image.close()
    image = None
    depth = None
    normal = None
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  image_path=cam_info.image_path, depth=resized_depth, normal=resized_normal)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        sys.stdout.write('\r')
        sys.stdout.write("\tReading camera {}/{}".format(id + 1, len(cam_infos)))
        sys.stdout.flush()

        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry