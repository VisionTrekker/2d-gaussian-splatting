# import os
#
# GPU_ID = 2
#
# # depth_ratio：有界场景(TnT、DTU)为1，伪表面深度图使用中值深度；无边界或大场景(MipNeRF360)为0，使用平均深度，减少伪影
# # lambda_dist：MipNeRF360为0；TnT环绕场景为100，大场景为10；DTU为1000
# # 10 > 100 > 1000不能用，depth在自拍数据集的结果挺好，但对高斯集中在表面没有影响
# # Dataset/3DGS_Dataset/input/
# # remote_data/dataset_reality/
# # scenes={"meetingroom_06":"cuda", "ling_src1":"cpu"}
# scenes={"berlin_425": "cpu"}
#
# for idx, scene in enumerate(scenes.items()):
#     ############ 训练 ############
#     print("--------------------------------------------------------------")
#     # 可选 --load_depth --load_normal
#     # 使用曼哈顿对齐
#     # --manhattan --platform "cc" \
#     # --pos "-2.999485015869 13.355422973633 24.822856903076" \
#     # --rot "0.985306143761 -0.060848191381 0.159591123462 0.138802871108 -0.259215950966 -0.955793321133 0.099526859820 0.963900744915 -0.246961146593" \
#     cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
#             python train.py \
#             -s ../../remote_data/dataset_opensource/{scene[0]} \
#             -r -1 \
#             --data_device "{scene[1]}" \
#             -m output/{scene[0]}_0.05gtnormal_0.1gtdepth \
#             --port 6015 \
#             --load_normal \
#             --lambda_normal 0.05 \
#             --load_depth \
#             --lambda_dist 0.1 \
#             --depth_ratio 1.0 \
#             --iterations 30_000 \
#             --densify_from_iter 500 \
#             --densify_until_iter 15_000 \
#             --densification_interval 100 \
#             --test_iterations 7_000 15_000 30_000 \
#             --save_iterations 15_000 30_000'
#     print(cmd)
#     os.system(cmd)
#
#     ############ 渲染+提取mesh ############
#     # 可选：--unbounded --skip_train --skip_test --skip_mesh
#     print("--------------------------------------------------------------")
#     cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
#             python render.py \
#             -s ../../remote_data/dataset_opensource/{scene[0]} \
#             -m output/{scene[0]}_0.05gtnormal_0.1gtdepth \
#             --mesh_res 1024 \
#             --skip_mesh \
#             --skip_test'
#     print(cmd)
#     os.system(cmd)
#
#
# ############ 中心裁剪小物体 ############
# print("--------------------------------------------------------------")
# cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
#         python center_obj.py \
#         -s ../../Dataset/3DGS_Dataset/input/Minions_cxk \
#         -m output/Minions_cxk_0.05gtnormal_gtsurfel_0.1gtdepth \
#         --little_object True \
#         --iteration 30000'
# print(cmd)
# os.system(cmd)



############ 评测 (DTU Dataset) ############
# print("--------------------------------------------------------------")
# cmd = f'python scripts/dtu_eval.py -s ../../Dataset/3DGS_Dataset/input/VID_gaoshipai1 -m output/gaoshipai'
#print(cmd)
# os.system(cmd)



import os

GPU_ID = 3

# depth_ratio：有界场景(TnT、DTU)为1，伪表面深度图使用中值深度；无边界或大场景(MipNeRF360)为0，使用平均深度，减少伪影
# lambda_dist：MipNeRF360为0；TnT环绕场景为100，大场景为10；DTU为1000
# 10 > 100 > 1000不能用，depth在自拍数据集的结果挺好，但对高斯集中在表面没有影响
# Dataset/3DGS_Dataset/input/
# remote_data/dataset_reality/
# scenes={"meetingroom_06":"cuda", "ling_src1":"cpu"}
# scenes={"gm_Museum": "cuda"}

############ 训练 ############
print("--------------------------------------------------------------")
# 可选 --load_depth --load_normal
# 使用曼哈顿对齐
# --manhattan --platform "cc" \
# --pos "-2.999485015869 13.355422973633 24.822856903076" \
# --rot "0.985306143761 -0.060848191381 0.159591123462 0.138802871108 -0.259215950966 -0.955793321133 0.099526859820 0.963900744915 -0.246961146593" \


cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
        python train.py \
        -s ../../remote_data/dataset_reality/anji_qiyu \
        --data_device "cpu" \
        -m output/anji_qiyu_depth \
        --port 6015 \
        --load_depth \
        --lambda_dist 0.1 \
        --depth_ratio 0.0 \
        --iterations 30_000 \
        --densify_from_iter 500 \
        --densify_until_iter 15000 \
        --densification_interval 100 \
        --opacity_reset_interval 30000 \
        --test_iterations 7_000 15_000 30_000 \
        --save_iterations 15_000 30_000'
print(cmd)
os.system(cmd)

############ 渲染+提取mesh ############
# 可选：--unbounded --skip_train --skip_test --skip_mesh
# print("--------------------------------------------------------------")
# cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
#         python render.py \
#         -s ../../remote_data/dataset_reality/siyue/2floor_multiview_20240905_standard \
#         -m ../../remote_data/result/liuzhi/2DGS/2floor_multiview_20240905_standard \
#         --mesh_res 1024 \
#         --render_path \
#         --skip_mesh \
#         --skip_test'
# print(cmd)
# os.system(cmd)


############ 中心裁剪小物体 ############
print("--------------------------------------------------------------")
cmd = f'CUDA_VISIBLE_DEVICES={GPU_ID} \
        python center_obj.py \
        -s ../../Dataset/3DGS_Dataset/input/Minions_cxk \
        -m output/Minions_cxk_0.05gtnormal_gtsurfel_0.1gtdepth \
        --little_object True \
        --iteration 30000'
