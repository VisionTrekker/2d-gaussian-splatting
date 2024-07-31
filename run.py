import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# depth_ratio：有界场景(TnT、DTU)为1，伪表面深度使用中值深度；无边界场景(MipNeRF360)为0，使用期望深度，减少伪影
# lambda_dist：MipNeRF360为0；TnT环绕场景为100，大场景为10；DTU为1000

# training
print("--------------------------------------------------------------")
cmd = f"python train.py \
        -s ../../Dataset/3DGS_Dataset/input/gm_Museum \
        -m output/gm_Museum_10depth_0.05gtnormal_dr1 \
        --port 6012 \
        --load_normal \
        --lambda_dist 10.0 \
        --lambda_normal 0.05 \
        --depth_ratio 1.0 \
        --iterations 30_000 \
        --densify_from_iter 500 \
        --densify_until_iter 15_000 \
        --densification_interval 100 \
        --test_iterations 7000 15_000 30_000 \
        --save_iterations 30_000"
# print(cmd)
# os.system(cmd)

# render + mesh extraction (bounded volume)
print("--------------------------------------------------------------")
cmd = f"python render.py \
        -s ../../Dataset/3DGS_Dataset/input/gm_Museum \
        -m output/gm_Museum_10depth_0.05gtnormal_dr1 \
        --unbounded \
        --mesh_res 1024"
#cmd = f"python render.py -s ../../Dataset/3DGS_Dataset/input/VID_gaoshipai1 -m output/gaoshipai --mesh_res 1024"
print(cmd)
os.system(cmd)


# evaluation (DTU Dataset)
print("--------------------------------------------------------------")
cmd = f"python scripts/dtu_eval.py -s ../../Dataset/3DGS_Dataset/input/VID_gaoshipai1 -m output/gaoshipai"
#print(cmd)
# os.system(cmd)
