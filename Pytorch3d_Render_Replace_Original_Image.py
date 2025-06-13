import torch
import numpy as np
import os
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import math
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    AmbientLights,
    TexturesVertex
)
import torch.nn.functional as F

def convert_nuscenes_to_pytorch3d(cam_in_ann, device,fov,aspect_ratio):
    """将nuScenes相机位姿转换为Pytorch3D相机参数，使用传入的转换矩阵"""
    # 提取旋转和平移,输入的其实是相机的姿态矩阵
    R_nu = cam_in_ann[:3, :3]  
    T_nu = cam_in_ann[:3, 3]   

    #nuscenes是z向上，x向右，y向下，pytorch3d是x向左，y向上，因此使用绕z轴旋转180度矩阵
    R_old = np.array([
    [-1,  0,  0],  
    [0, -1,  0],  
    [0,  0,  1]   
    ])
    R_pytorch3d = R_nu
    T_pytorch3d = T_nu
    # print("R_pytorch3d: ", R_pytorch3d)
    eye = torch.tensor(T_pytorch3d, dtype=torch.float32).unsqueeze(0).to(device)
    # print("eye",eye)
    print("R_nu:",R_nu)
    print("T_nu",T_nu)
    
    #左乘矩阵和右乘矩阵的区别是左乘是世界坐标系下旋转，右乘是相对于当前坐标系旋转
    
    Rotation =torch.tensor(R_nu@R_old, dtype=torch.float32).unsqueeze(0).to(device)
    #T=-RC
    #这个转置相当于把姿态旋转矩阵Rc变成了外参旋转矩阵的R
    T = -torch.bmm(Rotation.transpose(1, 2), eye.unsqueeze(-1)).squeeze(-1)
    print("T_we", T)
    print("Rotation_lixiang", Rotation)
    
    # 构建Pytorch3D相机（这里使用FoVPerspectiveCameras，可根据需要调整fov等参数）
    # 似乎和pytorch3d对应不上的点就是文档里面说这个R是外参矩阵的R，实际上输入的是姿态矩阵
    cameras = FoVPerspectiveCameras(
        device=device, R=Rotation, T=T, fov=fov,aspect_ratio=aspect_ratio, degrees=True
    )

    return cameras, Rotation, T

def blend_images(original_img, rendered_img, mask_img, alpha=1.0):
    """
    在mask区域用渲染图像替换原图像
    Args:
        original_img: 原始RGB图像 (H, W, 3)
        rendered_img: 渲染图像 (H, W, 3) 
        mask_img: 掩码图像 (H, W, 3) 或 (H, W)
        alpha: 混合透明度
    Returns:
        blended_img: 混合后的图像
    """
    # 确保所有图像都是float类型且在0-1范围内
    original_img = original_img.astype(np.float32) / 255.0 if original_img.max() > 1.0 else original_img.astype(np.float32)
    rendered_img = rendered_img.astype(np.float32) if rendered_img.max() <= 1.0 else rendered_img.astype(np.float32) / 255.0
    
    # 处理mask，确保是0-1范围的float
    if len(mask_img.shape) == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    mask_img = mask_img.astype(np.float32) / 255.0 if mask_img.max() > 1.0 else mask_img.astype(np.float32)
    
    # 将mask扩展到3个通道
    mask_3d = np.stack([mask_img] * 3, axis=-1)
    
    # 在mask区域混合图像
    blended_img = original_img * (1 - mask_3d * alpha) + rendered_img * mask_3d * alpha
    
    # 确保值在0-1范围内
    blended_img = np.clip(blended_img, 0, 1)
    
    return blended_img

objfile='car_pytorch3d_last_E2E_output_forward_-z_up_y/pytorch3d_Etron.obj'
# 修改：添加原图数据集路径
seg_datapath="./test_seg"
rgb_datapath="./test_rgb"

device = 'cuda:0'

# 加载两个数据集
nusc_seg = NuScenes(version='v1.14', dataroot=seg_datapath, verbose=True)
nusc_rgb = NuScenes(version='v1.14', dataroot=rgb_datapath, verbose=True)

print(f"分割数据集中共有 {len(nusc_seg.scene)} 个场景")
print(f"原图数据集中共有 {len(nusc_rgb.scene)} 个场景")

texture_size=6
verts, faces, aux = load_obj(objfile) 
tex_maps = aux.texture_images
image = None
if tex_maps is not None and len(tex_maps) > 0:
    verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
    faces_uvs = faces.textures_idx.to(device)  # (F, 3)
    image = list(tex_maps.values())[0].to(device)[None]

verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
faces_uvs = faces.textures_idx.to(device) 
tex = TexturesUV(
    verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image
)
mesh = Meshes(
    verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=tex
)
R, T = look_at_view_transform(2.7, 0, 180) 
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# 更新纹理（若需要）
tex = TexturesUV(
    verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image 
)
mesh.textures = tex

# 创建基本输出目录
output_base_dir = 'rendered_images_6_13_test_with_rgb'
os.makedirs(output_base_dir, exist_ok=True)

view_angle_dict={
    "CAM_FRONT": 70,
    "CAM_FRONT_RIGHT": 70,
    "CAM_FRONT_LEFT": 70,
    "CAM_BACK_RIGHT": 70,
    "CAM_BACK_LEFT": 70,
    "CAM_BACK": 110,
}

blur = 0
raster_settings = RasterizationSettings(
                image_size=[900, 900], 
                blur_radius=blur,
                faces_per_pixel=1,
                bin_size=0,
            )
lights = AmbientLights(device=device)
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
)
renderer.to(device)
renderer.eval()

# 为所有传感器预先创建目录
sensors = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
for scene_idx in range(len(nusc_seg.scene)):
    scene_dir = f'{output_base_dir}/scene_{scene_idx}'
    os.makedirs(scene_dir, exist_ok=True)
    for sensor in sensors:
        sensor_dir = f'{scene_dir}/{sensor}'
        os.makedirs(sensor_dir, exist_ok=True)

# 遍历每个场景  
for scene_idx, (scene_seg, scene_rgb) in enumerate(zip(nusc_seg.scene, nusc_rgb.scene)):
    scene_dir = f'{output_base_dir}/scene_{scene_idx}'
    print(f"处理场景 {scene_idx} (SEG ID: {scene_seg['token']}, RGB ID: {scene_rgb['token']}): {scene_seg['description']}")
    
    # 获取场景的第一个样本
    first_sample_token_seg = scene_seg['first_sample_token']
    first_sample_token_rgb = scene_rgb['first_sample_token']
    
    current_sample_seg = nusc_seg.get('sample', first_sample_token_seg)
    current_sample_rgb = nusc_rgb.get('sample', first_sample_token_rgb)
    
    # 遍历场景中的所有样本
    sample_idx = 0
    while True:
        print(f"  处理样本 {sample_idx} (SEG ID: {current_sample_seg['token']}, RGB ID: {current_sample_rgb['token']})")
        
        # 检查该样本是否有注释
        if len(current_sample_seg['anns']) == 0:
            print(f"  样本 {sample_idx} 没有注释，跳过")
            
            # 移动到下一个样本
            if 'next' not in current_sample_seg or not current_sample_seg['next']:
                break
            
            current_sample_seg = nusc_seg.get('sample', current_sample_seg['next'])
            current_sample_rgb = nusc_rgb.get('sample', current_sample_rgb['next'])
            sample_idx += 1
            continue
        
        # 用于保存合并视图的列表    
        rendered_images = []
        mask_images = []
        original_images = []
        blended_images = []
        
        for sensor_idx, sensor in enumerate(sensors):
            fov = view_angle_dict[sensor]
            
            # 获取分割数据
            cam_data_seg = nusc_seg.get('sample_data', current_sample_seg['data'][sensor])
            filename_seg = cam_data_seg["filename"]
            filename_seg = filename_seg.replace('\\', '/')
            mask_img_path = os.path.normpath(os.path.join(seg_datapath, filename_seg))
            
            # 获取原图数据
            cam_data_rgb = nusc_rgb.get('sample_data', current_sample_rgb['data'][sensor])
            filename_rgb = cam_data_rgb["filename"]
            filename_rgb = filename_rgb.replace('\\', '/')
            rgb_img_path = os.path.normpath(os.path.join(rgb_datapath, filename_rgb))
            
            # 加载图像
            mask_img_np = cv2.imread(mask_img_path, cv2.IMREAD_UNCHANGED)
            rgb_img_np = cv2.imread(rgb_img_path, cv2.IMREAD_COLOR)
            
            if mask_img_np is None:
                raise ValueError(f"无法加载mask图像文件: {filename_seg}")
            if rgb_img_np is None:
                raise ValueError(f"无法加载RGB图像文件: {filename_rgb}")
            
            print(f"    处理传感器: {sensor}")
            print(f"    mask文件: {filename_seg}")
            print(f"    rgb文件: {filename_rgb}")
            
            # 使用分割数据集的标定信息进行渲染（假设两个数据集的标定信息相同）
            ego_pose = nusc_seg.get('ego_pose', cam_data_seg['ego_pose_token'])
            calibrated_sensor = nusc_seg.get('calibrated_sensor', cam_data_seg['calibrated_sensor_token'])
            vehicle_annotations = []
            for ann_token in current_sample_seg['anns']:
                ann = nusc_seg.get('sample_annotation', ann_token)
                if 'vehicle' in ann['category_name']:
                    vehicle_annotations.append(ann)

            # 打印或检查结果
            print(f"找到 {len(vehicle_annotations)} 个 vehicle 类别的标注")
            sample_annotation=vehicle_annotations[0]
            renderer_vehicle_tranlation = sample_annotation["translation"]
            renderer_vehicle_rotation = sample_annotation["rotation"]

            print(f"ego_pose_translation:{ego_pose['translation']}")
            print(f"calibrated_sensor_translation:{calibrated_sensor['translation']}")
            print(f"sample_annotation:{renderer_vehicle_rotation}")

            global_from_ego = transform_matrix(ego_pose['translation'], Quaternion(ego_pose["rotation"]), inverse=False)
            ego_from_sensor = transform_matrix(calibrated_sensor["translation"], Quaternion(calibrated_sensor["rotation"]), inverse=False)
            
            ego_from_sensor[0:3, 0:3] = ego_from_sensor[0:3, 0:3]
            target_from_global = transform_matrix(renderer_vehicle_tranlation, Quaternion(renderer_vehicle_rotation), inverse=True)
            
            #现在的pose是target_from_sensor
            pose = target_from_global.dot(global_from_ego.dot(ego_from_sensor))
            cameras, Rotation, T = convert_nuscenes_to_pytorch3d(pose, device, fov, 1)
            
            # 渲染
            imgs_pred = renderer(mesh, cameras=cameras)
            imgs_pred = imgs_pred[0, ...]
            imgs_pred = imgs_pred.squeeze(0)
            imgs_pred = imgs_pred / torch.max(imgs_pred)
            
            # 转换为numpy数组
            rendered_img_np = (imgs_pred.detach().cpu().numpy())
            rendered_img_np = rendered_img_np[:,:,:3]
            print("rendered_img_np_shape", rendered_img_np.shape)
            
            # 创建混合图像：在mask区域用渲染结果替换原图
            blended_img_np = blend_images(rgb_img_np, rendered_img_np, mask_img_np)
            
            # 保存到列表中用于合并视图
            rendered_images.append(rendered_img_np.copy())
            mask_images.append(mask_img_np.copy())
            original_images.append(rgb_img_np.copy())
            blended_images.append(blended_img_np.copy())
            
            # 保存每个传感器的单独图像
            sensor_dir = f'{scene_dir}/{sensor}'
            
            # 保存原始RGB图像
            rgb_output_filename = f'{sensor_dir}/original_scene{scene_idx}_sample{sample_idx}.png'
            cv2.imwrite(rgb_output_filename, rgb_img_np)
            print(f'  保存了 {sensor} 的原始RGB图像: {rgb_output_filename}')
            
            # 保存渲染图像
            render_output_filename = f'{sensor_dir}/render_scene{scene_idx}_sample{sample_idx}.png'
            cv2.imwrite(render_output_filename, rendered_img_np*255)
            print(f'  保存了 {sensor} 的渲染图像: {render_output_filename}')
            
            # 保存掩码图像
            mask_output_filename = f'{sensor_dir}/mask_scene{scene_idx}_sample{sample_idx}.png'
            cv2.imwrite(mask_output_filename, mask_img_np)
            print(f'  保存了 {sensor} 的掩码图像: {mask_output_filename}')
            
            # 保存混合图像（主要输出）
            blended_output_filename = f'{sensor_dir}/blended_scene{scene_idx}_sample{sample_idx}.png'
            cv2.imwrite(blended_output_filename, (blended_img_np * 255).astype(np.uint8))
            print(f'  保存了 {sensor} 的混合图像: {blended_output_filename}')
        
        # 如果成功处理了6个视图，保存组合视图
        if len(blended_images) == 6:
            # 混合图像的组合视图
            blended_row1 = np.hstack((blended_images[0], blended_images[1], blended_images[2]))
            blended_row2 = np.hstack((blended_images[3], blended_images[4], blended_images[5]))
            multi_view_blended = np.vstack((blended_row1, blended_row2))
            
            blended_output_filename = f'{scene_dir}/multi_view_blended_scene{scene_idx}_sample{sample_idx}.png'
            cv2.imwrite(blended_output_filename, (multi_view_blended * 255).astype(np.uint8))
            print(f'  保存了多视图混合图像: {blended_output_filename}')
            
            # 原始RGB图像的组合视图
            rgb_row1 = np.hstack((original_images[0], original_images[1], original_images[2]))
            rgb_row2 = np.hstack((original_images[3], original_images[4], original_images[5]))
            multi_view_rgb = np.vstack((rgb_row1, rgb_row2))
            
            rgb_output_filename = f'{scene_dir}/multi_view_original_scene{scene_idx}_sample{sample_idx}.png'
            cv2.imwrite(rgb_output_filename, multi_view_rgb)
            print(f'  保存了多视图原始图像: {rgb_output_filename}')
            
            # 渲染图像的组合视图
            render_row1 = np.hstack((rendered_images[0], rendered_images[1], rendered_images[2]))
            render_row2 = np.hstack((rendered_images[3], rendered_images[4], rendered_images[5]))
            multi_view_render = np.vstack((render_row1, render_row2))
            
            render_output_filename = f'{scene_dir}/multi_view_render_scene{scene_idx}_sample{sample_idx}.png'
            cv2.imwrite(render_output_filename, (multi_view_render * 255).astype(np.uint8))
            print(f'  保存了多视图渲染图像: {render_output_filename}')
            
            # mask图像的组合视图
            mask_row1 = np.hstack((mask_images[0], mask_images[1], mask_images[2]))
            mask_row2 = np.hstack((mask_images[3], mask_images[4], mask_images[5]))
            multi_view_mask = np.vstack((mask_row1, mask_row2))
            
            mask_output_filename = f'{scene_dir}/multi_view_mask_scene{scene_idx}_sample{sample_idx}.png'
            cv2.imwrite(mask_output_filename, multi_view_mask)
            print(f'  保存了多视图mask图像: {mask_output_filename}')
        
        # 移动到下一个样本
        if 'next' not in current_sample_seg or not current_sample_seg['next']:
            break
        if 'next' not in current_sample_rgb or not current_sample_rgb['next']:
            break
            
        current_sample_seg = nusc_seg.get('sample', current_sample_seg['next'])
        current_sample_rgb = nusc_rgb.get('sample', current_sample_rgb['next'])
        sample_idx += 1

print("完成所有场景的渲染")