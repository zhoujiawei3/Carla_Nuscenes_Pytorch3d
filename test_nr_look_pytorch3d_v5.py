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
    # 提取旋转和平移
    R_nu = cam_in_ann[:3, :3]  # nuScenes中的旋转矩阵
    T_nu = cam_in_ann[:3, 3]   # nuScenes中的平移向量

    # 应用变换：转换矩阵将nuScenes坐标系映射到PyTorch3D坐标系
    # print("R_nu: ", R_nu)
    # print("T_nu: ", T_nu)
    
    R_pytorch3d = R_nu
    T_pytorch3d = T_nu
    # print("R_pytorch3d: ", R_pytorch3d)
    eye = torch.tensor(T_pytorch3d, dtype=torch.float32).unsqueeze(0).to(device)
    # print("eye",eye)
    z_axis=F.normalize(torch.tensor(R_nu[0,:3]*[1,-1,1],dtype=torch.float32).unsqueeze(0).to(device), eps=1e-5)
    up=torch.tensor((0.0,0.0,1.0),dtype=torch.float32).unsqueeze(0).to(device)
    x_axis=F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    Rotation = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    T = -torch.bmm(Rotation, eye.unsqueeze(-1)).squeeze(-1)
    print("T_we", T)
    print("Rotation_we", Rotation)
    # eye=torch.tensor([0,5,0],dtype=torch.float32).to(device).unsqueeze(0)
    # Rotation, T = look_at_view_transform(eye=eye)
    # print("T_They", T)
    # print("Rotation_They", Rotation)
    # 构建Pytorch3D相机（这里使用FoVPerspectiveCameras，可根据需要调整fov等参数）
    cameras = FoVPerspectiveCameras(
        device=device, R=Rotation.transpose(1, 2), T=T, fov=fov,aspect_ratio=aspect_ratio, degrees=True
    )

    return cameras, Rotation, T

mapping_matric=np.array([
    #原始数据集生成使用的矩阵
    # [0,0,1],
    # [1,0,0],
    # [0,1,0]
    # 我自己觉得应该是的矩阵,这是上面的逆矩阵
    [0,1,0],
    [0,0,1],
    [1,0,0]
    
])


objfile='car_pytorch3d_last_E2E_output_forward_-z_up_y/pytorch3d_Etron.obj'
datapath="dataset_4_3_rgb_test_withCar_2_scene_rgb_differentfov_rectangle_right"
device = 'cuda:0'
nusc = NuScenes(version='v1.14', dataroot=datapath, verbose=True)
print(f"NuScenes 数据集中共有 {len(nusc.scene)} 个场景")

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
# blur = 0
# raster_settings = RasterizationSettings(
#     image_size=[900, 1600], 
#     blur_radius=blur,
#     faces_per_pixel=1,
#     bin_size=0,
# )
# lights = AmbientLights(device=device)
# renderer = MeshRenderer(
#     rasterizer=MeshRasterizer(
#         cameras=cameras, 
#         raster_settings=raster_settings
#     ),
#     shader=SoftPhongShader(
#         device=device, 
#         cameras=cameras,
#         lights=lights
#     )
# )
# renderer.to(device)
# renderer.eval()

# 更新纹理（若需要）
tex = TexturesUV(
    verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image 
)
mesh.textures = tex


# 为每个场景创建一个目录
os.makedirs('rendered_images', exist_ok=True)

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

# 遍历每个场景
for scene_idx, scene in enumerate(nusc.scene):
    scene_dir = f'rendered_images/scene_{scene_idx}'
    os.makedirs(scene_dir, exist_ok=True)
    
    print(f"处理场景 {scene_idx} (ID: {scene['token']}): {scene['description']}")
    
    # 获取场景的第一个样本
    first_sample_token = scene['first_sample_token']
    current_sample = nusc.get('sample', first_sample_token)
    
    # 遍历场景中的所有样本
    sample_idx = 0
    while True:
        print(f"  处理样本 {sample_idx} (ID: {current_sample['token']})")
        
        # 检查该样本是否有注释
        if len(current_sample['anns']) == 0:
            print(f"  样本 {sample_idx} 没有注释，跳过")
            
            # 移动到下一个样本
            if 'next' not in current_sample or not current_sample['next']:
                break
            current_sample = nusc.get('sample', current_sample['next'])
            sample_idx += 1
            continue
            
        sensors = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        images = []
        
        mask_images = []

        
        for sensor in sensors:
            fov=view_angle_dict[sensor]
            cam_data = nusc.get('sample_data', current_sample['data'][sensor])
            filename = cam_data["filename"]
            
            filename = filename.replace('\\', '/')
            # 添加前缀 'dataset_3_27_rgb_test_withoutCar_test_test_test' 并标准化路径
            mask_img_path = os.path.normpath(os.path.join(datapath, filename))
            
            mask_img_np = cv2.imread(mask_img_path, cv2.IMREAD_UNCHANGED)
            
            if mask_img_np is None:
                raise ValueError(f"无法加载图像文件: {filename}")
            mask_images.append(mask_img_np)
            
            
            print(f"    处理传感器: {sensor}, 文件: {filename}")
            
            ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
            calibrated_sensor = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            vehicle_annotations = []
            for ann_token in current_sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                if 'vehicle' in ann['category_name']:
                    vehicle_annotations.append(ann)

            # 打印或检查结果
            print(f"找到 {len(vehicle_annotations)} 个 vehicle 类别的标注")
            sample_annotation=vehicle_annotations[0]

            print(f"ego_pose_translation:{ego_pose['translation']}")
            print(f"calibrated_sensor_translation:{calibrated_sensor['translation']}")
            print(f"sample_annotation:{sample_annotation['translation']}")
            global_from_ego = transform_matrix(ego_pose['translation'], Quaternion(ego_pose["rotation"]), inverse=False)
            ego_from_sensor = transform_matrix(calibrated_sensor["translation"], Quaternion(calibrated_sensor["rotation"]), inverse=False)
            ego_from_sensor[0:3, 0:3] = mapping_matric@ego_from_sensor[0:3, 0:3]
            target_from_global = transform_matrix(sample_annotation["translation"], Quaternion(sample_annotation["rotation"]), inverse=True)
            
            pose = target_from_global.dot(global_from_ego.dot(ego_from_sensor))
            cameras, Rotation, T = convert_nuscenes_to_pytorch3d(pose, device,fov,1)
            
  
            

            imgs_pred = renderer(mesh, cameras=cameras)
            # print("render_shape",imgs_pred.shape)
            # exit()
            imgs_pred = imgs_pred[0, ...]
            
            imgs_pred = imgs_pred.squeeze(0)
            # imgs_pred = imgs_pred.transpose(2, 0).transpose(1, 2)
            imgs_pred = imgs_pred / torch.max(imgs_pred)
            # 转换为255范围的uint8
            img_np = (imgs_pred.detach().cpu().numpy() * 255) #shape 3,900,900
            img_np[:,:,3]=255.0
            

            # # 在图像上添加传感器名称
            # img_np = cv2.putText(img_np, sensor, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            images.append(img_np)
            
            

        # 如果成功渲染了6个视图
        if len(images) == 6:
            # 获取场景中自车和目标车辆的位置
            ego_pose_token = cam_data['ego_pose_token']
            ego_pose = nusc.get('ego_pose', ego_pose_token)
            target_pose = sample_annotation
            
            # # 绘制BEV视图
            # bev_view = draw_bev_view(ego_pose, target_pose)
            
            # # 添加文本信息
            # info_text = f"Scene: {scene_idx}, Sample: {sample_idx}"
            # bev_view = cv2.putText(bev_view, info_text, (10, bev_view.shape[0] - 10), 
            #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            # 调整两行视角图像的尺寸，保持六个相机视图的排列
            row1 = np.hstack((images[0], images[1], images[2]))
            row2 = np.hstack((images[3], images[4], images[5]))
            
            # # 创建一个与前两行相同宽度的第三行，用于放置BEV视图和其他可能的信息
            # row_width = row1.shape[1]
            # row_height = images[0].shape[0]  # 假设所有图像高度相同
            
            # # 调整BEV视图大小以适应行高
            # bev_view = cv2.resize(bev_view, (row_height, row_height))
            
            # # 创建第三行（白色背景）
            # row3 = np.ones((row_height, row_width, 3), dtype=np.uint8) * 255
            
            # # 将BEV视图放在第三行的中心
            # start_x = (row_width - row_height) // 2
            # row3[:, start_x:start_x + row_height] = bev_view
            
            # 垂直堆叠三行
            multi_view_image = np.vstack((row1, row2))
            
            # 保存拼接后的图像
            output_filename = f'{scene_dir}/multi_view_scene{scene_idx}_sample{sample_idx}.png'
            cv2.imwrite(output_filename, multi_view_image)
            print(f'  保存了多视图图像: {output_filename}')
            
            # 调整两行视角图像的尺寸，保持六个相机视图的排列
            mask_row1 = np.hstack((mask_images[0], mask_images[1], mask_images[2]))
            mask_row2 = np.hstack((mask_images[3], mask_images[4], mask_images[5]))
            
            # 垂直堆叠两行
            multi_view_mask_image = np.vstack((mask_row1, mask_row2))
            
            # 保存拼接后的mask图像
            output_mask_filename = f'{scene_dir}/multi_view_mask_scene{scene_idx}_sample{sample_idx}.png'
            cv2.imwrite(output_mask_filename, multi_view_mask_image)
            print(f'  保存了多视图mask图像: {output_mask_filename}')
            
        
            
        # 移动到下一个样本
        if 'next' not in current_sample or not current_sample['next']:
            break
        current_sample = nusc.get('sample', current_sample['next'])
        sample_idx += 1

print("完成所有场景的渲染")