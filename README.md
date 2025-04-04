# Carla_Nuscenes_Pytorch3d

#将carla_nuscenes库生成的nuscenes数据集的多视角渲染图像复现出来

## 代码构成
carla_nuscenes.py：生成nuscenes数据集（目前config设置是正方形渲染）
test_nr_look_pytorch3d_v5.py Pytorch3D渲染代码（目前只支持正方形渲染）

## Requirements:
pytorch3d

## Run:
```bash
python test_nr_look_pytorch3d_v5.py
```