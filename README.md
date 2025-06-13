# Carla_Nuscenes_Pytorch3d

This project creates driving scenes in **CARLA** where a single ego vehicle is surrounded by **exactly one other vehicle**, simulating simple interaction scenarios. The scenes are exported in the **nuScenes** data format, and the surrounding vehicle is replaced with high-fidelity **PyTorch3D** renderings. The final output supports multi-view visualization of realistic driving scenarios with enhanced vehicle appearance and rendering consistency.

<p align="center">
  <img src="https://raw.githubusercontent.com/zhoujiawei3/Carla_Nuscenes_Pytorch3d/main/upload_rendered_images_6_13_test_with_rgb/scene_0/multi_view_original_scene0_sample0.png" width="45%"/>
  <img src="https://raw.githubusercontent.com/zhoujiawei3/Carla_Nuscenes_Pytorch3d/main/upload_rendered_images_6_13_test_with_rgb/scene_0/multi_view_blended_scene0_sample0.png" width="45%"/>
</p>

---

## Project Structure
```bash
Carla_Nuscenes_Pytorch3d/
├── carla_nuscenes/ # Code to generate nuScenes-style dataset from CARLA
│ ├── generate.py
│ ├── config/
│ ├── utils/
│ └── ... # Other utility and config files
├── Pytorch3d_Render_Replace_Original_Image.py # Render using PyTorch3D and replace vehicles in original images
├── Pytorch3d_Render_Replace_masked_Image.py # Render using PyTorch3D and replace vehicles in masked regions
├── test_rgb/ # Generated nuScenes-style RGB images
├── test_seg/ # Corresponding segmentation masks (aligned with RGB keyframes)
```


---

## Dataset Generation

Generate RGB and segmentation-style data from CARLA in the nuScenes structure:

```bash
cd carla_nuscenes

# Generate segmentation (mask) images
python generate.py --mode segmentation --image-size 900 900 --count 2 --random-seed 0 --root ./test_seg

# Generate RGB images
python generate.py --mode rgb --image-size 900 900 --count 2 --random-seed 0 --root ./test_rgb
```

## Requirements:
- **PyTorch3D**
- **nuscenes-devkit**
Install with:
```bash
pip install pytorch3d
pip install nuscenes-devkit
```


## Run Rendering Replacement
```bash
python Pytorch3d_Render_Replace_Original_Image.py
```

or

```bash
python Pytorch3d_Render_Replace_masked_Image.py
```

## Citation

If you find this repository helpful, please consider citing our paper:

```bibtex
@misc{zhou2025carlanuscenespytorch3d,
  author       = {Zhou, Jiawei},
  title        = {Carla_Nuscenes_Pytorch3d: High-Fidelity Vehicle Rerendering in CARLA's multiview output Using PyTorch3D},
  year         = {2025},
  howpublished = {\url{https://github.com/zhoujiawei3/Carla_Nuscenes_Pytorch3d}},
}
```

## Acknowledgements

We would like to thank the developers of [carla_nuscenes](https://github.com/cf206cd/carla_nuscenes) upon which our work is built.
