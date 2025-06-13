import argparse
import os
import yaml
from yamlinclude import YamlIncludeConstructor
from carla_nuscenes.generator import Generator

# 注册 !include 支持
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)

# 解析命令行参数
parser = argparse.ArgumentParser(description="CARLA Dataset Generator")
parser.add_argument("--mode", type=str, required=True, choices=["rgb", "segmentation"])
parser.add_argument("--image-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
parser.add_argument("--count", type=int, default=5, help="Override the number of the genrated scenes")
parser.add_argument("--random-seed", type=int, default=0, help="Random seed for data generation")
parser.add_argument("--root", type=str, help="Override dataset root path") 
args = parser.parse_args()

# 配置路径
config_map = {
    "rgb": "./configs/config_rgb.yaml",
    "segmentation": "./configs/config_segmentation.yaml"
}
config_path = config_map[args.mode]

# 加载 YAML 配置
with open(config_path, 'r') as f:
    config = yaml.load(f.read(), Loader=yaml.FullLoader)

# 修改 image_size
if args.image_size:
    for world in config.get("worlds", []):
        for capture in world.get("captures", []):
            for scene in capture.get("scenes", []):
                width, height = args.image_size
                calibrated = scene["calibrated_sensors"]
                for sensor in calibrated.get("sensors", []):
                    if sensor.get("bp_name") == "sensor.camera.rgb":
                        options = sensor.setdefault("options", {})
                        options["image_size_x"] = str(width)
                        options["image_size_y"] = str(height)

                    elif sensor.get("bp_name") == "sensor.camera.semantic_segmentation":
                        options = sensor.setdefault("options", {})
                        options["image_size_x"] = str(width)
                        options["image_size_y"] = str(height)

# 修改 scenes 中的 count 字段
if args.count is not None:
    for world in config.get("worlds", []):
        for capture in world.get("captures", []):
            for scene in capture.get("scenes", []):
                scene["count"] = args.count

# 如果传入了 root 参数，覆盖 config 中的 dataset.root
if args.root:
    config["dataset"]["root"] = args.root 

# 创建 Generator 实例，传入 random_seed
runner = Generator(config, random_seed=args.random_seed)

# 执行生成
if os.path.exists(config["dataset"]["root"]):
    runner.generate_dataset(True)
else:
    runner.generate_dataset(False)
