from .client import Client
from .dataset import Dataset
import traceback
import carla
import time
class Generator:
    def __init__(self,config,random_seed=0):
        self.config = config
        self.collect_client = Client(self.config["client"],random_seed=random_seed)

    def generate_dataset(self,load=False):
        self.dataset = Dataset(**self.config["dataset"],load=load)
        print(self.dataset.data["progress"])
        for sensor in self.config["sensors"]:
            self.dataset.update_sensor(sensor["name"],sensor["modality"])
        for category in self.config["categories"]:
            self.dataset.update_category(category["name"],category["description"],category["index"])
        for attribute in self.config["attributes"]:
            self.dataset.update_attribute(attribute["name"],category["description"])
        for visibility in self.config["visibility"]:
            self.dataset.update_visibility(visibility["description"],visibility["level"])

        for world_config in self.config["worlds"][self.dataset.data["progress"]["current_world_index"]:]:
            try:
                self.collect_client.generate_world(world_config)
                map_token = self.dataset.update_map(world_config["map_name"],world_config["map_category"])
                for capture_config in world_config["captures"][self.dataset.data["progress"]["current_capture_index"]:]:
                    log_token = self.dataset.update_log(map_token,capture_config["date"],capture_config["time"],
                                            capture_config["timezone"],capture_config["capture_vehicle"],capture_config["location"])
                    for scene_config in capture_config["scenes"][self.dataset.data["progress"]["current_scene_index"]:]:
                        for scene_count in range(self.dataset.data["progress"]["current_scene_count"],scene_config["count"]):
                            self.dataset.update_scene_count()
                            self.add_one_scene(log_token,scene_config)
                            self.dataset.save()
                        self.dataset.update_scene_index()
                    self.dataset.update_capture_index()
                self.dataset.update_world_index()
            except:
                traceback.print_exc()
            finally:
                self.collect_client.destroy_world()
                
    def add_one_scene(self,log_token,scene_config):
        try:
            calibrated_sensors_token = {}
            samples_data_token = {}
            instances_token = {}
            samples_annotation_token = {}

            self.collect_client.generate_scene(scene_config)
            scene_token = self.dataset.update_scene(log_token,scene_config["description"])

            for instance in self.collect_client.walkers+self.collect_client.vehicles:
                instance_token = self.dataset.update_instance(*self.collect_client.get_instance(scene_token,instance))
                instances_token[instance.get_actor().id] = instance_token
                samples_annotation_token[instance.get_actor().id] = ""
            
            for sensor in self.collect_client.sensors:
                calibrated_sensor_token = self.dataset.update_calibrated_sensor(scene_token,*self.collect_client.get_calibrated_sensor(sensor))
                calibrated_sensors_token[sensor.name] = calibrated_sensor_token
                samples_data_token[sensor.name] = ""

            sample_token = ""
            for frame_count in range(int(scene_config["collect_time"]/self.collect_client.settings.fixed_delta_seconds)):#collect time 5s，每一个tick是0.01s
                print("frame count:",frame_count)
                self.collect_client.tick()
                time.sleep(0.5) 
                if (frame_count+1)%int(scene_config["keyframe_time"]/self.collect_client.settings.fixed_delta_seconds) == 0 and frame_count!=0:
                    print("current_timestamp:",self.collect_client.get_timestamp())
                    sample_token = self.dataset.update_sample(sample_token,scene_token,*self.collect_client.get_sample())
                    for sensor in self.collect_client.sensors:
                        if sensor.bp_name in ['sensor.camera.rgb','sensor.other.radar','sensor.lidar.ray_cast','sensor.camera.semantic_segmentation']:
                            # data_list = sensor.get_data_list()
                            # if len(data_list) != 6:
                            #     print(len(data_list))
                            #     while 
                            # while len(sensor.get_data_list())!=6: 
                            #     print(len(sensor.get_data_list()))
                            #     time.sleep(0.1) 
                            
                            # if len(sensor.get_data_list()) !=6:
                            #     print(len(sensor.get_data_list()))
                            for idx,sample_data in enumerate(sensor.get_data_list()):
                                if sensor.bp_name == 'sensor.camera.semantic_segmentation':
                                    sample_data[1].convert(carla.ColorConverter.CityScapesPalette)
                                ego_pose_token = self.dataset.update_ego_pose(scene_token,calibrated_sensors_token[sensor.name],*self.collect_client.get_ego_pose(sample_data))
                                is_key_frame = False
                                if idx == len(sensor.get_data_list())-1:
                                    is_key_frame = True
                                    print("key_timestamp:",sample_data[1].timestamp)
                                    print("time_delta:",sample_data[1].timestamp-self.collect_client.get_timestamp())
                                if sensor.bp_name == 'sensor.lidar.ray_cast' and is_key_frame:
                                    for sensor_second in self.collect_client.sensors:
                                        if sensor_second.bp_name == "sensor.lidar.ray_cast_semantic":
                                            sem_sample_data_second = sensor_second.get_data_list()[-1]
                                            self.dataset.update_sem_lidar_data(ego_pose_token,sem_sample_data_second)
                                
                                samples_data_token[sensor.name] = self.dataset.update_sample_data(samples_data_token[sensor.name],calibrated_sensors_token[sensor.name],sample_token,ego_pose_token,is_key_frame,*self.collect_client.get_sample_data(sample_data),*self.collect_client.get_vehicle_transform(sample_data))

                    for instance in self.collect_client.walkers+self.collect_client.vehicles:
                        if self.collect_client.get_visibility(instance) > 0:
                            samples_annotation_token[instance.get_actor().id]  = self.dataset.update_sample_annotation(samples_annotation_token[instance.get_actor().id],sample_token,*self.collect_client.get_sample_annotation(scene_token,instance))
                    for sensor in self.collect_client.sensors:
                        sensor.get_data_list().clear()
        except:
            traceback.print_exc()
        finally:
            # settings=self.collect_client.world.get_settings()
            # settings.synchronous_mode = False
            # settings.fixed_delta_seconds = None
            # self.collect_client.world.apply_settings(settings)
            self.collect_client.destroy_scene()