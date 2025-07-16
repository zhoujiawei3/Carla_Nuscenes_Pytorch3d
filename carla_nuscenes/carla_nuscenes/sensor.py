import numpy as np
import carla
from .actor import Actor
import queue

#specific carla's semantic label can be found in LibCarla/source/carla/rpc

carla_to_nuscenes_map = {
    0: 0,    # None → noise
    1: 24,   # Roads → flat.driveable_surface
    2: 26,   # Sidewalks → flat.sidewalk
    3: 28,   # Buildings → static.manmade
    4: 28,   # Walls → static.manmade
    5: 28,   # Fences → static.manmade
    6: 28,   # Poles → static.manmade
    7: 28,   # TrafficLight → static.manmade
    8: 28,   # TrafficSigns → static.manmade
    9: 30,   # Vegetation → static.vegetation
    10: 27,  # Terrain → flat.terrain
    11: 29,  # Sky → static.other
    12: 2,   # Pedestrians → human.pedestrian.adult
    13: 5,   # Rider → human.pedestrian.personal_mobility
    14: 17,  # Car → vehicle.car
    15: 23,  # Truck → vehicle.truck
    16: 16,  # Bus → vehicle.bus.rigid
    17: 22,  # Train → vehicle.trailer
    18: 21,  # Motorcycle → vehicle.motorcycle
    19: 14,  # Bicycle → vehicle.bicycle
    20: 28,  # Static → static.manmade
    21: 3,   # Dynamic → human.pedestrian.child
    22: 29,  # Other → static.other
    23: 25,  # Water → flat.other
    24: 24,  # RoadLines → flat.driveable_surface
    25: 27,  # Ground → flat.terrain
    26: 28,  # Bridge → static.manmade
    27: 25,  # RailTrack → flat.other
    28: 28,  # GuardRail → static.manmade
}
def parse_image(image):
    array = np.ndarray(
            shape=(image.height, image.width, 4),
            dtype=np.uint8, buffer=image.raw_data,order="C")
    return array

def parse_lidar_data(lidar_data):
    points = []
    current_channel = 0
    end_idx = lidar_data.get_point_count(current_channel)
    for idx,data in enumerate(lidar_data):
        point = [data.point.x,data.point.y,data.point.z,data.intensity,current_channel]
        if idx==end_idx:
            current_channel+=1
            end_idx+=lidar_data.get_point_count(current_channel)
        points.append(point)
    return np.array(points)

def parse_semlidar_data(semlidar_data):
    tags = []
    for idx,data in enumerate(semlidar_data):
        tag = data.object_tag
        tags.append(carla_to_nuscenes_map[tag])
    return np.array(tags, dtype=np.uint8)

def parse_radar_data(radar_data):
    points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4')).copy()
    return points

# def parse_data(data):
#     if isinstance(data,carla.Image):
#         return parse_image(data)
#     elif isinstance(data,carla.RadarMeasurement):
#         return parse_radar_data(data)
#     elif isinstance(data,carla.LidarMeasurement):
#         return parse_lidar_data(data)
#     elif isinstance(data, carla.SemanticLidarMeasurement):
#         return parse_semlidar_data(data)

def get_data_shape(data):
    if isinstance(data,carla.Image):
        return data.height,data.width
    else:
        return 0,0
class Sensor(Actor):
    def __init__(self, name, **args):
        super().__init__(**args)
        self.name = name
        self.data_list = []
        self.data_queue = queue.Queue()
        self.vehicle = None
    def get_data_list(self):
        return self.data_list
    def add_vehicle(self, vehicle):
        self.vehicle=vehicle
    def set_actor(self, id):
        super().set_actor(id)
        self.actor.listen(self.add_data)
    
    def spawn_actor(self):
        super().spawn_actor()
        self.actor.listen(self.add_data)

    def get_last_data(self):
        if self.data_list:
            return self.data_list[-1]
        else:
            return None
            
    def add_data(self,data):
        try:
            if self.vehicle is not None:
                self.data_list.append((self.actor.parent.get_transform(),data,self.vehicle.get_transform()))
            else:
                self.data_list.append((self.actor.parent.get_transform(),data))
        except:
            self.data_list.append((self.actor.parent.get_transform(),data))

    def get_transform(self):
        return self.actor.get_transform()