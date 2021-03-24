import typing
import random
import numpy as np


class VehicleManager:
    def __init__(self, world):
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicles = []

    def add_vehicle(self, args):
        new_vehicle = Vehicle(
            self.blueprint_library.find("model3"),
            random.choice(self.world.get_map().get_spawn_points()),
            self.world,
            self
        )
        car_model = None
        try:
            self.spawn_vehicle(new_vehicle)
            self.vehicles.append(new_vehicle)
        except:
            pass

class Vehicle:

    def __init__(self, car_model: str, spawn_position: np.ndarray, world, vehicle_manager: VehicleManager):
        self.position = spawn_position
        self.car_model = car_model
        self.world = world
        self.actor = None
        self.sensors = {}
        self.spawn()

    def destroy(self):
        for (_, v) in enumerate(self.sensors):
            v.destroy()
        self.actor.destroy()

    def spawn(self):
        self.actor = self.world.spawn_actor(self.position, self.car_model)

    def add_sensor(self, sensor_id, sensor):
        if self.sensors.keys().__contains__(sensor_id):
            self.sensors[sensor_id].destroy()
        self.sensors[sensor_id] = sensor

    def act(self):
        pass

class DataGatheringBot(Vehicle):
    def __init__(self, car_model: str, spawn_position: np.ndarray, world, vehicle_manager: VehicleManager):
        super().__init__(car_model, spawn_position, world, vehicle_manager)

    def act(self):
        # drive around. gather photos periodically. save photos.
        pass

    def take_photograph(self):
        self.sensors["front_camera"].take_photo() #TODO do properly
