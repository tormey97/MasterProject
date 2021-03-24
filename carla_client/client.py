from carla_client import sensor, vehicles
import carla # TODO why does it not resolve?
import argparse

class Client:

    def __init__(self):
        self.vehicle_manager = vehicles.VehicleManager()
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(15.0)
        self.world = self.client.get_world()
        pass

    def run(self):

        '''
        Starts the Carla client. Initializes the connection to the carla server.
        :return:
        '''

