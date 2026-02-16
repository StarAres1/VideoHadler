import cv2
from pygrabber.dshow_graph import FilterGraph
from Camera import Camera
class CameraManager:
    def __init__(self):
        self.connected_cameras = []

    def find_cameras(self):
        graph = FilterGraph()
        devices = graph.get_input_devices()

        for i in range(len(devices)):
            info = {
                'index': i,
                'name': devices[i]
            }
            self.connected_cameras.append(info)

    def create_camera(self, index, name):
        return Camera(index, name)