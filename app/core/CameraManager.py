from pygrabber.dshow_graph import FilterGraph
from app.core.Camera import Camera


class CameraManager:
    def __init__(self, video_frame, ui):
        self.cameras = []
        self.video_frame = video_frame
        self.ui = ui
        self._resolution_cache = {}

    def find_cameras(self, gui_list):
        gui_list.clear()
        self.cameras.clear()

        graph = FilterGraph()
        devices = graph.get_input_devices()

        for i in range(len(devices)):
            self.cameras.append(self.create_camera(i, devices[i]))
            gui_list.addItem(f"{i}: {devices[i]}")

    def current_camera(self, index: int = 0):
        if not self.cameras:
            return None
        if 0 <= index < len(self.cameras):
            return self.cameras[index]
        return self.cameras[0]

    def create_camera(self, index, name):
        camera = Camera(index, name, self.video_frame, self.ui)
        if index in self._resolution_cache:
            camera.supported_resolutions = list(self._resolution_cache[index])
        return camera

    def get_cached_resolutions(self, index: int):
        return list(self._resolution_cache.get(index, []))

    def set_cached_resolutions(self, index: int, resolutions):
        self._resolution_cache[index] = list(resolutions or [])
