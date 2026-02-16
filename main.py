from CameraManager import CameraManager
import threading
import time

number = 0

if __name__ == "__main__":
    cameraManager = CameraManager()
    cameraManager.find_cameras()

    connected_camera = cameraManager.connected_cameras
    print(connected_camera)

    camera1 = cameraManager.create_camera(connected_camera[number]["index"], connected_camera[number]["name"])
    camera2 = cameraManager.create_camera(connected_camera[1]["index"], connected_camera[1]["name"])

    t1 = threading.Thread(target=camera1.capture_video)
    t2 = threading.Thread(target=camera2.capture_video)

    t1.start()
    t2.start()

    t1.join()
    t2.join()