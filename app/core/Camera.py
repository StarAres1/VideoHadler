import cv2
from datetime import datetime

class Camera:
    def __init__(self, index, name):
        self.cap = None
        self.index = index
        self.name = name
        self.flag_record = False
        self.fourcc = None
        self.width = None
        self.height = None
        self.output = None
        self.fps = None

        self.connect()

    def connect(self):
        self.cap = cv2.VideoCapture(self.index)
        return self.cap.isOpened()


    def disconnect(self):
        if self.flag_record:
            self.stop_record()

        self.cap.release()


    def capture_video(self):

        if not self.get_property():
            return False

        # Следующая строка временно!!!! До момента создания GUI
        self.start_record()
        try:
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    continue

                if self.flag_record:
                    self.write_video(frame)

                cv2.imshow(self.name, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(e)
        finally:
            self.disconnect()

    def start_record(self):
        self.flag_record = True

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID') # для формата avi
        filename = f'video_output/{self.name}_{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.avi'
        self.output = cv2.VideoWriter(
            filename.replace(" ", "_"),
            self.fourcc,
            self.fps,
            (self.width, self.height))

    def stop_record(self):
        self.flag_record = False
        self.fourcc = None

        self.output.release()
        self.output = None

    def write_video(self, frame):
        if self.output == None:
            return

        self.output.write(frame)

    def get_property(self):
        if not self.cap.isOpened():
            return False

        ret, frame = self.cap.read()

        if not ret:
            return False

        self.height, self.width, channel = frame.shape
        print("channel: ", channel)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print("fps: ", self.fps)

        print("Свойства получены")
        return True