import cv2
from datetime import datetime
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class Camera:
    def __init__(self, index, name):
        self.cap = None
        self.index = index
        self.name = name
        self.flag_record = False
        self.flag_capture = False
        self.fourcc = None
        self.width = None
        self.height = None
        self.channel = None
        self.output = None
        self.fps = None

    def connect(self):
        self.cap = cv2.VideoCapture(self.index)
        return self.cap.isOpened()


    def disconnect(self):
        if self.flag_record:
            self.stop_record()

        self.cap.release()


    def capture_video(self, video_frame):

        self.flag_capture = True

        if not self.get_property():
            return False

        # Следующая строка временно!!!! До момента создания GUI
        self.start_record()
        try:
            while self.flag_capture:
                ret, bgr_frame = self.cap.read()

                if not ret:
                    continue

                frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                bytes_per_line = self.channel * self.width

                q_image = QImage(frame.data, self.width, self.height, bytes_per_line, QImage.Format_RGB888)

                pixmap = QPixmap.fromImage(q_image)
                video_frame.setPixmap(pixmap)

                if self.flag_record:
                    self.write_video(frame)

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

        self.height, self.width, self.channel = frame.shape

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        return True

    def stop_capture(self):
        self.flag_capture = False