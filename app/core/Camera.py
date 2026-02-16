import cv2

class Camera:
    def __init__(self, index, name):
        self.cap = None
        self.index = index
        self.name = name

        self.connect()

    def connect(self):
        self.cap = cv2.VideoCapture(self.index)
        return self.cap.isOpened()


    def disconnect(self):
        self.cap.release()


    def capture_video(self):
        if not self.cap.isOpened:
            return
        try:
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    continue

                cv2.imshow(self.name, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(e)
        finally:
            self.disconnect()

        