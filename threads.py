from queue import Queue
from threading import Thread
import cv2
import numpy as np
import time


class MyQueue:
    def __init__(self, maxsize, name=None):
        self.queue = Queue(maxsize=maxsize)
        self.maxsize = maxsize
        self.name = name
        self.total, self.count, self.put_total = -1, -1, 0

    def put(self, item):
        if self.maxsize > 0 and self.qsize() >= self.maxsize:
            self.queue.get()
            self.task_done()
        self.queue.put(item)
        self.put_total += 1

    def get(self):
        if self.total == -1:
            self.total, self.count, self.start_time = 0, 0, time.time()
        self.total, self.count = self.total+1, self.count+1
        if self.count == 100:
            print(self.name, "fps", self.count / (time.time() -
                                                  self.start_time), "total", self.total, "put", self.put_total)
            self.count, self.start_time = 0, time.time()
        try:
            item = self.queue.get()
        except:
            item = None
        return item

    def join(self):
        self.queue.join()

    def task_done(self):
        self.queue.task_done()

    def qsize(self):
        return self.queue.qsize()


class Worker(Thread):
    def __init__(self, iqueue=None, oqueue=None, processor=None):
        super(Worker, self).__init__()
        self.iqueue = iqueue
        self.oqueue = oqueue
        self.processor = processor
        self.stop = False
        self.setDaemon(True)

    def run(self):
        while True:
            item = self.iqueue.get() if self.iqueue is not None else -1
            if item is not None:
                item = self.processor.runOn(item)
                if item is not None and self.oqueue is not None:
                    self.oqueue.put(item)

            self.iqueue.task_done() if self.iqueue is not None else None
            if self.processor.isStop():
                self.stop = True
                print(self.processor.name, "stopped")
                break

    def isStop(self):
        return self.stop


class VideoCaptureProcessor:
    def __init__(self, path="/home/tqlong/Downloads/video.mp4"):
        self.path = path
        self.cap = cv2.VideoCapture(self.path)
        self.count = 0
        self.stop = False
        self.name = "VideoCaptureProcessor"

    def runOn(self, item):
        ret, frame = self.cap.read()
        self.count += 1
        print("capture", self.count)
        if not ret or self.count > 400:
            self.stop = True
        return dict(frame=frame, idx=self.count) if ret else None

    def isStop(self):
        return self.stop


class FaceLocator:
    def __init__(self, id):
        import face_recognition as fr
        self.fr = fr
        self.name = "FaceLocator"+str(id)

        self.fr.face_locations(
            np.zeros((100, 100, 3), dtype=np.uint8))

    def runOn(self, item):
        print(self.name)
        frame, idx = item['frame'], item['idx']
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = self.fr.face_locations(rgb_small_frame)
        item['face_locations'] = [(top*4, right*4, bottom*4, left*4)
                                  for top, right, bottom, left in face_locations]
        return item

    def isStop(self):
        return False


class VideoRenderProcessor:
    def __init__(self):
        self.stop = False
        self.name = "VideoRenderProcessor"

    def runOn(self, item):
        # print("render")
        frame, idx = item['frame'], item['idx']

        # print("frame", idx, frame.shape)

        output = frame.copy()
        face_locations = item['face_locations']
        for top, right, bottom, left in face_locations:
            cv2.rectangle(output, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.imshow("frame", output)
        cv2.waitKey(1)

    def isStop(self):
        return self.stop


NUM_FACE_LOCATOR = 24

frameq = MyQueue(maxsize=100, name="frame queue")
faceq = MyQueue(maxsize=100, name="face queue")

video_capture = Worker(oqueue=frameq, processor=VideoCaptureProcessor())
face_locators = [Worker(iqueue=frameq, oqueue=faceq, processor=FaceLocator(i))
                 for i in range(NUM_FACE_LOCATOR)]
video_render = Worker(iqueue=faceq, oqueue=None,
                      processor=VideoRenderProcessor())

threads = [video_capture, video_render] + face_locators
queues = [frameq, faceq]

[t.start() for t in threads]

video_capture.join()
[q.join() for q in queues]
print("All done")
