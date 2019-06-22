from multiprocessing import Process, Queue, Lock
from threading import Thread
import cv2
import numpy as np
import time
import face_recognition as fr
from queue import PriorityQueue, Full, Empty


class MyQueue():
    def __init__(self, name, maxsize=30):
        self.queue = Queue()
        self.name = name
        self.maxsize = maxsize
        self.lock = Lock()
        self.put_lock = Lock()

    def qput(self, item):
        # print("queue", self.name, self.queue.qsize())
        self.put_lock.acquire()
        try:
            while self.queue.qsize() >= self.maxsize:
                item = self.queue.get()
            self.queue.put(item)
        except:
            pass
        self.put_lock.release()

    def qget(self, timeout=None):
        # print("queue", self.name, self.queue.qsize())
        self.lock.acquire()
        try:
            item = self.queue.get(timeout=timeout)
            while self.queue.qsize() > self.maxsize:
                item = self.queue.get()
        except:
            item = None
        try:
            self.lock.release()
        except:
            pass
        return item


class Worker(Process):
    def __init__(self, iqueue=None, oqueues=None, processor=None):
        super(Worker, self).__init__()
        self.iqueue = iqueue
        self.oqueues = oqueues
        self.processor = processor
        self.deamon = True

    def run(self):
        self.processor.init()
        print(self.processor.name, "init")
        oqueue_idx = 0
        while True:
            item = self.iqueue.qget(2) if self.iqueue is not None else -1
            # if self.iqueue is not None:
            #     print(self.iqueue.name, self.iqueue.queue.qsize())
            if item is not None:
                item = self.processor.runOn(item)
                if self.oqueues is not None:
                    self.oqueues[oqueue_idx].qput(item)
                    oqueue_idx = (oqueue_idx+1) % len(self.oqueues)
                if self.processor.stop:
                    break
            else:  # No more item after 5 seconds
                break
        print(self.processor.name, "done")


class VideoCaptureProcessor:
    def __init__(self, path):
        self.path = path
        self.name = "VideoCaptureProcessor"

    def init(self):
        self.cap = cv2.VideoCapture(self.path)
        self.count = 0
        self.stop = False

    def runOn(self, item):
        ret, frame = self.cap.read()
        time.sleep(0.01)
        if ret:
            self.count += 1
            item = dict(frame=frame, idx=self.count)
            # print(frame.shape, self.count)
            if self.count > 8000:
                self.stop = True
        else:
            self.stop = True
            item = None
        return item


class VideoRenderProcessor:
    def __init__(self):
        self.name = "VideoRenderProcessor"

    def init(self):
        self.stop = False
        self.storage = PriorityQueue()
        self.render_thread = Thread(target=self.run)
        self.render_thread.setDaemon(True)
        self.render_thread.start()

    def runOn(self, item):
        idx = item['idx']
        item = self.render(item)
        self.storage.put((idx, item))

    def run(self):
        currentIdx = -1
        stop = False
        start_time, count, total = time.time(), 0, 0
        while not stop:
            time.sleep(0.005)
            while not self.storage.empty():
                try:
                    idx, item = self.storage.get()
                    count += 1
                    total += 1
                    if count % 100 == 0:
                        print("output fps", count /
                              (time.time()-start_time), "total", total)
                        start_time, count = time.time(), 0
                    if idx > currentIdx:
                        currentIdx = idx
                        cv2.imshow("frame", item['output'])
                        cv2.waitKey(1)
                except:
                    break

    def render(self, item):
        frame, idx = item['frame'], item['idx']
        output = frame.copy()
        if 'face_locations' in item:
            face_locations = item['face_locations']
            for top, right, bottom, left in face_locations:
                cv2.rectangle(output, (left, top),
                              (right, bottom), (0, 0, 255), 2)
        # if 'face_landmarks_list' in item:
        #     for face_landmarks in item['face_landmarks_list']:
        #         for facial_feature in face_landmarks.keys():
        #             features = np.array(
        #                 face_landmarks[facial_feature], dtype=np.int32).reshape(-1, 1, 2)
        #             cv2.polylines(
        #                 output, features, True, (0, 255, 0), 2)
        item['output'] = output
        return item


class FaceLocator:
    def __init__(self, id):
        self.name = "FaceLocator"+str(id)

    def init(self):
        self.stop = False

    def runOn(self, item):
        frame, idx = item['frame'], item['idx']
        item['rgb'] = frame[:, :, ::-1]
        # print("locator", self.name, idx)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = fr.face_locations(rgb_small_frame)
        item['face_locations'] = [(top*4, right*4, bottom*4, left*4)
                                  for top, right, bottom, left in face_locations]
        # item['face_landmarks_list'] = fr.face_landmarks(
        #     item['rgb'], item['face_locations'])

        # item['face_encodings'] = fr.face_encodings(
        #     item['rgb'], item['face_locations'])
        return item


if __name__ == "__main__":
    n_locator = 15
    frameq = [MyQueue("frameq "+str(i)) for i in range(n_locator)]
    faceq = MyQueue("faceq")
    video_capture = Worker(oqueues=frameq, processor=VideoCaptureProcessor(
        path="/home/tqlong/Downloads/video.mp4"))
    face_locators = [Worker(iqueue=frameq[i], oqueues=[faceq],
                            processor=FaceLocator(i)) for i in range(n_locator)]
    video_render = Worker(iqueue=faceq, oqueues=None,
                          processor=VideoRenderProcessor())

    processes = [video_capture, video_render] + face_locators
    [t.start() for t in processes]
    [t.join() for t in processes]
