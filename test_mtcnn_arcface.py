import face_model
import argparse
import cv2
import sys
import numpy as np
from multiprocessing import Process, Queue, Lock, Value
from threading import Thread
import time
from queue import PriorityQueue, Full, Empty


class MyQueue():
    def __init__(self, name, maxsize=10):
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
    def __init__(self, wait_for_init, iqueue=None, oqueues=None, processor=None):
        super(Worker, self).__init__()
        self.iqueue = iqueue
        self.oqueues = oqueues
        self.processor = processor
        self.deamon = True
        self.wait_for_init = wait_for_init

    def run(self):
        self.processor.init()
        print(self.processor.name, "init")

        self.wait_for_init.value -= 1
        while self.wait_for_init.value > 0:
            time.sleep(0.1)

        oqueue_idx = 0
        while True:
            item = self.iqueue.qget(2) if self.iqueue is not None else -1
            # if self.iqueue is not None:
            #     print(self.iqueue.name, self.iqueue.queue.qsize())
            if item is not None:
                item = self.processor.runOn(item)
                if self.oqueues is not None:
                    self.oqueues[oqueue_idx].qput(item)
                    oqueue_idx = (oqueue_idx + 1) % len(self.oqueues)
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
        time.sleep(0.04)
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
        count, total = 0, 0
        while not stop:
            time.sleep(0.005)
            while not self.storage.empty():
                try:
                    idx, item = self.storage.get()
                    count += 1
                    total += 1
                    if total == 1:
                        start_time = time.time()

                    if count % 100 == 0:
                        print("output fps", count /
                              (time.time() - start_time), "total", total)
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
                cv2.rectangle(output, (int(left), int(top)),
                              (int(right), int(bottom)), (0, 0, 255), 2)
        # if 'face_landmarks_list' in item:
        #     for face_landmarks in item['face_landmarks_list']:
        #         for facial_feature in face_landmarks.keys():
        #             features = np.array(
        #                 face_landmarks[facial_feature], dtype=np.int32).reshape(-1, 1, 2)
        #             cv2.polylines(
        #                 output, features, True, (0, 255, 0), 2)
        item['output'] = output
        return item


# class FaceLocator:
#     def __init__(self, id):
#         self.name = "FaceLocator" + str(id)

#     def init(self):
#         self.stop = False

#     def runOn(self, item):
#         frame, idx = item['frame'], item['idx']
#         item['rgb'] = frame[:, :, ::-1]
#         # print("locator", self.name, idx)
#         small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#         rgb_small_frame = small_frame[:, :, ::-1]
#         face_locations = fr.face_locations(rgb_small_frame)
#         item['face_locations'] = [(top * 4, right * 4, bottom * 4, left * 4)
#                                   for top, right, bottom, left in face_locations]
#         # item['face_landmarks_list'] = fr.face_landmarks(
#         #     item['rgb'], item['face_locations'])

#         # item['face_encodings'] = fr.face_encodings(
#         #     item['rgb'], item['face_locations'])
#         return item


class FaceLocator1:
    def __init__(self, id, args):
        self.name = "FaceLocator1_" + str(id)
        self.args = args

    def init(self):
        self.stop = False
        self.model = face_model.FaceModel(args)
        img = cv2.imread('Tom_Hanks_54745.png')
        img, _, _ = self.model.get_input(img)
        f1 = self.model.get_feature(img)

    def runOn(self, item):
        frame, idx = item['frame'], item['idx']
        # print("locator1", self.name, idx)
        ret = self.model.get_input(frame)
        if ret is not None:
            img, bbox, points = ret
            left, top, right, bottom = bbox
            # print("bbox", bbox)
            item['face_locations'] = [
                (top, right, bottom, left)]
            item['face_encodings'] = [self.model.get_feature(img)]
        # item['face_landmarks_list'] = fr.face_landmarks(
        #     item['rgb'], item['face_locations'])

        # item['face_encodings'] = fr.face_encodings(
        #     item['rgb'], item['face_locations'])
        return item


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='face model test')
    # general
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='', help='path to load model.')
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int,
                        help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int,
                        help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24,
                        type=float, help='ver dist threshold')
    args = parser.parse_args()

    n_locator = 5
    num_init = Value('i', n_locator + 2)
    frameq = [MyQueue("frameq_" + str(i)) for i in range(n_locator)]
    faceq = MyQueue("faceq")
    video_capture = Worker(oqueues=frameq, processor=VideoCaptureProcessor(
        path="/home/tqlong/Downloads/video.mp4"), wait_for_init=num_init)
    face_locators = [Worker(iqueue=frameq[i], oqueues=[faceq],
                            processor=FaceLocator1(i, args), wait_for_init=num_init) for i in range(n_locator)]
    video_render = Worker(iqueue=faceq, oqueues=None,
                          processor=VideoRenderProcessor(), wait_for_init=num_init)

    processes = [video_capture, video_render] + face_locators
    [t.start() for t in processes]
    [t.join() for t in processes]


# model = face_model.FaceModel(args)
# img = cv2.imread('Tom_Hanks_54745.png')
# img = model.get_input(img)
# #f1 = model.get_feature(img)
# # print(f1[0:10])
# gender, age = model.get_ga(img)
# print(gender)
# print(age)
# sys.exit(0)
# img = cv2.imread(
#     '/raid5data/dplearn/megaface/facescrubr/112x112/Tom_Hanks/Tom_Hanks_54733.png')
# f2 = model.get_feature(img)
# dist = np.sum(np.square(f1-f2))
# print(dist)
# sim = np.dot(f1, f2.T)
# print(sim)
# #diff = np.subtract(source_feature, target_feature)
# #dist = np.sum(np.square(diff),1)
