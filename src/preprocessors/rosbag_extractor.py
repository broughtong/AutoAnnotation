#!/usr/bin/python
import copy
import rospy
import pickle
import math
import sys
import os
import rosbag
import time
import concurrent
import concurrent.futures
import multiprocessing
import tqdm
import numpy as np
import laser_geometry.laser_geometry as lg
import sensor_msgs.point_cloud2 as pc2
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from tf_bag import BagTfTransformer
from scipy.spatial.transform import Rotation as R

import AutoAnnotation.src.utils.sentinel as sentinel

@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

class Extractor():
    def __init__(self, path, folder, filename, queue):

        self.path = path
        self.folder = folder
        self.filename = filename
        self.queue = queue

        self.lp = lg.LaserProjection()

        self.lastX, self.lastY = None, None
        self.distThresh = 2
        self.fileCounter = 0
        self.position = None
        self.scans = []
        self.transform = []
        self.ts = []

        self.scanTopics = ["/back_right/sick_safetyscanners/scan", 
                "/front/sick_safetyscanners/scan",
                "/back_left/sick_safetyscanners/scan",
                "/back_middle/scan"]
        self.synchroniseToTopic = self.scanTopics[-1] #MUST always be last from list (back middle higher fps)
        self.topicBuf = []
        for _ in range(len(self.scanTopics)-1):
            self.topicBuf.append(None)

        self.pointcloudScanTopic = ["/os_cloud_node/points"]
        self.pointcloudScanBuf = None
        self.pointclouds = []
        
        self.tfFrames = ["sick_back_left", 
                "sick_back_right", 
                "sick_back_middle",
                "sick_front",
                "os_sensor"]

    def run(self):
        
        self.queue.put("%s: Process Spawned" % (self.filename))

        try:
            with suppress_stdout_stderr():
                fn = os.path.join(self.path, self.folder, self.filename)
                self.bagtf = BagTfTransformer(os.path.join(self.path, self.folder, self.filename))
        except:
            self.queue.put("%s: Bag failed [1]" % (self.filename))
            self.queue.put(sentinel.PROBLEM)
            return 0

        self.extractStaticTFs()

        for topic, msg, t in rosbag.Bag(os.path.join(self.path, self.folder, self.filename)).read_messages():
            if topic in self.pointcloudScanTopic:
                self.pointcloudScanBuf = msg
            if topic in self.scanTopics:
                if topic == self.synchroniseToTopic:
                    self.processScan(msg, msg.header.stamp)
                else:
                    for i in range(len(self.scanTopics)):
                        if topic == self.scanTopics[i]:
                            self.topicBuf[i] = msg
            if topic == "/odom":
                self.position = msg.pose.pose.position
                self.orientation = msg.pose.pose.orientation

        self.saveScans()
        self.queue.put(sentinel.SUCCESS)
        if len(self.scans) != 0:
            self.queue.put("%s: Process complete, written %i frames" % (self.filename, len(self.scans)))
        else:
            self.queue.put("%s: Process complete" % (self.filename))
        return len(self.scans)

    def extractStaticTFs(self):

        self.staticTFs = {}

        for i in self.tfFrames:

            translation, quaternion = None, None

            try:
                with suppress_stdout_stderr():
                    tftime = self.bagtf.waitForTransform("base_link", i, None)
                    translation, quaternion = self.bagtf.lookupTransform("base_link", i, tftime)
            except:
                self.queue.put("%s: Warning, could not find static tf from %s to %s [1]" % (self.filename, i, "base_link"))
                continue

            r = R.from_quat(quaternion)
            mat = r.as_matrix()
            mat = np.pad(mat, ((0, 1), (0, 1)), mode='constant', constant_values=0)
            mat[0][-1] += translation[0]
            mat[1][-1] += translation[1]
            mat[2][-1] += translation[2]
            mat[3][-1] = 1

            self.staticTFs[i] = {}
            self.staticTFs[i]["translation"] = translation
            self.staticTFs[i]["quaternion"] = quaternion
            self.staticTFs[i]["mat"] = mat

    def odometryMoved(self):

        if self.lastX == None:
            if self.position == None:
                return False
            self.lastX = self.position.x
            self.lastY = self.position.y
            return True

        diffx = self.position.x - self.lastX
        diffy = self.position.y - self.lastY
        dist = ((diffx**2)+(diffy**2))**0.5
        
        if dist > self.distThresh:
            self.lastX = self.position.x
            self.lastY = self.position.y
            return True
        return False

    def combineScans(self, msgs, t):

        points = {}
        for msg in msgs:
            points[msg.header.frame_id] = []
            if t - msg.header.stamp > rospy.Duration(1, 0):
                self.queue.put("%s: Old Scan present" % (self.filename))
                return [], [], []

        for idx in range(len(msgs)):
            msgs[idx] = self.lp.projectLaser(msgs[idx])
            msg = pc2.read_points(msgs[idx])

            translation, quaternion = None, None
            
            try:
                with suppress_stdout_stderr():
                    translation, quaternion = self.bagtf.lookupTransform("base_link", msgs[idx].header.frame_id, msgs[idx].header.stamp)
            except:
                self.queue.put("%s: Warning, could not find tf from %s to %s [2]" % (self.filename, msgs[idx].header.frame_id, "base_link"))
                return [], [], []
            r = R.from_quat(quaternion)
            mat = r.as_matrix()
            mat = np.pad(mat, ((0, 1), (0,1)), mode='constant', constant_values=0)
            mat[0][-1] += translation[0]
            mat[1][-1] += translation[1]
            mat[2][-1] += translation[2]
            mat[3][-1] = 1

            for point in msg:
                intensity = point[3]
                point = np.array([*point[:3], 1])
                point = np.matmul(mat, point)
                point[-1] = intensity
                points[msgs[idx].header.frame_id].append(point)

        translation, quaternion = None, None
        try:
            with suppress_stdout_stderr():
                translation, quaternion = self.bagtf.lookupTransform("odom", "base_link", msgs[0].header.stamp)
        except:
            self.queue.put("%s: Warning, could not find tf from %s to %s [3]" % (self.filename, "base_link", "odom"))
            return [], [], []
        r = R.from_quat(quaternion)
        mat = r.as_matrix()
        mat = np.pad(mat, ((0, 1), (0,1)), mode='constant', constant_values=0)
        mat[0][-1] += translation[0]
        mat[1][-1] += translation[1]
        mat[2][-1] += translation[2]
        mat[3][-1] = 1

        transform = mat 

        for key, value in points.items():
            points[key] = self.filterRobot(value)

        return points, transform, t

    def filterRobot(self, points):

        newPoints = []

        for i in range(len(points)):
            p = points[i]
            if p[0] < 0.25 and p[0] > -1.4:
                if p[1] < 1.5 and p[1] > -1.5:
                    continue
            if p[0] < -1.3 and p[0] > -4.8:
                if p[1] < 1.3 and p[1] > -1.3:
                    continue
            newPoints.append(p)

        return newPoints

    def saveScans(self):

        if len(self.scans) == 0:
            self.queue.put("%s: Skipping saving empty bag" % (self.filename))
            return

        os.makedirs(os.path.join(scansPath, self.folder), exist_ok=True)
        os.makedirs(os.path.join(pointcloudsPath, self.folder), exist_ok=True)
        os.makedirs(os.path.join(dataPath, self.folder), exist_ok=True)

        fn = os.path.join(scansPath, self.folder, self.filename + ".pickle")
        with open(fn, "wb") as f:
            pickle.dump({"scans": self.scans}, f)
        fn = os.path.join(pointcloudsPath, self.folder, self.filename + ".pickle")
        with open(fn, "wb") as f:
            pickle.dump({"pointclouds": self.pointclouds}, f)
        fn = os.path.join(dataPath, self.folder, self.filename + ".pickle")
        with open(fn, "wb") as f:
            pickle.dump({"transform": self.transform, "ts": self.ts, "staticTFs": self.staticTFs}, f)

    def processScan(self, msg, t):

        t = rospy.Time(t.secs, t.nsecs)

        if None in self.topicBuf:
            return

        msgs = copy.deepcopy(self.topicBuf)
        msgs.append(msg)
        #comb_msgs = [*self.topicBuf, msg]

        if not self.odometryMoved():
            return

        combined, transform, ts = self.combineScans(msgs, t)
        if len(combined) == 0:
            return
        for key in combined.keys():
            combined[key] = np.array(combined[key])
            combined[key] = combined[key].reshape(combined[key].shape[0], 4)
        self.scans.append(combined)
        self.transform.append(transform)
        self.ts.append(ts)

        if self.pointcloudScanBuf is not None:
            self.pointclouds.append(self.cloudToArray(self.pointcloudScanBuf))
        else:
            self.pointclouds.append([])
        self.fileCounter += 1

        self.pointcloudScanBuf = None
        for i in range(len(self.topicBuf)):
            self.topicBuf[i] = None

    def cloudToArray(self, msg):

        translation, quaternion = None, None

        try:
            with suppress_stdout_stderr():
                translation, quaternion = self.bagtf.lookupTransform("base_link", msg.header.frame_id, msg.header.stamp)
        except:
            self.queue.put("%s: Warning, could not find tf from %s to %s [4]" % (self.filename, msg.header.frame_id, "base_link"))
            return []

        r = R.from_quat(quaternion)
        mat = r.as_matrix()
        mat = np.pad(mat, ((0, 1), (0,1)), mode='constant', constant_values=0)
        mat[0][-1] += translation[0]
        mat[1][-1] += translation[1]
        mat[2][-1] += translation[2]
        mat[3][-1] = 1

        points = pc2.read_points(msg, skip_nans=True)
        points = np.fromiter(points, dtype=np.dtype((float, 9)))
        points = np.pad(points, [(0, 0), (0, 1)], mode="constant", constant_values=1) ##
        points = np.apply_along_axis(self.transformPoint, 1, points, mat)
        robotPoints = np.where(points[:, 0] == None)
        points = np.delete(points, robotPoints, axis=0)

        return points

    def transformPoint(self, point, mat):
        #p = np.append(point[:3], 1)# np.array([*point[:3], 1])
        p = point[[0, 1, 2, -1]]
        p = np.matmul(mat, p)
        if (p[0]**2 + p[1]**2)**0.5 < 2.5:
            return [None] * 9
        return [*p[:3], *point[3:-1]]

def listener(q, total):
    pbar = tqdm.tqdm(total=total)
    for item in iter(q.get, sentinel.EXIT):
        if item == sentinel.SUCCESS or item == sentinel.PROBLEM:
            pbar.update()
        else:
            tqdm.tqdm.write(str(item))

if __name__ == "__main__":
    
    datasetPath = "../../data/rosbags/"
    """
    dataTopics = {
        "scans": [
            "/back_right/sick_safetyscanners/scan",
            "/front/sick_safetyscanners/scan",
            "/back_left/sick_safetyscanners/scan",
            "/back_middle/scan"
            ],
        "pointclouds": [
            "/os_cloud_node/points"
            ]
        }
    """
    scansPath = "../../data/scans"
    pointcloudsPath = "../../data/pointclouds"
    dataPath = "../../data/meta"

    manager = multiprocessing.Manager()
    queue = manager.Queue()

    jobs = []
    for files in os.walk(datasetPath):
        for filename in files[2]:
            if filename[-4:] == ".bag":
                path = datasetPath
                folder = files[0][len(path):]
                jobs.append(Extractor(path, folder, filename, queue))

    listenProcess = multiprocessing.Process(target=listener, args=(queue, len(jobs)))
    listenProcess.start()

    workers = 7
    futures = []
    queue.put("Starting %i jobs with %i workers" % (len(jobs), workers))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for job in jobs:
            f = ex.submit(job.run)
            futures.append(f)
            time.sleep(0.1)

        for future in futures:
            try:
                res = future.result()
            except Exception as e:
                queue.put("Process exception: " + str(e))

    results = []
    for i, job in enumerate(jobs):
        f = futures[i]
        value = [os.path.relpath(job.path, "../data/"), job.folder, job.filename, f.result()]
        results.append(value)

    with open(os.path.join(dataPath, "statistics.pkl"), "wb") as f:
        pickle.dump(results, f)
    
    queue.put(sentinel.EXIT)
    listenProcess.join()
