#!/usr/bin/python
import rospy
import pickle
import math
import sys
import os
import time
import concurrent
import concurrent.futures
import multiprocessing
import tqdm
import numpy as np
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from scipy.spatial.transform import Rotation as R

import sensor_msgs.point_cloud2 as pc2

import AutoAnnotation.src.utils.lidar as lidar
import AutoAnnotation.src.utils.sentinel as sentinel

@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

class RViz():
    def __init__(self, path, folder, filename, queue, outputPath, scanField):

        self.path = path
        self.folder = folder
        self.filename = filename
        self.queue = queue
        self.outputPath = outputPath
        self.scanField = scanField

        self.data = {}
        self.initialisedPubs = False
    
    def run(self):

        self.queue.put("%s: Process spawned" % (self.filename))

        os.makedirs(os.path.join(self.outputPath, self.folder), exist_ok=True)

        with open(os.path.join(self.path, self.scanField, self.folder, self.filename), "rb") as f:
            self.data.update(pickle.load(f))
        with open(os.path.join(self.path, "meta", self.folder, self.filename), "rb") as f:
            self.data.update(pickle.load(f))
        try:
            with open(os.path.join(self.path, "pointclouds", self.folder, self.filename), "rb") as f:
                self.data.update(pickle.load(f))
        except:
            pass


        self.readProject()
        
        fn = os.path.join(self.outputPath, self.folder, self.filename)
        with open(fn, "wb") as f:
            pickle.dump(self.data, f)
        
        self.queue.put("%s: Process complete" % (self.filename))

    def readProject(self):

        transform = self.data["transform"]
        ts = self.data["ts"]
        scans = self.data["scans"]

        if self.initialisedPubs == False:
            self.initialisedPubs = True
            self.pubs = {}
            rospy.init_node("visualisation_debugger", anonymous=True)
            for scanner in scans[0].keys():
                self.pubs[scanner] = rospy.Publisher(scanner, pc2, queue_size=10)
            self.pubs["combined"] = rospy.Publisher("combined", pc2.PointCloud2, queue_size=10)

        for frameIdx in range(len(scans)):
            combined = lidar.combineScans(scans[frameIdx])
            msg = pc2.PointCloud2()
            msg.header.frame_id = "base_link"
            points = pc2.create_cloud_xyz32(msg.header, combined[:, :3])
            self.pubs["combined"].publish(points)
            time.sleep(0.05)
            self.queue.put(sentinel.SUCCESS)


        #for point in combined:
            
            


        """
        #move everything into the odom frame
        for idx in range(len(annotations)):
            robopose = np.array([-4.2, 0, 0, 1])
            mat = np.array(transform[idx])
            robopose = np.matmul(mat, robopose)[:2]
            robotPoses.append(robopose)
            r = R.from_matrix(mat[:3, :3])
            yaw = r.as_euler('zxy', degrees=False)
            yaw = yaw[0]
            detections = []

            for det in range(len(annotations[idx])):
                point = np.array([*annotations[idx][det][:2], 0, 1])
                point = np.matmul(mat, point)
                point = list(point[:2])

                orientation = annotations[idx][det][2] + yaw
                detections.append([*point, orientation])

            annotationsOdom.append(detections)
        with suppress_stdout_stderr():
            annotationsOdom = np.array(annotationsOdom)
        """

def listener(q, total):
    pbar = tqdm.tqdm(total=total)
    for item in iter(q.get, sentinel.EXIT):
        if item == sentinel.SUCCESS or item == sentinel.PROBLEM:
            pbar.update()
        else:
            tqdm.tqdm.write(str(item))

if __name__ == "__main__":

    projectFilename = "2022-02-15-15-08-59.bag"#022-06-21-20-45-50.bag"
    dataPath = "../../data"
    scanField = "scans"
    outputPath = "./data/"

    count = 0
    with open(os.path.join(dataPath, "meta", "statistics.pkl"), "rb") as f:
        data = pickle.load(f)
    for i in data:
        if projectFilename in i[2]:
            count += i[-1]

    manager = multiprocessing.Manager()
    queue = manager.Queue()
    listenProcess = multiprocessing.Process(target=listener, args=(queue, count))
    listenProcess.start()

    jobs = []
    for files in os.walk(os.path.join(dataPath, scanField)):
        for filename in files[2]:
            if projectFilename not in filename:
                continue
            path = dataPath
            folder = files[0][len(os.path.join(path, scanField))+1:]
            jobs.append(RViz(path, folder, filename, queue, outputPath, scanField))

    workers = 4
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

    queue.put(sentinel.EXIT)
    listenProcess.join()
