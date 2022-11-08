#!/usr/bin/python
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

import AutoAnnotation.src.utils.sentinel as sentinel

@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

class Temporal():
    def __init__(self, path, folder, filename, queue, detectionDistance, interpolateFrames, interpolateRequired, extrapolateFrames, extrapolateRequired, datasetPath, outputPath):

        self.path = path
        self.folder = folder
        self.filename = filename
        self.queue = queue
        self.datasetPath = datasetPath
        self.outputPath = outputPath

        self.detectionDistance = detectionDistance
        self.interpolateFrames = interpolateFrames
        self.interpolateRequired = interpolateRequired
        self.extrapolateFrames = extrapolateFrames
        self.extrapolateRequired = extrapolateRequired
        self.underRobotDistance = 1.8
        self.data = {}
        self.originalDetections = 0
        self.newDetections = 0
    
    def run(self):

        self.queue.put("%s: Process spawned" % (self.filename))

        os.makedirs(os.path.join(self.outputPath, self.folder), exist_ok=True)

        with open(os.path.join(self.path, "detector/default", self.folder, self.filename), "rb") as f:
            self.data.update(pickle.load(f))
        with open(os.path.join(self.path, "meta", self.folder, self.filename), "rb") as f:
            self.data.update(pickle.load(f))

        self.data["eroded"] = [[]]

        self.calculateNDetections()
        self.temporal()
        
        fn = os.path.join(self.outputPath, self.folder, self.filename)
        self.queue.put(fn)
        with open(fn, "wb") as f:
            pickle.dump(self.data, f)
        
        self.calculateNDetections()

        self.queue.put("%s: Process complete, n of detections %i -> %i" % (self.filename, self.originalDetections, self.newDetections))

    def temporal(self):

        transform = self.data["transform"]
        ts = self.data["ts"]
        annotations = self.data["annotations"]
        annotationsOdom = []

        robotPoses = []

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

        #if two detections close to each other in odom frame inside some temporal window
        #lerp between them
        lerped = [] #includes real detections
        lerpedOnly = [] #exclusively lerped
        for i in range(len(annotations)):
            lerped.append([])
            lerpedOnly.append([])

        ctr = 0
        for mainIdx in range(len(annotations) - self.interpolateFrames):
            for car in annotationsOdom[mainIdx]: #for each car in the current scan:

                cars = [car]
                carFrames = [mainIdx]

                for smallWindowIdx in range(mainIdx+1, mainIdx+self.interpolateFrames+1):
                    for possibleMatchIdx in range(len(annotationsOdom[smallWindowIdx])):
                        otherCar = annotationsOdom[smallWindowIdx][possibleMatchIdx]
                        
                        dx = cars[-1][0] - otherCar[0]
                        dy = cars[-1][1] - otherCar[1]
                        dist = ((dx**2) + (dy**2))**0.5

                        if dist < self.detectionDistance:
                            cars.append(otherCar)
                            carFrames.append(smallWindowIdx)

                if len(cars) < self.interpolateRequired:
                    continue
                if carFrames[-1] - carFrames[0] < self.interpolateRequired: #all consecutive
                    continue

                for frameIdx in range(len(carFrames[:-1])):
                    if carFrames[frameIdx+1] - carFrames[frameIdx] == 1: #consecutive frame
                        continue

                    startX, endX = cars[frameIdx][0], cars[frameIdx+1][0]
                    startY, endY = cars[frameIdx][1], cars[frameIdx+1][1]
                    startR, endR = cars[frameIdx][2], cars[frameIdx+1][2]

                    nFramesBetween = carFrames[frameIdx+1] - carFrames[frameIdx]

                    for i in range(1, nFramesBetween):  #carFrames[frameIdx]+1, carFrames[frameIdx+1]):

                        interval = i / nFramesBetween
                        x = self.lerp(startX, endX, interval)
                        y = self.lerp(startY, endY, interval)
                        r = self.lerp(startR, endR, interval, rot=True)

                        robopose = robotPoses[carFrames[frameIdx]+i]
                        dx = x - robopose[0] 
                        dy = y - robopose[1]
                        dist = ((dx**2) + (dy**2))**0.5

                        if dist > self.underRobotDistance: #stop detections from under the robot
                            lerped[carFrames[frameIdx]+i].append([x, y, r])
                            lerpedOnly[carFrames[frameIdx]+i].append([x, y, r])
            self.queue.put(sentinel.SUCCESS)
            ctr += 1
        while ctr < len(annotations):
            ctr += 1
            self.queue.put(sentinel.SUCCESS)

        #add real detections to lerped dataset
        for frameIdx in range(len(annotationsOdom)):
            for detectionIdx in range(len(annotationsOdom[frameIdx])):
                lerped[frameIdx].append(annotationsOdom[frameIdx][detectionIdx])

        #erode - remove single dets with no surrounding
        eroded = []
        for i in range(len(annotations)):
            eroded.append([])
        for mainIdx in range(len(annotations)):
            for car in lerped[mainIdx]: #for each car in the current scan:

                foundCar = False

                if mainIdx > 0:
                    for otherCar in lerped[mainIdx-1]:
                        dx = car[0] - otherCar[0]
                        dy = car[1] - otherCar[1]
                        dist = ((dx**2) + (dy**2))**0.5

                        if dist < self.detectionDistance:
                            foundCar = True
                            break

                if mainIdx < len(annotations)-1:
                    for otherCar in lerped[mainIdx+1]:
                        dx = car[0] - otherCar[0]
                        dy = car[1] - otherCar[1]
                        dist = ((dx**2) + (dy**2))**0.5

                        if dist < self.detectionDistance:
                            foundCar = True
                            break

                if foundCar == True:
                    eroded[mainIdx].append(car)

        cint = 0
        for i in eroded:
            cint += len(i)

        #some shitty nms due to interpolated windowing
        newEroded = []
        for mainIdx in range(len(annotations)):
            newFrame = []

            for carIdx in range(len(eroded[mainIdx])): #for each car in the current scan:
                
                unique = True
                car = eroded[mainIdx][carIdx]
                for otherCarIdx in range(carIdx+1, len(eroded[mainIdx])):

                    otherCar = eroded[mainIdx][otherCarIdx]

                    dx = car[0] - otherCar[0]
                    dy = car[1] - otherCar[1]
                    dist = ((dx**2) + (dy**2))**0.5

                    if dist < 0.4:
                        unique = False

                if unique == True:
                        newFrame.append(car)
             
            newEroded.append(newFrame)
        eroded = newEroded

        #extrapolate
        extrapolated = []
        extrapolatedOnly = []
        for i in range(len(annotations)):
            extrapolated.append([])
            extrapolatedOnly.append([])
        ctr = 0
        for frameIdx in range(len(annotations)):
            for detectionIdx in range(len(eroded[frameIdx])):

                #for each detection, we will check forward and back 1 frame

                #forward
                if frameIdx < len(eroded)-1:
                    nextFrame = eroded[frameIdx+1]
                    foundCar = False
                    for otherDetectionIdx in range(len(nextFrame)):
                        car = eroded[frameIdx][detectionIdx]
                        otherCar = eroded[frameIdx+1][otherDetectionIdx]
                        dx = car[0] - otherCar[0]
                        dy = car[1] - otherCar[1]
                        dist = ((dx**2) + (dy**2))**0.5

                        if dist < self.detectionDistance:
                            foundCar = True
                    if foundCar == False:
                        #no car on next frame, lets check back that there has been a few dets, then extrap
                        #it will have been interped, so we can just check self.extrapolateRequired
                        allRequired = True
                        for i in range(1, self.extrapolateRequired+1):
                            checkFrame = frameIdx - i
                            if checkFrame < 0:
                                allRequired = False
                                break
                            
                            foundMatchInFrame = False
                            for checkCar in eroded[checkFrame]:
                                car = eroded[frameIdx][detectionIdx]
                                dx = car[0] - checkCar[0]
                                dy = car[1] - checkCar[1]
                                dist = ((dx**2) + (dy**2))**0.5

                                if dist < self.detectionDistance:
                                    foundMatchInFrame = True
                                    break
                            
                            if foundMatchInFrame == False:
                                allRequired = False
                                break

                        if allRequired:
                            for i in range(frameIdx+1, frameIdx + self.extrapolateFrames):
                                if i <= len(eroded)-1: 
                                    robopose = robotPoses[i]
                                    dx = car[0] - robopose[0] 
                                    dy = car[1] - robopose[1]
                                    dist = ((dx**2) + (dy**2))**0.5
                    
                                    if dist > self.underRobotDistance: #stop detections from under the robot
                                        extrapolated[i].append(car)
                                        extrapolatedOnly[i].append(car)
                #backward
                if frameIdx > 0:
                    prevFrame = eroded[frameIdx-1]
                    foundCar = False
                    for otherDetectionIdx in range(len(prevFrame)):
                        car = eroded[frameIdx][detectionIdx]
                        otherCar = eroded[frameIdx-1][otherDetectionIdx]
                        dx = car[0] - otherCar[0]
                        dy = car[1] - otherCar[1]
                        dist = ((dx**2) + (dy**2))**0.5

                        if dist < self.detectionDistance:
                            foundCar = True
                    if foundCar == False:
                        #no car on prev frame, lets check forward that there has been a few dets, then extrap
                        #it will have been interped, so we can just check self.extrapolateRequired
                        allRequired = True

                        for i in range(1, self.extrapolateRequired+1):
                            checkFrame = frameIdx + i
                            if checkFrame >= len(eroded):
                                allRequired = False
                                break
                            
                            foundMatchInFrame = False
                            for checkCar in eroded[checkFrame]:
                                car = eroded[frameIdx][detectionIdx]
                                dx = car[0] - checkCar[0]
                                dy = car[1] - checkCar[1]
                                dist = ((dx**2) + (dy**2))**0.5

                                if dist < self.detectionDistance:
                                    foundMatchInFrame = True
                                    break
                            
                            if foundMatchInFrame == False:
                                allRequired = False
                                break

                        if allRequired:
                            for i in range(frameIdx - self.extrapolateFrames, frameIdx):
                                if i >= 0:
                                    robopose = robotPoses[i]
                                    dx = car[0] - robopose[0] 
                                    dy = car[1] - robopose[1]
                                    dist = ((dx**2) + (dy**2))**0.5

                                    if dist > self.underRobotDistance: #stop detections from under the robot
                                        extrapolated[i].append(car)
                                        extrapolatedOnly[i].append(car)
            self.queue.put(sentinel.SUCCESS)
            ctr += 1
        while ctr < len(annotations):
            ctr += 1
            self.queue.put(sentinel.SUCCESS)

        #add real detections to extrapolated
        for frameIdx in range(len(annotationsOdom)):
            for detectionIdx in range(len(eroded[frameIdx])):
                extrapolated[frameIdx].append(eroded[frameIdx][detectionIdx])

        cl = 0
        for i in lerped:
            cl += len(i)
        cer = 0
        for i in eroded:
            cer += len(i)
        cex = 0
        for i in extrapolated:
            cex += len(i)
            
        self.queue.put(str(self.originalDetections) + "->" + str(cl) + "->" + str(cint) + "->" + str(cer) + "->" + str(cex))

        #some shitty nms due to one extrapolater going forward possibly overlapping one going backwards
        newExtrapolated = []
        for mainIdx in range(len(annotations)):
            newFrame = []

            for carIdx in range(len(extrapolated[mainIdx])): #for each car in the current scan:
                
                unique = True
                car = extrapolated[mainIdx][carIdx]
                for otherCarIdx in range(carIdx+1, len(extrapolated[mainIdx])):

                    otherCar = extrapolated[mainIdx][otherCarIdx]

                    dx = car[0] - otherCar[0]
                    dy = car[1] - otherCar[1]
                    dist = ((dx**2) + (dy**2))**0.5

                    if dist < self.detectionDistance:
                        unique = False

                if unique == True:
                    newFrame.append(car)
             
            newExtrapolated.append(newFrame)

        extrapolated = newExtrapolated

        self.data["annotationsOdom"]  = annotationsOdom
        self.data["lerpedOdom"] = lerped
        self.data["extrapOdom"] = extrapolated
        self.data["lerped"] = []
        self.data["lerpedOnly"] = []
        self.data["eroded"] = []
        self.data["extrapolated"] = []
        self.data["extrapolatedOnly"] = []

        #move back to robot frame
        for idx in range(len(annotations)):
            mat = np.array(transform[idx])
            mat = np.linalg.inv(mat)
            r = R.from_matrix(mat[:3, :3])
            yaw = r.as_euler('zxy', degrees=False)
            yaw = yaw[0]
            detections = []
            detectionsExtrap = []
            detectionsOnly = []
            detectionsExtrapOnly = []
            erodedRob = []

            for det in range(len(lerped[idx])):
                point = np.array([*lerped[idx][det][:2], 0, 1])
                point = np.matmul(mat, point)
                point = list(point[:2])
                
                orientation = lerped[idx][det][2] + yaw
                detections.append([*point, orientation])
                
            for det in range(len(extrapolated[idx])):
                point = np.array([*extrapolated[idx][det][:2], 0, 1])
                point = np.matmul(mat, point)
                point = list(point[:2])
                
                orientation = extrapolated[idx][det][2] + yaw
                detectionsExtrap.append([*point, orientation])
                
            for det in range(len(lerpedOnly[idx])):
                point = np.array([*lerpedOnly[idx][det][:2], 0, 1])
                point = np.matmul(mat, point)
                point = list(point[:2])
                
                orientation = lerpedOnly[idx][det][2] + yaw
                detectionsOnly.append([*point, orientation])
                
            for det in range(len(extrapolatedOnly[idx])):
                point = np.array([*extrapolatedOnly[idx][det][:2], 0, 1])
                point = np.matmul(mat, point)
                point = list(point[:2])
                
                orientation = extrapolatedOnly[idx][det][2] + yaw
                detectionsExtrapOnly.append([*point, orientation])

            for det in range(len(eroded[idx])):
                point = np.array([*eroded[idx][det][:2], 0, 1])
                point = np.matmul(mat, point)
                point = list(point[:2])
                
                orientation = eroded[idx][det][2] + yaw
                erodedRob.append([*point, orientation])

            self.data["lerped"].append(detections)
            self.data["extrapolated"].append(detectionsExtrap)
            self.data["lerpedOnly"].append(detectionsOnly)
            self.data["extrapolatedOnly"].append(detectionsExtrapOnly)
            self.data["eroded"].append(erodedRob)
        with suppress_stdout_stderr():
            self.data["lerped"] = np.array(self.data["lerped"])
            self.data["extrapolated"] = np.array(self.data["extrapolated"])
            self.data["lerpedOnly"] = np.array(self.data["lerpedOnly"])
            self.data["extrapolatedOnly"] = np.array(self.data["extrapolatedOnly"])
            self.data["eroded"] = np.array(self.data["eroded"])

    def lerp(self, a, b, i, rot=False):
        if rot:
            a = a % math.pi
            b = b % math.pi
            if abs(a - b) > (math.pi/2):
                if a < b:
                    a += math.pi
                else:
                    b += math.pi
        return a + i * (b - a)
    
    def calculateNDetections(self):
        for i in self.data["annotations"]:
            self.originalDetections += len(i)
        for i in self.data["eroded"]:
            self.newDetections += len(i)

def listener(q, total):
    pbar = tqdm.tqdm(total=total)
    for item in iter(q.get, sentinel.EXIT):
        if item == sentinel.SUCCESS or item == sentinel.PROBLEM:
            pbar.update()
        else:
            tqdm.tqdm.write(str(item))

if __name__ == "__main__":

    dataPath = "../../data"
    outputPath = "../../data/temporal/default"

    count = 0
    with open(os.path.join(dataPath, "meta", "statistics.pkl"), "rb") as f:
        data = pickle.load(f)
    for i in data:
        count += i[-1]

    manager = multiprocessing.Manager()
    queue = manager.Queue()
    listenProcess = multiprocessing.Process(target=listener, args=(queue, count*2))
    listenProcess.start()

    jobs = []
    for files in os.walk(os.path.join(dataPath, "detector/default")):
        for filename in files[2]:
            path = dataPath
            folder = files[0][len(os.path.join(path, "detector/default"))+1:]
            jobs.append(Temporal(path, folder, filename, queue, 1.0, 20, 10, 20, 10, dataPath, outputPath))
            #distance thresh, interp window, interp dets req, extrap window, extrap dets req

    #jobs = jobs[:1]
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
