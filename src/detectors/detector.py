#!/usr/bin/python
import shutil
import pickle
import math
import sys
import time
import os
import tqdm
import numpy as np
import concurrent
import concurrent.futures
import multiprocessing
from sklearn.cluster import DBSCAN

import cv2

import AutoAnnotation.src.utils.sentinel as sentinel
import AutoAnnotation.src.utils.lidar as lidar
import AutoAnnotation.src.utils.draw as draw
import AutoAnnotation.src.utils.geometry as geometry

class Detector():
    def __init__(self, path, folder, filename, queue, outputPath):

        self.queue = queue
        self.path = path
        self.folder = folder
        self.filename = filename
        self.outputPath = outputPath
        self.fileCounter = 0
        self.detections = []
        self.relaxed = []
        self.data = {}
        self.maxPointsInInner = 5
        self.nDetections = 0

    def run(self):
        
        self.queue.put("%s: Process spawned" % (self.filename))

        os.makedirs(os.path.join(self.outputPath, self.folder), exist_ok=True)

        with open(os.path.join(self.path, "scans", self.folder, self.filename), "rb") as f:
            self.data.update(pickle.load(f))

        for idx in range(len(self.data["scans"])):
            scan = self.data["scans"][idx]
            cars = self.processScan(scan, method="strict")
            self.nDetections += len(cars)
            self.detections.append(cars)
            self.fileCounter += 1
            self.queue.put(sentinel.SUCCESS)
        
        self.data = {}
        self.data["annotations"] = self.detections
        fn = os.path.join(outputPath, self.folder, self.filename)
        with open(fn, "wb") as f:
            pickle.dump(self.data, f)
        
        self.queue.put("%s: Process complete, detected %i objects" % (self.filename, self.nDetections))

    def processScan(self, scan, method="strict"):

        full = lidar.combineScans(scan)
        p = lidar.combineScans({"sick_back_left": scan["sick_back_left"], "sick_back_right": scan["sick_back_right"]})
        cars = self.detect(p, full)
        return cars

    def detect(self, points, scan):

        dbscan = DBSCAN(eps=0.25, min_samples=4)
        clusterer = dbscan.fit_predict(points[:, :2])

        nClusters = np.unique(clusterer)
        nClusters = len(nClusters[nClusters != -1])

        centres = np.empty((0, 2))
        for i in range(nClusters):
            cluster = points[np.nonzero(clusterer == i)]
            centre = np.mean(cluster, axis=0)
            #if centre[0] >0 or centre[1] < 0:
            #    continue
            centres = np.vstack((centres, [centre[0], centre[1]]))

        pairedCentres = []
        width = 1.54
        widthTolerance = 0.11
        for i in range(len(centres)):
            for j in range(i + 1, len(centres)):
                dx = centres[i][0] - centres[j][0]
                dy = centres[i][1] - centres[j][1]
                dist = (dx**2 + dy**2)**0.5
                if dist < width + widthTolerance and dist > width - widthTolerance:
                    pairedCentres.append([centres[i], i, centres[j], j])

        length = 2.555
        lengthTolerance = 0.15
        diagonal = 2.92
        diagonalTolerance = 0.18
        triplets = []
        tripletsConfirmed = []
        tripletsHollow = []
        fourthTolerance = 0.3
        carVertices = []
        carVerticesInner = []
        for pair in pairedCentres:
            for i in range(len(centres)):
                if i == pair[1] or i == pair[3]:
                    continue
                dx = centres[i][0] - pair[0][0]
                dy = centres[i][1] - pair[0][1]
                dista = (dx**2 + dy**2)**0.5
                dx = centres[i][0] - pair[2][0]
                dy = centres[i][1] - pair[2][1]
                distb = (dx**2 + dy**2)**0.5

                cornerPos = None
                cornerIdx = None
                otherPos = None
                otherIdx = None
                dist = None

                if dista < distb:
                    cornerPos = pair[0]
                    cornerIdx = pair[1]
                    otherPos = pair[2]
                    otherIdx = pair[3]
                    dist = dista
                else:
                    cornerPos = pair[2]
                    cornerIdx = pair[3]
                    otherPos = pair[0]
                    otherIdx = pair[1]
                    dist = distb

                if dist < length + lengthTolerance and dist > length - lengthTolerance:
                    triplets.append([cornerPos, cornerIdx, otherPos, otherIdx, centres[i], i])
                    dx = centres[i][0] - otherPos[0]
                    dy = centres[i][1] - otherPos[1]
                    dist = (dx**2 + dy**2)**0.5
                    if dist < diagonal + diagonalTolerance and dist > diagonal - diagonalTolerance:
                        tripletsConfirmed.append([cornerPos, cornerIdx, otherPos, otherIdx, centres[i], i])

                        midX = (otherPos[0] + centres[i][0])/2
                        midY = (otherPos[1] + centres[i][1])/2
                        deltaX = midX - cornerPos[0]
                        deltaY = midY - cornerPos[1]
                        fourthX = cornerPos[0] + (2*deltaX)
                        fourthY = cornerPos[1] + (2*deltaY)

                        for j in range(len(centres)):
                            if j == cornerIdx or j == otherIdx or j == i:
                                continue
                            dx = fourthX - centres[j][0]
                            dy = fourthY - centres[j][1]
                            dist = (dx**2 + dy**2)**0.5
                            if dist < fourthTolerance:
                                fourthX = centres[j][0]
                                fourthY = centres[j][1]

                        vertices = [cornerPos, otherPos, centres[i], [fourthX, fourthY]]
                        carVertices.append(vertices)

                        innerRatio = 0.2
                        if innerRatio <= 0 or innerRatio >= 0.5:
                            #self.queue.put(str(innerRation) + " not within 0<x<0.5")
                            pass

                        newVertices = []

                        current = vertices[0]
                        opposite = vertices[3] 
                        newVertexX = (innerRatio*current[0]) + ((1-innerRatio)*opposite[0])
                        newVertexY = (innerRatio*current[1]) + ((1-innerRatio)*opposite[1])
                        newVertex = [newVertexX, newVertexY]
                        newVertices.append(newVertex)
                                         
                        current = vertices[1]
                        opposite = vertices[2] 
                        newVertexX = (innerRatio*current[0]) + ((1-innerRatio)*opposite[0])
                        newVertexY = (innerRatio*current[1]) + ((1-innerRatio)*opposite[1])
                        newVertex = [newVertexX, newVertexY]
                        newVertices.append(newVertex)
                        
                        current = vertices[2]
                        opposite = vertices[1] 
                        newVertexX = (innerRatio*current[0]) + ((1-innerRatio)*opposite[0])
                        newVertexY = (innerRatio*current[1]) + ((1-innerRatio)*opposite[1])
                        newVertex = [newVertexX, newVertexY]
                        newVertices.append(newVertex)
                        
                        current = vertices[3]
                        opposite = vertices[0] 
                        newVertexX = (innerRatio*current[0]) + ((1-innerRatio)*opposite[0])
                        newVertexY = (innerRatio*current[1]) + ((1-innerRatio)*opposite[1])
                        newVertex = [newVertexX, newVertexY]
                        newVertices.append(newVertex)

                        pointsInside = 0
                        counterPos = (10000, 0)
                        for p in points:
                            intersects = 0

                            for idx in range(len(newVertices)):
                                pa = newVertices[idx]
                                pb = newVertices[0]
                                if idx != len(newVertices)-1:
                                    pb = newVertices[idx+1]

                                if geometry.lineIntersect([p[0], p[1]], counterPos, pa, pb):
                                    intersects += 1

                            if intersects % 2:
                                pointsInside += 1
                        
                        if pointsInside <= self.maxPointsInInner:
                            carVerticesInner.append(newVertices)

        res = 1024
        scale = 25
        img = np.zeros((res, res, 3))
        img.fill(255)
            
        for point in scan:
            x, y = point[:2]
            x *= scale
            y *= scale
            x = int(x)
            y = int(y)
            try:
                img[x+int(res/2), y+int(res/2)] = [0, 0, 0]
            except:
                pass

        for i in centres:
            x, y = i[:2]
            x *= scale
            y *= scale
            x = int(x)
            y = int(y)
            col = [0, 0, 255]
            try:
                img[x+int(res/2), y+int(res/2)] = col
            except:
                pass

            try:
                img[x+int(res/2)+1, y+int(res/2)+1] = col
            except:
                pass
            try:
                img[x+int(res/2)+1, y+int(res/2)-1] = col
            except:
                pass
            try:
                img[x+int(res/2)-1, y+int(res/2)+1] = col
            except:
                pass
            try:
                img[x+int(res/2)-1, y+int(res/2)-1] = col
            except:
                pass

        for i in pairedCentres:
            x, y = i[0][:2]
            x *= scale
            y *= scale
            x = int(x)
            y = int(y)
            col = [255, 0, 0]
            try:
                img[x+int(res/2), y+int(res/2)] = col
            except:
                pass

            try:
                img[x+int(res/2)+1, y+int(res/2)+1] = col
            except:
                pass
            try:
                img[x+int(res/2)+1, y+int(res/2)-1] = col
            except:
                pass
            try:
                img[x+int(res/2)-1, y+int(res/2)+1] = col
            except:
                pass
            try:
                img[x+int(res/2)-1, y+int(res/2)-1] = col
            except:
                pass
            x, y = i[2][:2]
            x *= scale
            y *= scale
            x = int(x)
            y = int(y)
            try:
                img[x+int(res/2), y+int(res/2)] = col
            except:
                pass

            try:
                img[x+int(res/2)+1, y+int(res/2)+1] = col
            except:
                pass
            try:
                img[x+int(res/2)+1, y+int(res/2)-1] = col
            except:
                pass
            try:
                img[x+int(res/2)-1, y+int(res/2)+1] = col
            except:
                pass
            try:
                img[x+int(res/2)-1, y+int(res/2)-1] = col
            except:
                pass

        for i in tripletsConfirmed:
        #for i in triplets:
            col = [0, 255, 0]
            x, y = i[0][:2]
            x *= scale
            y *= scale
            x = int(x)
            y = int(y)
            try:
                img[x+int(res/2), y+int(res/2)] = col
            except:
                pass

            try:
                img[x+int(res/2)+1, y+int(res/2)+1] = col
            except:
                pass
            try:
                img[x+int(res/2)+1, y+int(res/2)-1] = col
            except:
                pass
            try:
                img[x+int(res/2)-1, y+int(res/2)+1] = col
            except:
                pass
            try:
                img[x+int(res/2)-1, y+int(res/2)-1] = col
            except:
                pass
            x, y = i[2][:2]
            x *= scale
            y *= scale
            x = int(x)
            y = int(y)
            try:
                img[x+int(res/2), y+int(res/2)] = col
            except:
                pass

            try:
                img[x+int(res/2)+1, y+int(res/2)+1] = col
            except:
                pass
            try:
                img[x+int(res/2)+1, y+int(res/2)-1] = col
            except:
                pass
            try:
                img[x+int(res/2)-1, y+int(res/2)+1] = col
            except:
                pass
            try:
                img[x+int(res/2)-1, y+int(res/2)-1] = col
            except:
                pass
            x, y = i[4][:2]
            x *= scale
            y *= scale
            x = int(x)
            y = int(y)
            try:
                img[x+int(res/2), y+int(res/2)] = col
            except:
                pass

            try:
                img[x+int(res/2)+1, y+int(res/2)+1] = col
            except:
                pass
            try:
                img[x+int(res/2)+1, y+int(res/2)-1] = col
            except:
                pass
            try:
                img[x+int(res/2)-1, y+int(res/2)+1] = col
            except:
                pass
            try:
                img[x+int(res/2)-1, y+int(res/2)-1] = col
            except:
                pass

        carVertices.append

        carPos = []
        for i in tripletsConfirmed:
            break
            midX = (i[2][0] + i[4][0])/2
            midY = (i[2][1] + i[4][1])/2
            carPos.append([midX, midY])

            x, y = midX, midY
            x *= scale
            y *= scale
            x = int(x)
            y = int(y)
            x += int(res/2)
            y += int(res/2)
            size = 4
            img[x-size:x+size, y-size:y+size] = [255, 255, 0]
        
        for i in carVerticesInner:
            for val in i:
                x, y = val[0], val[1]
                x *= scale
                y *= scale
                x = int(x)
                y = int(y)
                x += int(res/2)
                y += int(res/2)
                size = 6
                img[x-size:x+size, y-size:y+size] = [255, 0, 255]

        dilation=3
        kernel = np.ones((dilation, dilation), 'uint8')
        img = cv2.erode(img, kernel, iterations=1)
        cv2.imwrite(visualisationPath + "/" + self.filename + str(self.fileCounter) + ".png", img)

        cars = []
        for i in carVerticesInner:
            x = 0
            y = 0
            for p in i:
                x += p[0]
                y += p[1]
            x /= 4
            y /= 4
            dX = i[0][0] - i[2][0]
            dY = i[0][1] - i[2][1]
            angle = math.atan2(dY, dX)
            angle = angle % math.pi
            car = [x, y, angle]
            cars.append(car)

        return cars

def listener(q, total):
    pbar = tqdm.tqdm(total=total)
    for item in iter(q.get, sentinel.EXIT):
        if item == sentinel.SUCCESS or item == sentinel.PROBLEM:
            pbar.update()
        else:
            tqdm.tqdm.write(str(item))

if __name__ == "__main__":

    visualisationPath = "../../visualisation/detector"
    datasetPath = "../../data"
    outputPath = "../../data/detector/default"

    count = 0
    with open(os.path.join(datasetPath, "meta", "statistics.pkl"), "rb") as f:
        data = pickle.load(f)
    for i in data:
        count += i[-1]

    os.makedirs(outputPath, exist_ok=True)

    manager = multiprocessing.Manager()
    queue = manager.Queue()
    listenProcess = multiprocessing.Process(target=listener, args=(queue, count))
    listenProcess.start()

    jobs = []
    for files in os.walk(os.path.join(datasetPath, "scans")):
        for filename in files[2]:
            path = datasetPath
            folder = files[0][len(os.path.join(datasetPath, "scans"))+1:]
            jobs.append(Detector(path, folder, filename, queue, outputPath))
    
    workers = 8
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

