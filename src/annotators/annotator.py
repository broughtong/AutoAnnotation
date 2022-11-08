#!/usr/bin/python
import random
import pickle
import math
import sys
import tqdm
import os
import time
import numpy as np
import concurrent
import concurrent.futures
import multiprocessing
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from scipy.spatial.transform import Rotation as R

import AutoAnnotation.src.utils.sentinel as sentinel
import AutoAnnotation.src.utils.lidar as lidar
import AutoAnnotation.src.utils.geometry as geometry
import AutoAnnotation.src.utils.draw as draw
import AutoAnnotation.src.annotators.backends.maskrcnn as maskrcnn
import AutoAnnotation.src.annotators.backends.pointcloud as pointcloud

@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

class Annotator():
    def __init__(self, path, folder, filename, queue, scanField, labelSources, methods, outputPath, movementThreshold):

        self.path = path
        self.folder = folder
        self.filename = filename
        self.queue = queue
        self.scanField = scanField
        self.labelSources = labelSources
        self.outputPath = outputPath
        self.fileCounter = 0
        self.data = {}
        self.labelData = {}
        self.annotators = []
        self.movementThreshold = movementThreshold

        for method in methods:
            for key, value in self.labelSources.items():
                self.annotators.append(method.AnnotatorBackend(outputPath, scanField, key, value))

    def run(self):
        
        if self.filename in gtBags:
            self.queue.put("%s: Process spawned (ground-truth mode)" % (self.filename))
        else:
            self.queue.put("%s: Process spawned" % (self.filename))

        with open(os.path.join(self.path, self.scanField, self.folder, self.filename), "rb") as f:
            self.data.update(pickle.load(f))
        with open(os.path.join(self.path, "meta", self.folder, self.filename), "rb") as f:
            self.data.update(pickle.load(f))

        for key, value in self.labelSources.items():
            
            with open(os.path.join(self.path, key, self.folder, self.filename), "rb") as f:
                self.labelData = pickle.load(f)

            for i in self.annotators:
                self.annotate(key, value)

    def annotate(self, annotationType, annotationField):

        previousX, previousY = None, None
        for frameIdx in range(len(self.data[self.scanField])):

            scan = self.data[self.scanField][frameIdx]
            combined = [lidar.combineScans(scan)]
            annotations = [self.labelData[annotationField][frameIdx]]

            if self.filename not in gtBags:
                #throw away frames without much movement
                transform = self.data["transform"][frameIdx]
                x = transform[0][-1]
                y = transform[1][-1]

                if previousX == None and previousY == None:
                    previousX = x
                    previousY = y
                else:
                    diffx = x - previousX
                    diffy = y - previousY
                    dist = ((diffx**2)+(diffy**2))**0.5

                    if dist < self.movementThreshold:
                        self.queue.put(sentinel.SUCCESS)
                        continue

                    previousX = x
                    previousY = y
                
                if len(annotations[0]) == 0:
                    self.queue.put(sentinel.SUCCESS)
                    continue

                #perform augmentations
                for rotation in rotations:
                    if rotation == 0:
                        continue
                    r = np.identity(4)
                    orientationR = R.from_euler('z', 0)
                    if rotation != 0:
                        orientationR = R.from_euler('z', rotation)
                        r = R.from_euler('z', rotation)
                        r = r.as_matrix()
                        r = np.pad(r, [(0, 1), (0, 1)])
                        r[-1][-1] = 1

                    newScan = np.copy(combined[0])
                    for point in range(len(newScan)):
                        newScan[point] = np.matmul(r, newScan[point])
                    combined.append(newScan)

                    newAnnotations = np.copy(annotations[0])
                    for i in range(len(newAnnotations)):
                        v = [*annotations[0][i][:2], 1, 1]
                        v = np.matmul(r, v)[:2]
                        o = orientationR.as_euler('zxy', degrees=False)
                        o = o[0]
                        o += annotations[0][i][2]
                        newAnnotations[i] = [*v, o]
                    annotations.append(newAnnotations)

            for augIdx in range(len(combined)):

                for annotator in self.annotators:
                    
                    filename = self.filename + "-" + str(self.fileCounter)
                    data = combined[augIdx]
                    labels = annotations[augIdx]
                    annotator.annotate(filename, data, labels)
                    self.fileCounter += 1
                
            self.queue.put(sentinel.SUCCESS)

            continue
            return
            if self.filename in gtBags and 1==2:
                fn = os.path.join(outputPath, scanField, "mask", "all", "imgs", self.filename + "-" + str(frame) + ".png")
                draw.drawImgFromPoints(fn, newScan, [], [], [], [], dilation=3)
                
                fn = os.path.join(outputPath, scanField, "mask", "all", "annotations", self.filename + "-" + str(frame) + ".png")
                carPoints, nonCarPoints = geometry.getInAnnotation(newScan, newAnnotations)
                badAnnotation = draw.drawAnnotation(fn, frame, newAnnotations)

                fn = os.path.join(outputPath, scanField, "mask", "all", "debug", self.filename + "-" + str(frame) + ".png")
                draw.drawImgFromPoints(fn, newScan, [], [], newAnnotations, [], dilation=None)

                fn = os.path.join(outputPath, scanField, "pointcloud", "all", "cloud", self.filename + "-" + str(frame) + ".")
                self.saveCloud(fn, newScan)
                fn = os.path.join(outputPath, scanField, "pointcloud", "all", "annotations", self.filename + "-" + str(frame) + ".")
                self.saveAnnotations(fn, newScan, newAnnotations)

            #raw
            if self.filename not in gtBags and len(newAnnotations):

                for method in self.annotators:
                    filename = self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation)

                    method.annotate(filename, combined, newAnnotations, scanField)

                fn = os.path.join(outputPath, scanField, "mask", "all", "imgs", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + ".png")
                draw.drawImgFromPoints(fn, newScan, [], [], [], [], dilation=5)
                
                fn = os.path.join(outputPath, scanField, "mask", "all", "annotations", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + ".png")
                carPoints, nonCarPoints = geometry.getInAnnotation(newScan, newAnnotations)
                badAnnotation = self.drawAnnotation(fn, frame, newAnnotations)

                fn = os.path.join(outputPath, scanField, "mask", "all", "debug", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + ".png")
                draw.drawImgFromPoints(fn, newScan, [], [], newAnnotations, [], dilation=None)

                fn = os.path.join(outputPath, scanField, "pointcloud", "all", "cloud", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + ".")
                #self.saveCloud(fn, newScan)
                fn = os.path.join(outputPath, scanField, "pointcloud", "all", "annotations", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + ".ply")
                #self.saveAnnotations(fn, newScan, newAnnotations, cloud)

            if flipV and len(newAnnotations) and self.filename not in gtBags :

                fScan = np.copy(newScan)
                for point in range(len(newScan)):
                    fScan[point][0] = -fScan[point][0]
                fAnnotations = np.copy(newAnnotations)
                for i in range(len(annotations)):
                    fAnnotations[i][0] = -fAnnotations[i][0]
                    fAnnotations[i][2] = -fAnnotations[i][2]

                fn = os.path.join(outputPath, scanField, "mask", "all", "imgs", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + "-V.png")
                draw.drawImgFromPoints(fn, fScan, [], [], [], [], dilation=3)
                
                fn = os.path.join(outputPath, scanField, "mask", "all", "annotations", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + "-V.png")
                carPoints, nonCarPoints = geometry.getInAnnotation(fScan, fAnnotations)
                badAnnotation = self.drawAnnotation(fn, frame, fAnnotations) 

                fn = os.path.join(outputPath, scanField, "mask", "all", "debug", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + "-V.png")
                draw.drawImgFromPoints(fn, fScan, [], [], fAnnotations, [], dilation=None)

                fn = os.path.join(outputPath, scanField, "pointcloud", "all", "cloud", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + "-V.")
                self.saveCloud(fn, newScan)
                fn = os.path.join(outputPath, scanField, "pointcloud", "all", "annotations", self.filename + "-" + str(frame) + "-" + '{0:.2f}'.format(rotation) + "-V.")
                #self.saveAnnotations(fn, newScan, newAnnotations)

def listener(q, total):
    pbar = tqdm.tqdm(total=total)
    for item in iter(q.get, sentinel.EXIT):
        if item == sentinel.SUCCESS or item == sentinel.PROBLEM:
            pbar.update()
        else:
            tqdm.tqdm.write(str(item))

if __name__ == "__main__":

    methods = [maskrcnn]#, pointcloud]
    datasetPath = "../../data"
    outputPath = "../../annotations"
    scanFields = ["scans"]
    labelSources = {"temporal/default": "extrapolated"}
    movementThreshold = 0.5
    gtPath = "../../data/gt"
    gtBags = []

    #augmentations
    flipV = True
    rotations = [0, 1, 2, 3, 4, 5, 6]
    rotations = [0, 1]
    saltPepperNoise = 0.001

    count = 0
    with open(os.path.join(datasetPath, "meta", "statistics.pkl"), "rb") as f:
        data = pickle.load(f)
    for i in data:
        count += i[-1]

    manager = multiprocessing.Manager()
    queue = manager.Queue()
    listenProcess = multiprocessing.Process(target=listener, args=(queue, count))
    listenProcess.start()

    for files in os.walk(gtPath):
        for fn in files[2]:
            if "-lidar.pkl" in fn:
                fn = fn.split("-")[:-1]
                fn = "-".join(fn)
                fn += ".bag.pickle"
                gtBags.append(fn)

    jobs = []
    for scanField in scanFields:
        for files in os.walk(os.path.join(datasetPath, scanField)):
            for filename in files[2]:
                path = datasetPath
                folder = files[0][len(path)+len(scanField)+2:]
                jobs.append(Annotator(path, folder, filename, queue, scanField, labelSources, methods, outputPath, movementThreshold))

    #jobs = jobs[:1]
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
