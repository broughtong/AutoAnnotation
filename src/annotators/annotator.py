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
    
        #augmentations
        self.flipV = False
        self.rotations = [0, 1, 2, 3, 4, 5, 6]
        self.rotations = [0]
        self.saltPepperNoise = 0.001

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
        
        self.queue.put("%s: Process complete" % (self.filename))

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
                for rotation in self.rotations:
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
                        pPos, pAttrib = newScan[point][:3], newScan[point][3:]
                        pPos = np.concatenate([pPos, [1]])
                        pPos = np.matmul(r, pPos)
                        newScan[point] = np.concatenate([pPos[:3], pAttrib])
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

                if self.flipV:
                    for idx in range(len(combined)):
                        v = random.random()
                        if v > 0.5:
                            fScan = np.copy(combined[idx])
                            for point in range(len(fScan)):
                                fScan[point][0] = -fScan[point][0]
                            combined[idx] = fScan
                            fAnnotations = np.copy(annotations[idx])
                            for i in range(len(fAnnotations)):
                                fAnnotations[i][0] = -fAnnotations[i][0]
                                fAnnotations[i][2] = -fAnnotations[i][2]
                            annotations[idx] = fAnnotations
            
            for augIdx in range(len(combined)):

                for annotator in self.annotators:
                    
                    if len(combined) == 1:
                        filename = self.filename + "-" + str(self.fileCounter)
                    else:
                        filename = self.filename + "-" + str(self.fileCounter) + "-" + str(augIdx)
                    data = combined[augIdx]
                    labels = annotations[augIdx]
                    annotator.annotate(filename, data, labels)

            self.fileCounter += 1
            self.queue.put(sentinel.SUCCESS)

def listener(q, total):
    pbar = tqdm.tqdm(total=total)
    for item in iter(q.get, sentinel.EXIT):
        if item == sentinel.SUCCESS or item == sentinel.PROBLEM:
            pbar.update()
        else:
            tqdm.tqdm.write(str(item))

if __name__ == "__main__":

    methods = [maskrcnn, pointcloud]
    datasetPath = "../../data"
    outputPath = "../../annotations"
    scanFields = ["pointclouds", "scans"]
    labelSources = {"temporal/default": "extrapolated"}
    movementThreshold = 0.5
    gtPath = "../../data/gt"
    gtBags = []

    #calculate the total number of training samples
    count = 0
    with open(os.path.join(datasetPath, "meta", "statistics.pkl"), "rb") as f:
        data = pickle.load(f)
    for i in data:
        count += i[-1]
    count *= len(methods)
    count *= len(scanFields)
    ctr = 0
    for key in labelSources.items():
        ctr += 1
    count *= ctr

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

    workers = 3
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
