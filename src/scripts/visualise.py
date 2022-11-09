#!/usr/bin/python
import shutil
import multiprocessing
import pickle
import os
import numpy as np
import time
import tqdm
import concurrent
import concurrent.futures
import multiprocessing

import AutoAnnotation.src.utils.sentinel as sentinel
import AutoAnnotation.src.utils.draw as draw
import AutoAnnotation.src.utils.lidar as lidar

dataPath = "../../data"
scanFolder = "pointclouds"
scanField = "pointclouds"
annotationFolder = "temporal/default"
annotationField = "extrapolated"
outPath = "../../visualisation/pc"

class Visualise:
    def __init__(self, path, folder, filename, queue, outPath, scanFolder, scanField, annotationFolder, annotationField):

        self.path = path
        self.folder = folder
        self.filename = filename
        self.queue = queue
        self.outPath = outPath
        self.scanFolder = scanFolder
        self.scanField = scanField
        self.annotationFolder = annotationFolder
        self.annotationField = annotationField

        self.fileCounter = 0
        self.data = {}

    def run(self):
        
        self.queue.put("%s: Process spawned" % (self.filename))

        with open(os.path.join(self.path, self.annotationFolder, self.folder, self.filename), "rb") as f:
            self.data.update(pickle.load(f))
        try:
            with open(os.path.join(self.path, self.scanFolder, self.folder, self.filename), "rb") as f:
                self.data.update(pickle.load(f))
        except:
            for i in range(len(self.data[self.annotationField])):
                self.queue.put(sentinel.PROBLEM)

        for frameIdx in range(len(self.data[self.scanField])):
            self.drawFrame(frameIdx)
            self.queue.put(sentinel.SUCCESS)
        self.queue.put("%s: Process complete" % (self.filename))

    def drawFrame(self, idx):

        scan = self.data[self.scanField][idx]
        scan = lidar.combineScans(scan)

        fn = os.path.join(self.outPath, self.filename + "-" + str(idx) + ".png")
        os.makedirs(os.path.join(self.outPath, self.folder), exist_ok=True)
        
        draw.drawImgFromPoints(fn, scan, [], [], self.data[self.annotationField][idx], [], 3, False)

def listener(q, total):
    pbar = tqdm.tqdm(total=total)
    for item in iter(q.get, sentinel.EXIT):
        if item == sentinel.SUCCESS or item == sentinel.PROBLEM:
            pbar.update()
        else:
            tqdm.tqdm.write(str(item))

if __name__ == "__main__":
    
    count = 0
    with open(os.path.join(dataPath, "meta", "statistics.pkl"), "rb") as f:
        data = pickle.load(f)
    for i in data:
        count += i[-1]

    manager = multiprocessing.Manager()
    queue = manager.Queue()
    listenProcess = multiprocessing.Process(target=listener, args=(queue, count))
    listenProcess.start()

    try:
        shutil.rmtree(outPath)
    except:
        pass
    os.makedirs(outPath, exist_ok=True)

    jobs = []
    for files in os.walk(os.path.join(dataPath, scanFolder)):
        for filename in files[2]:
            path = dataPath
            folder = files[0][len(path)+len(scanFolder)+2:]
            jobs.append(Visualise(path, folder, filename, queue, outPath, scanFolder, scanField, annotationFolder, annotationField))

    jobs = jobs[:1]
    workers = 1
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
                queue.put("Process Exception: " + str(e))

    queue.put(sentinel.EXIT)
    listenProcess.join()
