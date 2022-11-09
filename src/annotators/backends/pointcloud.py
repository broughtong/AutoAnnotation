import os
import numpy as np

import AutoAnnotation.src.utils.geometry as geometry
import AutoAnnotation.src.utils.pointcloud as pointcloud

class AnnotatorBackend():

    def __init__(self, path, scanField, annotationFolder, annotationField):

        self.path = path
        self.scanField = scanField
        self.annotationFolder = annotationFolder.replace("/", "_")
        self.annotationField = annotationField

        self.folderName = os.path.join(path, scanField, self.annotationFolder + "_" + annotationField)

        os.makedirs(os.path.join(self.folderName, "pointcloud-bin", "all", "cloud"), exist_ok=True)
        os.makedirs(os.path.join(self.folderName, "pointcloud-bin", "all", "annotations"), exist_ok=True)
        os.makedirs(os.path.join(self.folderName, "pointcloud-bin", "all", "debug"), exist_ok=True)

        os.makedirs(os.path.join(self.folderName, "pointcloud-npy", "all", "cloud"), exist_ok=True)
        os.makedirs(os.path.join(self.folderName, "pointcloud-npy", "all", "annotations"), exist_ok=True)
        os.makedirs(os.path.join(self.folderName, "pointcloud-npy", "all", "debug"), exist_ok=True)

        os.makedirs(os.path.join(self.folderName, "pointcloud-ply", "all", "cloud"), exist_ok=True)
        os.makedirs(os.path.join(self.folderName, "pointcloud-ply", "all", "annotations"), exist_ok=True)
        os.makedirs(os.path.join(self.folderName, "pointcloud-ply", "all", "debug"), exist_ok=True)

    def annotate(self, filename, pc, annotations):
        
        #save raw pointclouds
        fn = os.path.join(self.folderName, "pointcloud-bin", "all", "cloud", filename)
        pointcloud.createBIN(fn, pc)

        fn = os.path.join(self.folderName, "pointcloud-npy", "all", "cloud", filename)
        pointcloud.createNPY(fn, pc)
        
        fn = os.path.join(self.folderName, "pointcloud-ply", "all", "cloud", filename)
        pointcloud.createPLY(fn, pc)

        #save annotations
        fn = os.path.join(self.folderName, "pointcloud-bin", "all", "annotations", filename + ".txt")
        self.generateAnnotations(fn, annotations)

        fn = os.path.join(self.folderName, "pointcloud-npy", "all", "annotations", filename + ".txt")
        self.generateAnnotations(fn, annotations)

        fn = os.path.join(self.folderName, "pointcloud-ply", "all", "annotations", filename + ".txt")
        self.generateAnnotations(fn, annotations)

        #save debug files (crop of cars)
        for i, a in enumerate(annotations):

            cpc, _ = geometry.getInAnnotation(pc, [a]) 
            if len(cpc) == 0:
                return
            cpc = np.array(cpc)

            fn = os.path.join(self.folderName, "pointcloud-bin", "all", "debug", filename + "-c" + str(i))
            pointcloud.createBIN(fn, cpc)
            fn = os.path.join(self.folderName, "pointcloud-npy", "all", "debug", filename + "-c" + str(i))
            pointcloud.createNPY(fn, cpc)
            fn = os.path.join(self.folderName, "pointcloud-ply", "all", "debug", filename + "-c" + str(i))
            pointcloud.createPLY(fn, cpc)

    def generateAnnotations(self, fn, annotations):

        with open(fn, "w") as f:
            f.write("# format: [x y z dx dy dz heading_angle category_name]\n")
            dx = 5.3
            dy = 2.0 
            dz = 2.8
            height = (dz/2) - 0.3
            className = "Car"
            for a in annotations:
                f.write("%f %f %f %f %f %f %f %s\n" % (a[0], a[1], height, dx, dy, dz, a[2], className))
