import numpy as np

def createBIN(filename, pointcloud):

    #x, y, z, intensity, t, reflectivity, ring, ambient, range 
    #bin is x, y, z, reflect
    try:
        pointcloud = np.delete(pointcloud, 8, 1)
    except:
        pass
    try:
        pointcloud = np.delete(pointcloud, 7, 1)
    except:
        pass
    try:
        pointcloud = np.delete(pointcloud, 6, 1)
    except:
        pass
    try:
        pointcloud = np.delete(pointcloud, 4, 1)
    except:
        pass
    try:
        pointcloud = np.delete(pointcloud, 3, 1)
    except:
        pass
    pointcloud = pointcloud.astype("float32")
    np.save(filename + ".bin", pointcloud)

def createNPY(filename, pointcloud):
    
    #x, y, z, intensity, t, reflectivity, ring, ambient, range 
    #bin is x, y, z, reflect
    try:
        pointcloud = np.delete(pointcloud, 8, 1)
    except:
        pass
    try:
        pointcloud = np.delete(pointcloud, 7, 1)
    except:
        pass
    try:
        pointcloud = np.delete(pointcloud, 6, 1)
    except:
        pass
    try:
        pointcloud = np.delete(pointcloud, 4, 1)
    except:
        pass
    try:
        pointcloud = np.delete(pointcloud, 3, 1)
    except:
        pass
    pointcloud = pointcloud.astype("float32")
    np.save(filename + ".npy", pointcloud)

def createPLY(filename, pointcloud):
    
    header = """ply
    format ascii 1.0
    element vertex %i
    property float x
    property float y
    property float z
    end_header
    """
    header = header.replace("\t", "")
    header = header.replace("    ", "")
    
    with open(filename + ".ply", "w") as f:
        f.write(header % (len(pointcloud)))
        for p in pointcloud:
            f.write("%f %f %f\n" % (p[0], p[1], p[2]))
