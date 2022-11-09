import numpy as np

def combineScans(scans):
    
    if type(scans) == dict:
        combinedScans = []
        for key in scans.keys():
            scans[key] = np.array(scans[key])
            scans[key] = scans[key].reshape([scans[key].shape[0], 4])
            combinedScans.append(scans[key])

        return np.concatenate(combinedScans)

    if type(scans) == np.ndarray:
        return scans

