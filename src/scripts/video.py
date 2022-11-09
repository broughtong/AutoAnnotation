import os
import subprocess

frameRate = 20
path = "../../visualisation/pc"
outPath = "../detectorVideos"
os.makedirs(os.path.join(path, outPath), exist_ok=True)

bagNames = []
for files in os.walk(path):
    for filename in files[2]:
        filename = filename.split(".bag.pickle")[0] + ".bag.pickle"
        bagNames.append(filename)
bagNames = set(bagNames)

bashCommand = "ffmpeg -framerate %i -i %s.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -y %s.mp4"

for bag in bagNames:
    command = bashCommand % (frameRate, bag + "-%d", os.path.join(outPath, bag))
    print(command, flush=True)
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, cwd=path)
    process.wait()
