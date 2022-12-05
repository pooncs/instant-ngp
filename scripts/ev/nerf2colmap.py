import os
import json
import numpy as np
import sys
from scipy.spatial.transform import Rotation as SR
from databaseRead import getImageName

def do_system(arg):
	print(f"==== running: {arg}")
	err = os.system(arg)
	if err:
		print("FATAL: command failed")
		sys.exit(err)

out_align, out_images, out_cameras = [], [], []
datasetFolder = "/home/ubuntu/ws/data/nerf/shuttleTest8"
outputFolder = "/home/ubuntu/ws/data/nerf/shuttleTest8/sparse/2"
databasePath = os.path.join(datasetFolder, "colmap.db")
imagePath = os.path.join(datasetFolder, "images")

do_system(f"colmap feature_extractor --database_path {databasePath} --image_path {imagePath}")
imname = getImageName(databasePath)

with open(os.path.join(datasetFolder, "transforms.json")) as jsonFile:
    meta = json.load(jsonFile)

nerf2colmap = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

frames = meta['frames']

for i, frame in enumerate(frames):
    c2w = np.matmul(frame['transform_matrix'], nerf2colmap)
    r = SR.from_matrix(np.array(c2w[:3, :3]))
    qvec = r.as_quat()
    tvec = np.array(c2w[:3, -1])
    k1, k2, p1, p2 = frame['k1'], frame['k2'], frame['p1'], frame['p2']
    w, h, fl_x, fl_y = frame['w'], frame['h'], frame['fl_x'], frame['fl_y']
    cx, cy = frame['cx'], frame['cy']
    
    im_name = os.path.basename(frame['file_path'])
    im_ind = int(im_name.split("_")[1].split(".")[0])

    out_cameras.append(f"{i} OPENCV {w} {h} {fl_x} {fl_y} {cx} {cy} {k1} {k2} {p1} {p2} ")
    
    out_images.append(f"{i} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]} {i} image_{im_ind}.jpg")
    print(f"{im_ind} {imname[i]}")
    #out_align.append(f"image_{im_ind}.jpg {lat} {lon} {alt}")

os.makedirs(outputFolder, exist_ok=True)
with open(f'{outputFolder}/cameras.txt', 'w') as f:
    for line in out_cameras:
        f.write(line)
        f.write('\n')

with open(f'{outputFolder}/images.txt', 'w') as f:
    for line in out_images:
        f.write(line)
        f.write('\n')
        f.write('\n')

with open(f'{outputFolder}/points3D.txt', 'w') as f:
    f.write("")

do_system(f"colmap exhaustive_matcher --database_path {databasePath}")
do_system(f"colmap point_triangulator --database_path {databasePath} --image_path {imagePath} --input_path {outputFolder} --output_path {outputFolder}")