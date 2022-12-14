import os
import json
import numpy as np
import imageio.v2 as imageio
from glob import glob
from tqdm import tqdm

# Extract depth images from DSNeRF results -------------------------------------------------------
depth_dir = "/home/ubuntu/ws/DSNeRF/logs/release/evTest3/testset_100000"
base_dir = "/home/ubuntu/ws/data/nerf/shuttleTest8png"
json_dir = os.path.join(base_dir, "jsons/transforms.json")
out_json_dir = os.path.join(base_dir, "transforms_depth.json")
depth_paths = sorted(glob(depth_dir + "/*.npz"))
depth_save_dir = os.path.join(base_dir, "depths")
os.makedirs(depth_save_dir, exist_ok=True)

# From colmap database in dsnerf routine
names = [3,5,7,102,103,107,108,109,115,116,12,120,121,125,129,13,130,131,134,135,142,143,144,145,146,147,150,151,159,160,161,165,166,167,175,176,177,181,186,187,192,193,195,20,200,201,206,207,209,21,212,213,216,22,222,223,226,227,23,232,233,236,237,240,243,245,30,31,32,33,38,39,41,45,48,52,53,59,60,61,66,67,68,74,75,76,77,84,85,86,87,88,89,92,93,95,98,99]
depths = []
print("Load depths maps from DSNeRF")
for depth_path in tqdm(depth_paths):
    _depth = np.load(depth_path)["depth"]
    _depth -= np.min(_depth)
    _depth /= np.max(_depth)
    depths.append(_depth)

depths = np.stack(depths)
#depths -= np.percentile(depths, 5)
#int_scale = 255/np.percentile(depths, 99)
int_scale = 1#/np.max(depths)
depths = np.uint8(depths*int_scale)

print(f"Save depth maps to: {depth_save_dir}")
for i, depth in enumerate(tqdm(depths)):
    pngUri = os.path.join(depth_save_dir, f"image_{str(names[i]).zfill(2)}.png")
    imageio.imsave(pngUri, depth)

# Write images and add it into transforms.json --------------------------------------------------
with open(json_dir) as fp:
    meta = json.load(fp)

meta["enable_depth_loading"] = True
meta["integer_depth_scale"] = int_scale

print(f"Write to: {out_json_dir}")
for frame in meta["frames"]:
    outpath = os.path.join("./depths", os.path.basename(frame["file_path"]))
    frame["depth_path"] = outpath

with open(out_json_dir, "w") as fp:
    json.dump(meta, fp, indent=2)
    
