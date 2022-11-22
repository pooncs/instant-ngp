from utils import nerf_to_colmap
import json


data_dir = "/home/ubuntu/ws/data/nerf/shuttle4"
with open(os.path.join(data_dir, "transforms.json")) as jsonfile:
    meta = json.load(jsonfile)

for frame in meta['frames']:
    pose_nerf = frame['transform_matrix']
    pose_colmap = nerf_to_colmap(pose_nerf, )
    

