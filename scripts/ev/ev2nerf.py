"""
This script converts EVIW image metadata to NeRF expected orientation/scale in transforms.json file
Note: This script is the modified version of Nvidia Instant-ngp colmap2nerf.py script
Link: https://github.com/NVlabs/instant-ngp/blob/master/scripts/colmap2nerf.py

Reads from a folder including two subfolders:
 1) ./images/    ==> includes all images of imageset in jpg format
 2) ./metadata/  ==> includes all metadata of images in separate json files 

Sample Run Script: 
   python ev2nerf.py --imageset ./data/houston --aabb_scale 2 --scale 1

@author: semih.dinc (10/11/2022)
"""
import argparse
import json
import os
import math
import numpy as np
import shutil
import cv2
import pymap3d
from scipy.spatial.transform import Rotation as SR
from tqdm import tqdm

from util.instant_nerf_utils import *

from util.slippy_utils import tileEdges
from util.slippy_utils import lla_to_ecef


def ev2nerf(datasetFolder, AABB_SCALE, COORD_SYSTEM, SCALE, SLIPPY_TILE=[]):
  
   #Load all json files in the given directory into a list
   metadataFolder = datasetFolder + "/metadata/"
   json_files = [pos_json for pos_json in os.listdir(metadataFolder) if pos_json.endswith('.json')]

   #transforms.json file format
   out = {
      "aabb_scale": AABB_SCALE,
      "scale": SCALE,
      "frames": []
   }

   #read all metadata files in the datasetFolder and create a single transforms.json file
   up = np.zeros(3)
   
   #max and min coordinates of the scene in render cube
   max_scene = np.zeros(3) + np.iinfo(int).min
   min_scene = np.zeros(3) + np.iinfo(int).max
  
   #if there is slippy tile input calculate tile corners in lat/lon
   if SLIPPY_TILE != []:
      x,y,z = [int(i) for i in SLIPPY_TILE] #make sure, SLIPPY TILE is entered if enu is selected.
      lat2, lon1, lat1, lon2 = tileEdges(x,y,z) #lat lon coordinates of tile corners

      if COORD_SYSTEM == 'ecef':
         #ecef coordinates of tile corners
         print("- Keeping scene coordinates in ECEF for only input tile...")
         p1 = lla_to_ecef(lat1,lon1,0) 
         p2 = lla_to_ecef(lat1,lon2,0)
         p3 = lla_to_ecef(lat2,lon1,0)
         p4 = lla_to_ecef(lat2,lon2,0)
         
         #set max/min coordinates of scene based on input tile   
         max_scene, min_scene = calculateMinMaxAABB(p1,p2,p3,p4)
      elif COORD_SYSTEM == 'enu':
         print("- Keeping scene coordinates in ENU for only input tile...")
         #lat lon centers for geo->ENU
         lat0, lon0 = np.mean([lat2, lat1]), np.mean([lon2, lon1])
         rot_ECEF2ENUV = ecef2enuMat(lat0, lon0)
         
         #enu coordinates of the tile corners
         p1 = pymap3d.geodetic2enu(lat1, lon1, 0, lat0, lon0, 0)
         p2 = pymap3d.geodetic2enu(lat1, lon2, 0, lat0, lon0, 0)
         p3 = pymap3d.geodetic2enu(lat2, lon1, 0, lat0, lon0, 0)
         p4 = pymap3d.geodetic2enu(lat2, lon2, 0, lat0, lon0, 0)
         
         #set max/min coordinates of scene based on input tile   
         max_scene, min_scene = calculateMinMaxAABB(p1,p2,p3,p4)
         # min_scene[2] = -200
         # max_scene[2] = -100
      
   else:
      if COORD_SYSTEM == 'ecef':
         print("- Keeping whole scene coordinates in ECEF")
      elif COORD_SYSTEM == 'enu':
         print("- Keeping whole scene coordinates in ENU")
         p = getCorner(metadataFolder + json_files[0])
         lat0, lon0, _ = pymap3d.ecef2geodetic(p[0],p[1],p[2])
         rot_ECEF2ENUV = ecef2enuMat(lat0, lon0)

      else:
         print("- Keeping whole scene coordinates in colmap2nerf scaled space...")

   print('- Generating the transforms.json file...')
   #Main the for loop to process all images in the imageset
   for fileName in tqdm(sorted(json_files)): #sorted json files to give a little more predictability
      
      with open(metadataFolder + fileName) as jsonFile:
         image_metadata = json.load(jsonFile)
      
      #read the image index (ID)
      ind = fileName.split(".")[0].split("_")[2]
      name = datasetFolder + "/images/image_" + ind + ".jpg"
      b = sharpness(name)
      #b = 150
          
      intrinsics = image_metadata['interior_orientation']['camera_intrinsics']

      #focal length and principal point
      fl = intrinsics['pinhole']['focal_length_pixels']
      cx = intrinsics['pinhole']['principal_point_x']
      cy = intrinsics['pinhole']['principal_point_y']

      #distortion parameters
      k1 = intrinsics['browns']['k1']
      k2 = intrinsics['browns']['k2']
      k3 = intrinsics['browns']['k3']

      p1 = intrinsics['browns']['p1']
      p2 = intrinsics['browns']['p2']

      #image width and height
      dimensions = image_metadata['dimensions']['dimensions2d']
      w = dimensions['width']
      h = dimensions['height']

      angle_x = math.atan(w / (fl * 2)) * 2
      angle_y = math.atan(h / (fl * 2)) * 2

      fovx = angle_x * 180 / math.pi
      fovy = angle_y * 180 / math.pi
   
      #Calculate image transformation matrix from extrinsic parameters
      tvec_json = image_metadata["exterior_orientation"]["trajectory_poly"]["center"]
      rvec_json = image_metadata["exterior_orientation"]["trajectory_poly"]["rotation"]

      #Translation (in ECEF format) and Rotation (in Radians) Parameters
      t = np.array([tvec_json["x_p0"],tvec_json["y_p0"],tvec_json["z_p0"]]).reshape([3,1])
      r = np.array([rvec_json["x_p0"],rvec_json["y_p0"],rvec_json["z_p0"]])

      #conversion ECEF coordinates to camera-based space
      r[0] = -1 * r[0]
      r[1] = np.pi - r[1]
      r[2] = np.pi + r[2]

      #Pw = c2w*Pc + C ,where R is rotation matrix in XYZ order and C is camera center
      cam2world_Rot33 = SR.from_euler('XYZ',r).as_matrix()

      #get current frame coordinates in ECEF
      tl, tr, br, bl = getFrameCorners(image_metadata) 
      
      #If coordinate system is ENU, then covert from ECEF to ENU  
      if COORD_SYSTEM == 'enu':
         tl = pymap3d.ecef2enu(tl[0], tl[1], tl[2], lat0, lon0, 0)
         tr = pymap3d.ecef2enu(tr[0], tr[1], tr[2], lat0, lon0, 0)
         br = pymap3d.ecef2enu(br[0], br[1], br[2], lat0, lon0, 0)
         bl = pymap3d.ecef2enu(bl[0], bl[1], bl[2], lat0, lon0, 0)
                  
         t = pymap3d.ecef2enu(t[0], t[1], t[2], lat0, lon0, 0) # current frame pos from ECEF to ENU
         cam2world_Rot33 = np.dot(rot_ECEF2ENUV, cam2world_Rot33) # current frame orientation from ECEF to ENU
         
      #final 4x4 camera to world transformation matrix
      bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
      c2w = np.concatenate([np.concatenate([cam2world_Rot33, t], 1), bottom], 0)

      #we still keep an up vector in case coord system is scaled as in colmap2nerf.py
      up += c2w[0:3,3]

      #If tile input does not exist, keep overall min and max points of the scene from all frames
      if SLIPPY_TILE == []:
         max_p, min_p = calculateMinMaxAABB(tl, tr, br, bl)
         max_scene = [max([max_p[i],max_scene[i]]) for i in range(3)]
         min_scene = [min([min_p[i],min_scene[i]]) for i in range(3)]
      
      #Load image specific parameters into json
      frame = {}
      frame["camera_angle_x"] = angle_x
      frame["camera_angle_y"] = angle_y
      frame["fl_x"] = fl
      frame["fl_y"] = fl
      frame["k1"] = k1
      frame["k2"] = k2
      frame["p1"] = p1
      frame["p2"] = p2
      frame["cx"] = cx
      frame["cy"] = cy
      frame["w"] = w
      frame["h"] = h
      frame["file_path"] = "./images/image_" + ind + ".jpg"
      frame["sharpness"] = b
      frame["transform_matrix"] = c2w.tolist()
      
      #Add frame into json
      out["frames"].append(frame)
      # print("Image " + ind + " processed!")

   #add scene borders to transforms.json   
   out.update({'aabb': [[],[]]})
   out["aabb"][0] = min_scene
   out["aabb"][1] = max_scene
      
   #if we use original colmap2nerf coordinate system, ignore aabb and scale scene
   if COORD_SYSTEM == "":
      print("Scaling scene coordinates in [0,1] using Colmap2Nerf routine")
      scaleTransformations(out, up)
      del out['aabb']
      
   # out.update({'render_aabb': [[],[]]})
   # out["render_aabb"][0] = [0.1,0.5,0.1]
   # out["render_aabb"][1] = [0.9,0.9,0.9]

   #save transformations into transforms.json file
   with open(datasetFolder + "/transforms.json", "w") as outfile:
      json.dump(out, outfile, indent=2)

def parse_args():
   parser = argparse.ArgumentParser(description="convert EVIW metadata files for each image into nerf format transforms.json")
   parser.add_argument("--imageset", default="", help="input path to the EVIW images and metadata")
   parser.add_argument("--aabb_scale", default=1, choices=["1","2","4","8","16"], help="large scene scale factor. 1=scene fits in unit cube; power of 2 up to 16")
   parser.add_argument("--coord_system", default="", choices=["ecef", "enu"], help="Coordinate System to train and render the Nerf.")
   parser.add_argument("--scale", default=1, help="In normalized coordinate system, scale the scene. 1=scene is in original size.")
   parser.add_argument("--tile", default=[], nargs="*", help="In ECEF or ENU, set specific tile coordinates (x,y,z) to render. If empty, render whole scene.")
   args = parser.parse_args()
   return args

def main():
   args = parse_args()

   AABB_SCALE = int(args.aabb_scale)
   SCALE = int(args.scale)
   DATA_FOLDER = args.imageset
   SLIPPY_TILE = args.tile
   COORD_SYSTEM = args.coord_system

   # DATA_FOLDER = "/home/ubuntu/Documents/github/instant-ngp/data/nerf/shuttle"
   # AABB_SCALE = 2
   # COORD_SYSTEM = 'enu'
   # # SLIPPY_TILE = ["61824", "108529", "18"]
   # SLIPPY_TILE = ["30912", "54265", "17"]
   
   if DATA_FOLDER == "":
      print("Missing Input Folder. Please enter input directory path to EVIW imageset.")
   else:
      ev2nerf(DATA_FOLDER, AABB_SCALE, COORD_SYSTEM, SCALE, SLIPPY_TILE)
      print("transforms.json created in " + DATA_FOLDER + " directory")

if __name__ == "__main__":
    main()