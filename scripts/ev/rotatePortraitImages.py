"""
This script rotates all portrait images to landscape in the given imageset. 
Updates results in "transforms.json" file in data_dir

Reads from a folder including:
 1) ./images/    ==> includes all images of imageset in jpg format
 2) ./transforms.json  ==> transformations for each frame in the image set

Args:
--data_dir: input path to the EVIW images

Sample Run Script: 
   python rotatePortraitImages.py --data_dir /data/toyota --save_dir /data/toyota_landscape

@author: semih dinc (11/10/2022)
"""

import json 
import os
import cv2
import argparse
from scipy.spatial.transform import Rotation as SR
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Rotate all portrait images to landscape")
    parser.add_argument("--data_dir", default="", help="Input path to the EVIW images and transform.json")
    parser.add_argument("--save_dir", default="", help="Folder to save rotated images and transform.json")
    return parser.parse_args()
 
def rotateImages(data_dir, save_dir):
   with open(os.path.join(data_dir,"transforms.json")) as jsonFile:
      metadata = json.load(jsonFile)

   out = metadata
   print("Rotating portrait images...")
   print(f"Loading from: {data_dir}")
   for i, frame in tqdm(enumerate(metadata['frames']), total=len(metadata['frames'])):
      
      relpath = os.path.relpath(frame['file_path'])
      im = cv2.imread(os.path.join(data_dir, relpath))

      savepath = os.path.join(save_dir, relpath)
      if not os.path.exists(os.path.dirname(savepath)):
         os.makedirs(os.path.dirname(savepath))
         print('Folder not found, creating new dir: ' + os.path.dirname(savepath))

      w, h = int(frame['w']), int(frame['h'])
      
      if h > w:
         #swap values
         frame['cx'], frame['cy'] = frame['cy'], frame['cx']
         frame["camera_angle_x"], frame["camera_angle_y"] = frame["camera_angle_y"], frame["camera_angle_x"]
         frame['w'], frame['h'] = h, w
 
         #rotate the pose of the camera
         rotMat = SR.from_euler('Z',np.pi/2).as_matrix()
         c2w = np.array(frame["transform_matrix"])
         c2w[0:3,0:3] = c2w[0:3,0:3] @ rotMat
         frame["transform_matrix"] = c2w.tolist()

         #rotate the image
         im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)

         # Calculates sharpness for psnr calculation
         gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
         fm = cv2.Laplacian(gray, cv2.CV_64F).var()
         frame['sharpness'] = fm
      
      #update transforms.json frame
      out["frames"][i] = frame
      
      #write image into save path
      cv2.imwrite(savepath, im)
      #print(f'Image {i} saved to {os.path.relpath(savepath)}')
   
   #write transforms into transforms.json   
   with open(os.path.join(save_dir, "transforms.json"), "w") as outfile:
      print("Writing to " + os.path.join(save_dir, "transforms.json"))
      json.dump(out, outfile, indent=2)

if __name__ == "__main__":
    args = parse_args()
    
    #args.data_dir = "/home/ubuntu/data/shuttle_0_10"
    #args.save_dir = "/home/ubuntu/data/shuttle_landscape_0_10"
    
    rotateImages(args.data_dir, args.save_dir)
