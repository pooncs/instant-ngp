"""
This script scales EVIW images with existing instant-ngp style transforms.json
and output the scaled images and the updated transforms.json to a new directory
Link: https://github.com/NVlabs/instant-ngp

Reads from a folder including two subfolders:
 1) ./images/    ==> includes all images of imageset in jpg format
 2) ./transforms.json/  ==> includes instant-ngp style transforms.json

Args:
--data_dir: input path to the EVIW images and transform.json
--save_dir: Folder to place the scaled images and transform.json
--scale: Scaling factor (float or fractions)

Sample Run Script: 
   python strategic-rnd-playground/nerf-experiments/util/transformScaler.py --data_dir /Users/sam.poon/Documents/bin/test/nerf-experiments/data/toyota --save_dir /Users/sam.poon/Documents/bin/test/nerf-experiments/data/toyota3 --scale 1/8

@author: sam.poon (10/6/2022)
"""

import json 
import os
import cv2
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Scale EVIW images with nerf format transforms.json to new json")
    parser.add_argument("--data_dir", default="", help="input path to the EVIW images and transform.json")
    parser.add_argument("--save_dir", default="", help="Folder to place the scaled images and transform.json")
    parser.add_argument("--scale", default=1, help="Scaling factor")
    return parser.parse_args()

def scaleImages(data_dir, save_dir, scale):
    with open(os.path.join(data_dir,"transforms.json")) as jsonFile:
        metadata = json.load(jsonFile)

    print(f'Scaling images by {scale}')
    print(f"Loading from: {data_dir}")
    out = metadata
    for i, frame in tqdm(enumerate(metadata['frames']), total=len(metadata['frames'])):
        out['frames'][i]['fl_x'] = frame['fl_x']*scale
        out['frames'][i]['fl_y'] = frame['fl_y']*scale
        out['frames'][i]['cx'] = frame['cx']*scale
        out['frames'][i]['cy'] = frame['cy']*scale
        out['frames'][i]['w'] = frame['w']//(1/scale)
        out['frames'][i]['h'] = frame['h']//(1/scale)
        nw = int(out['frames'][i]['w'])
        nh = int(out['frames'][i]['h'])
        
        relpath = os.path.relpath(frame['file_path'])
        im = cv2.imread(os.path.join(data_dir, relpath))
        if scale > 1:
            im = cv2.resize(im, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            im = cv2.resize(im, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Calculates sharpness for psnr calculation
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        out['frames'][i]['sharpness'] = fm
        savepath = os.path.join(save_dir, relpath)

        if not os.path.exists(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath))
            print('Folder not found, creating new dir: ' + os.path.dirname(savepath))
        
        cv2.imwrite(savepath, im)
        #print(f'Scaling to {nw}x{nh}, saving to {os.path.relpath(savepath)}, sharpness: {fm}')

    with open(os.path.join(save_dir, "transforms.json"), "w") as outfile:
        print("Writing to " + os.path.join(save_dir, "transforms.json"))
        json.dump(out, outfile, indent=2)

if __name__ == "__main__":
    args = parse_args()
    scaleImages(args.data_dir, args.save_dir, eval(args.scale))