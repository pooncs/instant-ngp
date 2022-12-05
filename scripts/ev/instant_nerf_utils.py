import cv2
from tkinter import W, image_types
import numpy as np
import math
import json

def ecef2enuMat(lat, lon):
   lat0r, lon0r = np.radians(lat), np.radians(lon)
   
   rot_ECEF2ENUV = np.array([
      [-math.sin(lon0r),                 math.cos(lon0r),                  0              ],
      [-math.sin(lat0r)*math.cos(lon0r), -math.sin(lat0r)*math.sin(lon0r), math.cos(lat0r)],
      [math.cos(lat0r)*math.cos(lon0r),  math.cos(lat0r)*math.sin(lon0r),  math.sin(lat0r)]])
   
   return rot_ECEF2ENUV

def getCorner(filePath):
   with open(filePath) as jf:
      image_metadata = json.load(jf)
   tl, tr, br, bl = getFrameCorners(image_metadata)
   return tl

def getFrameCorners(image_metadata):
   exterior_ring = image_metadata['ground_footprint']['polygons'][0]['exterior']['ring']
   # Frame border coordinates: top_left->top_right->bottom_right->bottom_left
   tl, tr, br, bl, _ = exterior_ring
   p1 = [tl['x'],tl['y'],tl['z']]
   p2 = [tr['x'],tr['y'],tr['z']]
   p3 = [br['x'],br['y'],br['z']]
   p4 = [bl['x'],bl['y'],bl['z']]
   
   return p1,p2,p3,p4   

def calculateMinMaxAABB(p1,p2,p3,p4):
   max_p = np.zeros(3,)
   max_p[0] = max([p1[0], p2[0], p3[0], p4[0]])
   max_p[1] = max([p1[1], p2[1], p3[1], p4[1]])
   max_p[2] = max([p1[2], p2[2], p3[2], p4[2]])

   min_p = np.zeros(3,)
   min_p[0] = min([p1[0], p2[0], p3[0], p4[0]])
   min_p[1] = min([p1[1], p2[1], p3[1], p4[1]])
   min_p[2] = min([p1[2], p2[2], p3[2], p4[2]])
   
   return max_p.tolist(), min_p.tolist()

#Calculate image sharpness parameter (See Instant-NeRF for details)
def sharpness(imagePath):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = cv2.Laplacian(gray, cv2.CV_64F).var()
	return fm

# returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel (See Instant-NeRF for details)
def closest_point_2_lines(oa, da, ob, db): 	
   da = da / np.linalg.norm(da)
   db = db / np.linalg.norm(db)
   c = np.cross(da, db)
   denom = np.linalg.norm(c)**2
   t = ob - oa
   ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
   tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
   if ta > 0:
      ta = 0
   if tb > 0:
      tb = 0
   return (oa+ta*da+ob+tb*db) * 0.5, denom

# finds rotation matrix between two vectors a and b (See Instant-NeRF for details)
def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

#scales the list of transformations into NeRF expected format (in the region of a unit cube)
def scaleTransformations(out, up):
   nframes = len(out["frames"])

   up = up / np.linalg.norm(up)
   print("up vector was", up)
   R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
   R = np.pad(R,[0,1])
   R[-1, -1] = 1

   for f in out["frames"]:
      f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

   # find a central point they are all looking at
   print("computing center of attention...")
   totw = 0.0
   totp = np.array([0.0, 0.0, 0.0])
   for f in out["frames"]:
      mf = f["transform_matrix"][0:3,:]
      for g in out["frames"]:
         mg = g["transform_matrix"][0:3,:]
         p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
         if w > 0.00001:
            totp += p*w
            totw += w
   if totw > 0.0:
      totp /= totw
   print(totp) # the cameras are looking at totp
   for f in out["frames"]:
      f["transform_matrix"][0:3,3] -= totp

   avglen = 0.
   for f in out["frames"]:
      avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
   avglen /= nframes
   print("avg camera distance from origin", avglen)

   for f in out["frames"]:
      f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

   for f in out["frames"]:
      f["transform_matrix"] = f["transform_matrix"].tolist()

   #sort frames
   out["frames"] = sorted(out["frames"], key=lambda d: d['file_path']) 
