import numpy as np
from scipy.spatial.transform import Rotation
import os
import json

def sphericalPoses(p0,numberOfFrames):
	"""
	We first move the camera to [0,0,tz] in the world coordinate space. 
	Then we rotate the camera pos 45 degrees wrt X axis.
	Finally we rotate the camera wrt Z axis numberOfFrames times.
	Note: Camera space and world space (ENU) is actually aligned 
		X_c == X_w or E (east)
		Y_c == Y_w or N (north)
		Z_c == Z_w or U (up)
		Camera is positioned at [0,0,tz] it is actually looking down to -Z direction 
	"""
	transMat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,1750],[0,0,0,1]]).astype(float) #move camera to 0,0,1500
	
	#rotate camera 45 degrees wrt X axis
	rotMatX = np.identity(4)
	rotMatX[0:3,0:3] = Rotation.from_euler('X',np.pi/4).as_matrix()
	
	#first translate then rotate
	transMat = rotMatX @ transMat
	
	poses = []
	for angle in np.linspace(0,2*np.pi,numberOfFrames):

		rotMatZ = np.identity(4)
		rotMatZ[0:3,0:3] = Rotation.from_euler('Z',angle).as_matrix()

		myPose = rotMatZ @ transMat
		poses.append(myPose)

	poses = np.stack(poses, axis=0)
	return poses

def generateSphericalTestPoses(data_dir: str, numberOfFrames: int):
	with open(os.path.join(data_dir, 'transforms.json'), 'r') as fp:
		meta = json.load(fp)

	frame = meta["frames"][15]
	fname = os.path.join(data_dir, frame['file_path'][2:])

	factor = 1
	#image intrinsics
	focal, cx, cy = frame['fl_x'], frame['cx'], frame['cy']
	K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
	K[:2, :] /= factor
	ax = frame['camera_angle_x']
	ay = frame['camera_angle_y']

	w, h = frame['w']/factor, frame['h']/factor

	c2w = np.array(frame["transform_matrix"])
	camtoworlds = sphericalPoses(c2w, numberOfFrames)

	return camtoworlds, K, ax, ay

def nerf_to_ngp(xf, scale):
    mat = np.copy(xf)
    mat = mat[:-1,:]
    mat[:,1] *= -1 # flip axis
    mat[:,2] *= -1
    mat[:,3] *= scale #scale
    mat[:,3] -= [0.5, 0.5, 0.5] #offset
	
    mat = mat[[1,2,0], :] # swap axis
    
    rm = Rotation.from_matrix(mat[:,:3]) 
    
    # quaternion (x, y, z, w) and translation
    return rm.as_quat(), mat[:,3] + 0.025

def smooth_camera_path(c2w, angle_x):
    out = {"path":[], "time":1.0}
    
    for x in c2w:
        fov = (angle_x * 180 / np.pi)/2
        q, t = nerf_to_ngp(x)
        
        out['path'].append({
            "R": list(q),
            "T": list(t),
            "aperture_size":0.0,
            "fov":fov,
			"glow_mode":0,
			"glow_y_cutoff":0.0,
            "scale":1,
            "slice":0.0
        })
    return out

if __name__=='__main__':
    data_dir = "/home/ubuntu/ws/data/nerf/shuttle_ll"
    c2w, _, angle_x, _ = generateSphericalTestPoses(data_dir, 12)
    out = smooth_camera_path(c2w, angle_x)

    with open(os.path.join(data_dir, 'base_cam.json'), 'w') as outfile:
        json.dump(out, outfile, indent=2)