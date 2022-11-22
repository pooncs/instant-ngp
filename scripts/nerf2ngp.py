import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def nerf_to_ngp(xf):
    mat = np.copy(xf)
    mat = mat[:-1,:]
    mat[:,1] *= -1 # flip axis
    mat[:,2] *= -1
    mat[:,3] *= 0.33 #scale
    mat[:,3] += [0.5, 0.5, 0.5] #offset
    
    mat = mat[[1,2,0], :] # swap axis
    
    rm = R.from_matrix(mat[:,:3]) 
    
    # quaternion (x, y, z, w) and translation
    return rm.as_quat(), mat[:,3] + 0.025

def smooth_camera_path(path_to_transforms, ):
    out = {"path":[], "time":1.0}
    with open(path_to_transforms + '/bct.json') as f:
        data = json.load(f)
    
    n_frames = len(data['frames'])
    
    xforms = {}
    for i in range(n_frames):
        file = int(data['frames'][i]['file_path'].split('/')[-1][:-4])
        xform = data['frames'][i]['transform_matrix']
        xforms[file] = xform
        
    xforms = dict(sorted(xforms.items()))
    
    # linearly take 12 transformation from transfroms.json
    for ind in np.linspace(1, n_frames, 12, endpoint=True, dtype=int):
        q, t = nerf_to_ngp(np.array(xforms[ind]))
        
        out['path'].append({
            "R": list(q),
            "T": list(t),
            "dof":0.0,
            "fov":7.75,
            "scale":0,
            "slice":0.0
        })
        
    with open(path_to_transforms+'/bct2.json', "w") as outfile:
        json.dump(out, outfile, indent=2)
        
smooth_camera_path('/home/ubuntu/ws/data/nerf/shuttle_ll')