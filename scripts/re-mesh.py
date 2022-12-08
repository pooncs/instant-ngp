"""
python3 re-mesh.py base.obj transforms.json export.obj
"""
import numpy as np
import open3d as o3d
from open3d import *
import copy
from datetime import datetime
import json
import sys

resolution_cloud = 250000
cam_size = 500
mesh_it_depth = 8

start = datetime.now()
def mesh_it(mesh):
    print("Meshing ...")
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            mesh, depth=mesh_it_depth)
    vertices_to_remove = densities < np.quantile(densities, 0.001)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return(mesh)
    
def export_it(name, mesh):
    print("Exporting mesh ...")
    o3d.io.write_triangle_mesh(name,
                               mesh,
                               write_triangle_uvs=True)
    print(mesh)
    print("exported " + name)

def remove_floaters(mesh):
    print("Removing floaters in mesh ...")
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    
    mesh_0 = copy.deepcopy(mesh)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100000
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0

def make_point_cloud(mesh):
    print("Make mesh into points ...")
    mesh.compute_vertex_normals()
    mesh_0 = remove_floaters(mesh)
    mesh = mesh_0.sample_points_poisson_disk(resolution_cloud)
    return mesh

def camera_view_mesh(mesh, type):
    print("Points visible from cam locations ...")
    jsonFile = open(sys.argv[2])
    data = json.load(jsonFile)
    frameInt = 0
    diameter = np.linalg.norm(
        np.asarray(mesh.get_max_bound()) - np.asarray(mesh.get_min_bound()))
    radius = diameter * cam_size
    pcd_combined = o3d.geometry.PointCloud()
    cameras = []
    for frame in data['frames']:
        x = data['frames'][frameInt]['transform_matrix'][0][3]
        y = data['frames'][frameInt]['transform_matrix'][1][3]
        z = data['frames'][frameInt]['transform_matrix'][2][3]
        frameInt = frameInt+1
        camera = [y,z,x]
        _, pt_map = mesh.hidden_point_removal(camera, radius)
        mesh_view = mesh.select_by_index(pt_map)
        pcd_combined += mesh_view
        cameras.append(camera)
        #o3d.visualization.draw_geometries([mesh_view])
    jsonFile.close()
    if type == "mesh":
        return pcd_combined
    else:
        return cameras

print("INPUT MESH: " + sys.argv[1])
print("CAM LOCATIONS: " + sys.argv[2])
print("OUTPUT MESH: " + sys.argv[3])

print("Open file ...")
mesh = o3d.io.read_triangle_mesh(sys.argv[1])
clean = remove_floaters(mesh)
cloud = make_point_cloud(clean)
camMesh = camera_view_mesh(cloud, "mesh")
cameras = camera_view_mesh(cloud, "cameras")
reMesh = mesh_it(camMesh)
finalMesh = remove_floaters(reMesh)

camColors = [[1, 0, 0] for i in range(len(cameras))]
camLocations = o3d.geometry.PointCloud()
camLocations.points = o3d.utility.Vector3dVector(cameras)
camLocations.colors = o3d.utility.Vector3dVector(camColors)

end = datetime.now()
print("Stats: ")
print(finalMesh)
time_start = start.strftime("%H:%M:%S")
time_end = end.strftime("%H:%M:%S")
print("Start time: ", time_start)
print("Finish time: ", time_end)
export_it(sys.argv[3], finalMesh)
o3d.visualization.draw_geometries([finalMesh, camLocations])

