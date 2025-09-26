import numpy as np
import plyfile
import skimage.measure
import torch
import point_cloud_utils as pcu

def create_mesh(mesh, filename, N=512, level_set=0.0):    
    ply_filename = filename

    # the voxel_origin is the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    cube = create_cube(N)

    vm = mesh.vertices
    fm = mesh.faces
    
    sdf_values, _, _ = pcu.signed_distance_to_mesh(cube.numpy(), vm.astype(np.float32), fm.astype(np.int32))
    # for occupancy instead of SDF, subtract 0.5 so the surface boundary becomes 0
    sdf_values = sdf_values.reshape(N, N, N) 

    #print("inference time: {}".format(time.time() - start_time))

    convert_sdf_samples_to_ply(
        sdf_values,
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        level_set
    )


def create_cube(N):
    # 定义坐标范围 [-1, 1]
    coords = torch.linspace(-1, 1, N)

    # 生成 3D 网格 (x,y,z)
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing="ij")

    # 拼接成 (N^3, 3)
    samples = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

    return samples  # shape: (N^3, 3)

def convert_sdf_samples_to_ply(
    pytorch_3d_sdf,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    level_set=0.0
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf = pytorch_3d_sdf

    # use marching_cubes_lewiner or marching_cubes depending on pytorch version 
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf, level=level_set, spacing=[voxel_size] * 3
        )
    except Exception as e:
        print("skipping {}; error: {}".format(ply_filename_out, e))
        return

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)