import trimesh
import numpy as np
import os
import json
from pathlib import Path

def scale(mesh, target_max=0.5, target_min=-0.5, axis=0):
    current_min = mesh.bounds[0][axis]
    current_max = mesh.bounds[1][axis]
    current_span = current_max - current_min

    # 计算缩放比例（保持其他轴比例）
    target_span = target_max - target_min
    scale_factor = target_span / current_span

    # 构建缩放矩阵（仅缩放目标轴）
    scale_matrix = np.eye(4)
    scale_matrix[axis, axis] = scale_factor

    # 应用缩放
    mesh.apply_transform(scale_matrix)

    # 平移到目标区间中心
    translated_center = (target_min + target_max) / 2
    current_center = mesh.bounds.mean(axis=0)[axis]
    translation = translated_center - current_center
    mesh.apply_translation([translation if i == axis else 0 for i in range(3)])
    return mesh

def calculate_sdf(mesh, point):
    # 计算点到网格的有符号距离
    sdf_value = trimesh.proximity.signed_distance(mesh, [point])
    return sdf_value

def spherical_shell_sampling(mesh, num_points=100000, layers=5):
    """
    生成多层球壳采样点
    :param layers: 球壳层数（内层->外层）
    """
    # 计算包围球参数
    radius = 1
    
    points = []
    for i in range(layers):
        # 当前层半径（从内到外）
        r = radius * (0.2 + 0.8 * i / (layers - 1))
        
        # 生成球面点（Fibonacci球算法）
        indices = np.arange(num_points // layers, dtype=np.float32)
        phi = np.arccos(1 - 2 * (indices + 0.5) / (num_points // layers))
        theta = np.pi * (1 + 5**0.5) * indices
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        # 平移至网格中心
        layer_points = np.column_stack([x, y, z])
        points.append(layer_points)
    
    return np.vstack(points)

def uniform_grid_samples(mesh, resolution=50):
    """在网格包围盒内生成均匀网格点"""
    bounds = mesh.bounds
    x = np.linspace(bounds[0][0]-0.1, bounds[1][0]+0.1, resolution)
    y = np.linspace(bounds[0][1]-0.1, bounds[1][1]+0.1, resolution)
    z = np.linspace(bounds[0][2]-0.1, bounds[1][2]+0.1, resolution)
    xx, yy, zz = np.meshgrid(x, y, z)
    return np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

# def surface_biased_samples(mesh, num_points=10000, std_dev=0.1):
#     """在表面法线方向生成高斯扰动点"""
#     surface_points = mesh.sample(num_points)
#     normals = mesh.vertex_normals[:num_points]  # 需预先计算法线
#     offsets = np.random.normal(scale=std_dev, size=(num_points, 3))
#     return surface_points + offsets * normals


def surface_biased_samples(mesh, num_points=10000, std_dev=0.1):
    """生成表面扰动采样点（自动处理顶点不足的情况）"""
    # 获取表面点（数量不超过顶点数）
    max_samples = len(mesh.vertices)
    actual_samples = min(num_points, max_samples)
    surface_points = mesh.sample(actual_samples)
    
    # 重复采样直到达到目标点数（可选）
    if actual_samples < num_points:
        repeat = num_points // actual_samples + 1
        surface_points = np.tile(surface_points, (repeat, 1))[:num_points]
    
    # 获取法线（需确保法线已计算）
    if not hasattr(mesh, 'vertex_normals') or len(mesh.vertex_normals) == 0:
        mesh.compute_vertex_normals()
    
    # 为每个采样点分配法线（通过最近顶点）
    _, vertex_ids = mesh.nearest.vertex(surface_points)
    normals = mesh.vertex_normals[vertex_ids]
    
    # 添加法线方向扰动
    offsets = np.random.normal(scale=std_dev, size=(num_points, 3))
    return surface_points + offsets * normals

def hybrid_sampling(mesh, grid_res=30, surface_points=20000, std_dev=0.2):
    """混合均匀网格和表面扰动采样"""
    uniform = uniform_grid_samples(mesh, grid_res)
    surface = surface_biased_samples(mesh, surface_points, std_dev)
    return np.vstack([uniform, surface])

def compute_sdf_trimesh(mesh, points):
    """
    使用trimesh计算点到网格的SDF（符号距离）
    :param mesh: trimesh.Trimesh 对象
    :param points: (N,3) 的查询点数组
    :return: (N,) 的SDF值数组（负值=内部）
    """
    # 检查网格是否封闭（水密性）
    if not mesh.is_watertight:
        mesh.fill_holes()  # 尝试自动修复
    
    if mesh.volume>0:
        flag = -1
    else:
        flag = 1
    # 计算符号距离
    sdf = trimesh.proximity.signed_distance(mesh, points)
    return sdf*flag

def compute_sdf_from_single_object(path, num_points=20000):
    """
    从单个对象文件计算SDF
    :param path: 对象文件路径
    :return: (N,3) 的采样点和对应的SDF值
    """
    mesh = trimesh.load(path, force='mesh', repair=True)
    mesh = scale(mesh)  # 缩放到[-0.5, 0.5]区间 
    #points = spherical_shell_sampling(mesh, num_points=num_points, layers=5)
    points = hybrid_sampling(mesh, grid_res=30, surface_points=num_points, std_dev=0.2)
    sdf_values = compute_sdf_trimesh(mesh, points)
    return points, sdf_values

class Manifold:
    def __init__(self, path_to_manifold):
        self.path_to_manifold = path_to_manifold
        self.manifold = os.path.join(path_to_manifold, 'manifold')
        self.simplify = os.path.join(path_to_manifold, 'simplify')
        self.manifold_temp = os.path.join(path_to_manifold, 'manifold_temp.obj')
    
    def stl_to_obj(self, stl_path, obj_path=None):
        """
        将STL文件转换为OBJ格式
        :param stl_path: 输入STL文件路径
        :param obj_path: 输出OBJ路径（默认原位生成同名文件）
        """
        if obj_path is None:
            obj_path = Path(stl_path).with_suffix('.obj')
        
        try:
            mesh = trimesh.load(stl_path, force='mesh', repair=True)
            mesh.export(obj_path, file_type='obj')
            print(f"转换成功: {stl_path} → {obj_path}")
            return obj_path
        except Exception as e:
            print(f"转换失败: {e}")
            return False

    def get_manifold(self, path, resolution=100000):
        path = Path(path)
        if path.suffix.lower() == '.stl':
            path = self.stl_to_obj(path)
    
        try:
            os.remove(self.manifold_temp)
        except:
            pass

        bat = f'{self.manifold} {path} {self.manifold_temp} {resolution}'
        os.system(bat)
    
    def get_simplifiy(self, path, output_path=None):
        path = Path(path)
        if path.suffix.lower() == '.stl':
            path = self.stl_to_obj(path)
        obj = trimesh.load(path, force='mesh', repair=True)
        if isinstance(obj, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(obj.geometry.values()))  # 合并所有网格
            # print(f"合并后的网格顶点数: {mesh.vertices.shape[0]}")
            # # 方法2：仅选择第一个子网格（按需选择）
            # # mesh = list(obj.geometry.values())[2]
        else:
            mesh = obj
        if not mesh.is_watertight:
            self.get_manifold(path)
            path = self.manifold_temp
        
        if not output_path:
            output_path = os.path.join(self.path_to_manifold, 'simplified.obj')
            try:
                os.remove(output_path)
            except:
                pass
        # print('simplify')
        bat = f'{self.simplify} -i {path} -o {output_path} -f 20000'
        os.system(bat)

class taxonomy_editor():
    def __init__(self, path_to_taxonomy):
        self.path_to_taxonomy = path_to_taxonomy
        # 读取JSON文件
        try:
            with open(self.path_to_taxonomy, 'r', encoding='utf-8') as f:
                self.data = json.load(f)  # data现在是Python字典
        except:
            self.data = {}
            print(f"无法读取文件 {self.path_to_taxonomy}，已新建文件。")
            with open(self.path_to_taxonomy, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2, ensure_ascii=False)

    def save(self):
        # 正确打开文件写入
        with open(self.path_to_taxonomy, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)


    #添加几何分类
    def add_object(self, name, parent_path=None):
        if parent_path:
            for key in parent_path.split('.'):
                self.data = self.data.setdefault(key, {})
        # 添加新分类
        try:
            self.data[name]
            print(f"分类 '{name}' 已存在！")
        except:
            self.data[name] = {}
    
    def add_key(self, name, key, value=None):
        try:
            self.data[name]
        except:
            print(f"分类 '{name}' 不存在！创建新分类。")
            self.add_object(name)
        
        try:
            self.data[name][key]
            print(f"键 '{key}' 已存在于分类 '{name}' 中！数据将被覆盖！")
        except:
            self.data[name][key] = {}

        if value is not None:
            self.data[name][key] = value
    
    def add_CFD(self,name, model_file_path, mesh_file_path, proj_file_path, case_list, Mach_list, AOA_list, cl_list, cd_list, cm_list):
        '''
        model_file_path: 模型文件路径
        mesh_file_path: 网格文件路径
        proj_file_path: CFD项目文件路径
        case_list: CFD计算结果列表  [case1.plt, case2.plt, ...]
        xx_list: 各个case的系数列表 [cl1, cl2, ...], [cd1, cd2, ...], [cm1, cm2, ...]
        '''
        # self.add_object(name)
        self.add_key(name, 'model', value=model_file_path)
        self.add_key(name, 'mesh', value=mesh_file_path)
        self.add_key(name, 'cfd', value=proj_file_path)
        for (i,case) in enumerate(case_list):
            self.add_key(name, f'case{i}')
            self.data[name][f'case{i}']['path'] = case
            self.data[name][f'case{i}']['Mach'] = Mach_list[i]
            self.data[name][f'case{i}']['AOA'] = AOA_list[i]
            self.data[name][f'case{i}']['cl'] = cl_list[i]
            self.data[name][f'case{i}']['cd'] = cd_list[i]
            self.data[name][f'case{i}']['cm'] = cm_list[i]

    def add_SDF(self, name, sdf_file_path):
        """
        添加SDF文件路径到分类
        :param name: 分类名称
        :param sdf_file_path: SDF文件路径
        """
        self.add_key(name, 'sdf', value=sdf_file_path)
                            




        


if __name__ == "__main__":
    # points, sdf_values = compute_sdf_from_single_object("E:\\Skyforge\\geo_model\\VspAircraft.stl")
    # print(f"Sampled {len(points)} points with SDF values.")
    # print(f'shape of points: {points.shape}, shape of sdf_values: {sdf_values.shape}')
    manifold = Manifold("D:\\SW\\Manifold-master\\build\\Release")
    manifold.get_simplifiy("D:\\SW\\Manifold-master\\build\\Release\\input.obj", output_path="D:\\SW\\Manifold-master\\build\\Release\\output.obj")
