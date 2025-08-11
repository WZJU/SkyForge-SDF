## Install

Install Manifold from https://github.com/hjwdzh/Manifold

Download https://github.com/WZJU/SkyForge-SDF/tree/main/guidance

## Run

Run Guidance.ipynb

### Manifold 

Instantiate manifold class

The path should be the directory where the manifold and simplify executable file stored. You should build Manifold from source code by yourself.

```python
manifold = Manifold(".\\")
```

get_manifold converts a shape to be watertight, and get_simplifiy makes a shape more simple(smaller size). 

You can use get_simplifiy method directly. If a shape is not watertight, get_manifold method will be called.

```python
path_to_model = ".\\VspAircraft.stl"
manifold.get_manifold(path_to_model, resolution=100000)
manifold.get_simplifiy(path_to_model, output_path=None)
```

### SDF(Signed Distance Field)

Compute SDF for the model. Execute compute_sdf_from_single_object() method.

Make sure that the model is watertight. OBJ files and STL files are available.

```python
path_to_manifold_model = './simplified.obj'
points, sdf_values = compute_sdf_from_single_object(path_to_manifold_model, num_points=100000)
```

### JSON files

Edit json files for further use.

```python
path_to_json = './taxonomy.json'
tax = taxonomy_editor(path_to_json) # A taxonomy file will be loaded. If the file do not exist, a file will be created

# Create a key for a new object
name = 'Object 01'
tax.add_object(name)

# Add CFD files and data to the key
tax.add_CFD(name,
             model_file_path= 'model_file_path',
             mesh_file_path= 'mesh_file_path', 
             proj_file_path= 'proj_file_path', 
             case_list= ['case1.plt', 'case2.plt', 'case3.plt'], 
             Mach_list= [0.75, 0.8, 0.85], 
             AOA_list= [1, 3, 5], 
             cl_list= [0.1, 0.5, 0.9], 
             cd_list= [0.05, 0.058, 0.06], 
             cm_list= [0.01, 0.05, 0.06])

tax.save()
```

