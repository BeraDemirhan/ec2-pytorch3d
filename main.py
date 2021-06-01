'''
This code's purpose is to deform an initial generic shape (e.g. sphere)
to fit a target shape.

We did:
- **load a mesh** from an '.obj' file
- use the PyTorch3D **Meshes** datastructure
- use 4 different PyTorch3D **mesh loss functions**
- set up an **optimization loop**

Starting from a sphere mesh, offset to each vertex in the mesh is learned
 with a way which the predicted mesh is closer to the target mesh at each 
optimization step. To this end, those are minimized:

+ `chamfer_distance`, which is the distance between the predicted (deformed) 
and target mesh, defined as the chamfer distance between the set of pointclouds 
resulting from **differentiably sampling points** from their surfaces.

However, solely minimizing the chamfer distance between the predicted and the 
target mesh will lead to a non-smooth shape (verify this by setting  
`w_chamfer=1.0` and all other weights to `0.0`).

Enforced smoothness by adding **shape regularizers** to the objective. 
Namely, added:
+ `mesh_edge_length`, which minimizes the length of the edges in the predicted mesh.
+ `mesh_normal_consistency`, which enforces consistency across the normals of 
neighboring faces.
+ `mesh_laplacian_smoothing`, which is the laplacian regularizer.
'''


from pytorch3d.utils import ico_sphere
import torch

from utility import (
    get_mesh, 
    optimization_loop,
    save
) 

def optimize_model(input_path, output_path, subdiv_level):
    # Set the device and chech whether cuda is available or not
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")

    # We initialize the source shape to be a sphere of radius 1
    trg_mesh, scale, center = get_mesh(input_path, device)
    src_mesh = ico_sphere(subdiv_level, device)
    _, out_mesh = optimization_loop(src_mesh, trg_mesh, device)
    save(out_mesh, scale, center, output_path)

if __name__ == "__main__":
    # Define input and output paths and desired subdiv_level
    input_path = "xxx.obj"
    output_path = "xxx_mesh_fit.obj"
    subdiv_level = 5
    
    optimize_model(input_path, output_path, subdiv_level)
