# ART Labs PyTorch3D -> EC2 Adaptation

This code's purpose is to deform an initial generic shape (e.g. sphere) to fit a target shape.

We did:
- **load a mesh** from an '.obj' file
- use the PyTorch3D **Meshes** datastructure
- use 4 different PyTorch3D **mesh loss functions**
- set up an **optimization loop**

Starting from a sphere mesh, offset to each vertex in the mesh is learned with a way which the predicted mesh is closer to the target mesh at each optimization step. To this end, those are minimized:

- `chamfer_distance`, which is the distance between the predicted (deformed) and target mesh, defined as the chamfer distance between the set of pointclouds resulting from **differentiably sampling points** from their surfaces.

However, solely minimizing the chamfer distance between the predicted and the target mesh will lead to a non-smooth shape (verify this by setting  `w_chamfer=1.0` and all other weights to `0.0`).

Enforced smoothness by adding **shape regularizers** to the objective. Namely, added:
- `mesh_edge_length`, which minimizes the length of the edges in the predicted mesh.
- `mesh_normal_consistency`, which enforces consistency across the normals of neighboring faces.
- `mesh_laplacian_smoothing`, which is the laplacian regularizer.

## Directory structure:

There is a utility, visualisation and main *.py file.

### visualisation.py

- **plot_pointcloud** will create a mathplotlib graph using sample_points.
- **visualize_loss** will create a mathplotlib graph for loss values. **kwargs is used for customizeable font and fig size.

### utility.py

- **get_mesh** takes path of the input object and availability of CUDA (device). Returns target mesh, scale and center.
- **optimization_loop** takes both source and target meshes and cuda information. Takes number of iterations and other weights and optimizes the final mesh with a for loop.  Calculates loss variables and returns them. Also returns optimized mesh.
- You can change niter (number of iterations). Increasing niter will increase process time but also quality. Recommended bounds are 5k to 10k for niter.
- **save** takes mesh which will be saved, scale, center and path of the output. Saves given mesh to the output path appropriately.

### main.py

- **optimize_model** checks availablity of CUDA and sets the device variable accordingly
- Then calls visualisation.py and utility.py functions with appropriate variables. Also defines input and output paths. Alo you can change subdiv_level to 4,5 or 6 (recommendation). This will increase the process time but it will also increase quality in some cases.

## Installation

- Launch a ec2 g3s.xlarge aws instance with at least 32GB HDD
- Transfer all files in this repo to the ec2 instance
- Transfer the input file ending with .obj
- change 'XXXXX' in PyTorch3D-bash.sh - aws configure key ID and access key
- Run the bash script: $ source PyTorch3D-bash.sh
- Wait for the CUDA and other Libraries to be installed
- Change input and output in the main.py accordingly
- Run the script: $ python3.7 main.py
- Once the process completed, you can tranfer the output file to your computer.