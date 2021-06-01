from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80

def plot_pointcloud(sample_points, title=""):
    # Sample points uniformly from the surface of the mesh.
    x, y, z = sample_points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()

def visualize_loss(loss_tuple, **kwargs):
    chamfer_losses, edge_losses, normal_losses, laplacian_losses = loss_tuple
    figsize = kwargs.setdefault("figsize", (13, 5))
    fontsize = kwargs.setdefault('fontsize', '16')
    chamfer_losses
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    ax.plot(chamfer_losses, label="chamfer loss")
    ax.plot(edge_losses, label="edge loss")
    ax.plot(normal_losses, label="normal loss")
    ax.plot(laplacian_losses, label="laplacian loss")
    ax.legend(fontsize=fontsize)
    ax.set_xlabel("Iteration", fontsize=fontsize)
    ax.set_ylabel("Loss", fontsize=fontsize)
    ax.set_title("Loss vs iterations", fontsize=fontsize)
