def plot3D(self, tensor):
    tensor = tensor.cpu().detach().numpy()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    x, y, z = torch.meshgrid(torch.arange(tensor.shape[0]), torch.arange(tensor.shape[1]), torch.arange(tensor.shape[2]))
    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Flatten the tensor and plot the points
    ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=tensor.flatten(), cmap='viridis')

    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Tensor Plot')
    plt.savefig('/root/dev/Merlion/3D_tensor_plot.png')
    # Show the plot
    plt.show()
    
def plot2D(self, tensor):
    tensor = tensor.cpu().detach().numpy()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # Create a heatmap plot
    plt.imshow(tensor, cmap='viridis', interpolation='nearest')
    plt.colorbar()

    # Add labels
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Tensor Plot')

    # Show the plot
    plt.show()