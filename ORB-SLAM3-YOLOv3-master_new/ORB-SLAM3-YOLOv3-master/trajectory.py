# import necessary modules
from numpy.core.fromnumeric import size
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

# set font style
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.figure(dpi=560)
plt.rcParams['figure.figsize'] = (10,10)
plt.xticks(size=40)
s = 24

# load data from file
gt = np.loadtxt("./kitti/VO.txt")

# Extract relevant columns
gtx = gt[:, 3]
gty = gt[:, 7]
gtz = gt[:, 6]  # You can modify this index according to your need

# Create a new figure and set it into 3D using add_subplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# set figure information
ax.set_title("Trajectory of indoor environment", size=s)
ax.set_xlabel("x (m)", size=s, labelpad=20)
ax.set_ylabel("y (m)", size=s, labelpad=20)
ax.set_zlabel("z (m)", size=s, labelpad=20)
ax.set_ylim(-0.5, 0.5)

# Set the label size for each axis
ax.xaxis.label.set_size(s)
ax.yaxis.label.set_size(s)
ax.zaxis.label.set_size(s)

# Set tick size and location
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.zaxis.set_major_locator(ticker.MultipleLocator(0.2))

# Set tick font size
ax.tick_params(labelsize=14)

# Draw the figure, the color is green for the trajectory
ax.plot(gtx, gty, gtz, c='g', label='Trajectory')
ax.scatter(gtx[0], gty[0], gtz[0], c='k', marker='o', s=100, label='Start')
ax.scatter(gtx[-1], gty[-1], gtz[-1], c='r', marker='x', s=100, label='End')

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(1.77, 0.2), fontsize=s)

# Save the figure
plt.savefig("Trajectory_of_indoor_environment.tiff", dpi=560, bbox_inches='tight')

# Show the plot (optional)
plt.show()