# import necessary modules
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker

# set font style
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.figure(dpi=560)
plt.rcParams['figure.figsize'] = (10, 10)
plt.xticks(size=40)
s = 24

# Load data from slam_poses.txt
# Format: timestamp tx ty tz qx qy qz qw
data = np.loadtxt("/home/t1102/Downloads/ORB-SLAM3-YOLOv3-master/Examples/Monocular/slam_poses.txt")

# Extract x, y, z coordinates (tx, ty, tz)
gtx = data[:, 1]  # tx
gty = data[:, 2]  # ty
gtz = data[:, 3]  # tz

# Check data range to avoid large or infinite values
print(f"x range: {gtx.min()} to {gtx.max()}")
print(f"y range: {gty.min()} to {gty.max()}")
print(f"z range: {gtz.min()} to {gtz.max()}")

# Filter out invalid or infinite values
mask = np.isfinite(gtx) & np.isfinite(gty) & np.isfinite(gtz)
gtx = gtx[mask]
gty = gty[mask]
gtz = gtz[mask]

# Create a new figure and set it into 3D using add_subplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set figure information
ax.set_title("Trajectory of indoor environment", size=s)
ax.set_xlabel("x (m)", size=s, labelpad=20)
ax.set_ylabel("y (m)", size=s, labelpad=20)
ax.set_zlabel("z (m)", size=s, labelpad=20)

# Limit the axes to the data range to avoid too many tick values
ax.set_xlim([gtx.min(), gtx.max()])
ax.set_ylim([gty.min(), gty.max()])
ax.set_zlim([gtz.min(), gtz.max()])

# Adjust tick intervals dynamically based on data range
ax.xaxis.set_major_locator(ticker.MultipleLocator((gtx.max() - gtx.min()) / 10))
ax.yaxis.set_major_locator(ticker.MultipleLocator((gty.max() - gty.min()) / 10))
ax.zaxis.set_major_locator(ticker.MultipleLocator((gtz.max() - gtz.min()) / 10))

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