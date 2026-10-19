# import necessary module
from numpy.core.fromnumeric import size
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
#set font style
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.figure(dpi=560)
plt.rcParams['figure.figsize'] = (10,10)
plt.xticks(size=40)
s = 32
# load data from file
# you can replace this using with open
ORB3 = np.loadtxt("./kitti/indoor/0801/f_0801-1.kitti") #ORB2
gt = np.loadtxt("./kitti/indoor/0801/f_ORB2-0801-1.kitti") #ORB3
mine = np.loadtxt("./kitti/indoor/0801/f_VO-0801-1.kitti") #VO
ORB2_N = np.loadtxt("./kitti/indoor/0801/ORB2-0801-1.kitti") #ORB2 yolo
ORB2 = np.loadtxt("./kitti/indoor/0803/f_0803-1.kitti") #gt

ORB3x = ORB3[:, 3]
ORB3y = ORB3[:, 7]
ORB3z = ORB3[:, 11]

ORB2_Nx = ORB2_N[:, 3]
ORB2_Ny = ORB2_N[:, 7]
ORB2_Nz = ORB2_N[:, 11]

ORB2x = ORB2[:,3]
ORB2y = ORB2[:,7]
ORB2z = ORB2[:,11]

gtx = gt[:,3]
gty = gt[:,7]
gtz = gt[:,11]

minex = mine[:,3]
miney = mine[:,7]
minez = mine[:,11]


# new a figure and set it into 3d
fig = plt.figure()
ax = fig.gca(projection='3d')

# set figure information
ax.labelsize = 40
#ax.set_title("Trajectory of different algorithm",size=s)
ax.set_title("Trajectory of proposed algorithm",size=s)
ax.set_xlabel("x (m)",size=s, labelpad=20)
ax.set_ylabel("y (m)",size=s, labelpad=20)
ax.set_zlabel("z (m)",size=s, labelpad=20)
ax.set_ylim(-0.5,0.5)



# draw the figure, the color is r = read
figure = ax.plot(ORB2x, ORB2y, ORB2z, label = 'groundtruth', c='k')
figure = ax.plot(ORB2_Nx, ORB2_Ny, ORB2_Nz, label = 'Proposed method', c='g')
figure = ax.plot(gtx, gty, gtz  , c='m', label = 'ORB3')
figure = ax.plot(ORB3x, ORB3y, ORB3z, label = 'ORB2', c='r')
figure = ax.plot(minex, miney, minez, c='b', label = 'VO')


plt.legend(loc = 'upper right',bbox_to_anchor=(1.77,0.5), fontsize=s)

plt.savefig(f"Trajectory of different algorithm.tiff", dpi=560, bbox_inches = 'tight')
#plt.savefig(f"Trajectory proposed algorithm.tiff", dpi=560, bbox_inches = 'tight')
plt.show()
