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
s = 24
# load data from file
# you can replace this using with open

gt = np.loadtxt("./kitti/f_3.txt")

ORB3 = np.loadtxt("./kitti/ORB3.txt")
ORB2_N = np.loadtxt("./kitti/ORB2_NoLC.txt")
ORB2 = np.loadtxt("./kitti/ORB2.txt")

ORB3x = ORB3[:, 3]
ORB3y = ORB3[:, 7]
ORB3z = ORB3[:, 11]

ORB2Nx = ORB2_N[:,3]
ORB2Ny = ORB2_N[:,7]
ORB2Nz = ORB2_N[:,11]

ORB2x = ORB2[:,3]
ORB2y = ORB2[:,7]
ORB2z = ORB2[:,11]


gtx = gt[:,3]
gty = gt[:,7]
gtz = gt[:,11]


error_ORB3 = []
error_ORB2N = []
error_ORB2 = []

mean = []
#error = np.zeros((3557,1))
for i in range(3557):
    dis = ((ORB3x[i]-gtx[i])**2+(ORB3y[i]-gty[i])**2+(ORB3z[i]-gtz[i])**2)**0.5
    error_ORB3.append(dis)
error_ORB3 = np.array(error_ORB3)

for i in range(3557):
    dis = ((ORB2x[i]-gtx[i])**2+(ORB2y[i]-gty[i])**2+(ORB2z[i]-gtz[i])**2)**0.5
    error_ORB2.append(dis)
error_ORB2 = np.array(error_ORB2)

for i in range(3557):
    dis = ((ORB2Nx[i]-gtx[i])**2+(ORB2Ny[i]-gty[i])**2+(ORB2Nz[i]-gtz[i])**2)**0.5
    error_ORB2N.append(dis)
error_ORB2N = np.array(error_ORB2N)



plt.figure('Draw')
plt.plot(error_ORB3,c='b',label = 'ORBSLAM3')
plt.plot(error_ORB2,c='r', label = 'ORBSLAM2')
plt.plot(error_ORB2N,c='g', label = 'VO')


#plt.title("Time-error",size=s)
plt.xlabel("time",size=s, labelpad=20)
plt.ylabel("error",size=s, labelpad=20)


# draw the figure, the color is r = read
plt.legend(loc = 'upper right',bbox_to_anchor=(0.5,1), fontsize=s)
plt.savefig(f"time-error.tiff", dpi=560, bbox_inches = 'tight')
plt.show()
