from matplotlib import pyplot as plt
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import numpy as np
import sys
from os.path import join as pjoin
import scipy.io as sio
import torch

data_dir = pjoin('/afs/crc.nd.edu/user/s/smalakpo/Manufacturing_ML/Samples/','NewData')
mat_fname = pjoin(data_dir, 'Indexed_TAT_Tmax_BinaryCT_Map.mat')
step_xy = .13
step_z = .05

mat_contents = sio.loadmat(mat_fname)
registered_data = np.asarray(mat_contents['IndexedInteriorMap'])
x = registered_data[:, 0]
y = registered_data[:, 1]
z = registered_data[:, 2]

len_data = lambda a, step:np.round((a-a.min())/step).max().astype(int) + 1
cor2ind = lambda a, step: np.round((a-a.min())/step).astype(int)
#print(type(cor2ind(x, step_xy)))
#print(type(len_data(x, step_xy)))

m, n, k = len_data(x, step_xy), len_data(y, step_xy), len_data(z, step_z)
#print(m, n, k)

# Labels
Y = np.zeros((m, n, k))-1
#print(cor2ind(x, step_xy).shape, cor2ind(y, step_xy).shape, cor2ind(z, step_z).shape)
Y[cor2ind(x, step_xy), cor2ind(y, step_xy), cor2ind(z, step_z)] = registered_data[:, 5]
#Y = np.expand_dims(Y, axis=3)
#print(Y[0:10,0:10,0])
#print(Y.shape)
#im = imshow(Y[:,:, 300],origin='lower')
#colorbar(im)
#plt.savefig("Y298.png",dpi=600)

#a, b, c = np.where(Y != -1)
#print(a.shape)

# Features
#registered_data[:, 4] = registered_data[:, 4]/100 
X = np.zeros((m, n, k, 2))-1
X[cor2ind(x, step_xy), cor2ind(y, step_xy), cor2ind(z, step_z)] = registered_data[:, 3:5]
#print(X.shape)

# Coordinates
#C = np.zeros((m, n, k, 3))-1
#C[cor2ind(x, step_xy), cor2ind(y, step_xy), cor2ind(z, step_z)]= registered_data[:, :3]
# Coordinates
Cx = np.zeros((m, n, k))-1
Cy = np.zeros((m, n, k))-1
Cz = np.zeros((m, n, k))-1
Cx[cor2ind(x, step_xy), cor2ind(y, step_xy), cor2ind(z, step_z)]= x
Cy[cor2ind(x, step_xy), cor2ind(y, step_xy), cor2ind(z, step_z)]= y
Cz[cor2ind(x, step_xy), cor2ind(y, step_xy), cor2ind(z, step_z)]= z
print(Y.shape)
print(X.shape)
print(Cx.shape)
#YC = np.concatenate((Y, C), axis=3)
#print(YC.shape)

np.savez("XY_raw.npz", X=X, Y=Y)
np.savez("Cx_raw.npz", X=Cx)
np.savez("Cy_raw.npz", X=Cy)
np.savez("Cz_raw.npz", X=Cz)