import sys
import numpy as np
import torch

data = np.load("XY_raw.npz")
X, Y = data["X"], data["Y"]
datx = np.load("Cx_raw.npz")
Cx = datx["X"]
daty = np.load("Cy_raw.npz")
Cy = daty["X"]
datz = np.load("Cz_raw.npz")
Cz = datz["X"]
#print(X.shape, Y.shape)

X, Y = torch.tensor(X), torch.tensor(Y)
Cx, Cy, Cz = torch.tensor(Cx), torch.tensor(Cy), torch.tensor(Cz) 
#print(X.shape, Y.shape)

#unfold = torch.nn.Unfold(kernel_size=(3, 3, 3))
k = [1, 3, 5, 7]
# label unfolding
#Yu = Y[:,:,:,0].unfold(0, k, 1).unfold(1, k, 1).unfold(2, k, 1).reshape(-1, k, k, k).reshape(-1, k**3)
Yu = Y.unfold(0, k[1], 1).unfold(1, k[1], 1).unfold(2, k[1], 1).reshape(-1, k[1], k[1], k[1]).reshape(-1, k[1]*k[1]*k[1])
#print(Yu.shape)

# feature unfolding
Xu = X.unfold(0, k[1], 1).unfold(1, k[1], 1).unfold(2, k[1], 1).reshape(-1, k[1], k[1], k[1], 2).reshape(-1, 2*(k[1]*k[1]*k[1]))
#print(Xu.shape)


#a, b = np.where(Yu == -1)
#print(a.shape, b.shape, a[:10], b[:10])

# Coordinates Unfolding
Cx = Cx.unfold(0, k[1], 1).unfold(1, k[1], 1).unfold(2, k[1], 1).reshape(-1, k[1], k[1], k[1]).reshape(-1, k[1]*k[1]*k[1])
#print(Cx.shape)
Cy = Cy.unfold(0, k[1], 1).unfold(1, k[1], 1).unfold(2, k[1], 1).reshape(-1, k[1], k[1], k[1]).reshape(-1, k[1]*k[1]*k[1])
Cz = Cz.unfold(0, k[1], 1).unfold(1, k[1], 1).unfold(2, k[1], 1).reshape(-1, k[1], k[1], k[1]).reshape(-1, k[1]*k[1]*k[1])
#Cu = Y[:,:,:,1:4].unfold(0, k, 1).unfold(1, k, 1).unfold(2, k, 1).unfold(3, k, 1).reshape(-1, k, k, k, 3).reshape(-1, k**3, 3)
indx_nan = torch.unique((Cx == -1).nonzero()[:, 0])
indx = np.setdiff1d(range(Cx.shape[0]), indx_nan)
indy_nan = torch.unique((Cy == -1).nonzero()[:, 0])
indy = np.setdiff1d(range(Cy.shape[0]), indy_nan)
indz_nan = torch.unique((Cz == -1).nonzero()[:, 0])
indz = np.setdiff1d(range(Cz.shape[0]), indz_nan)

Cx = torch.unsqueeze(Cx[indx, (k[1]*k[1]*k[1])//2], 1).numpy()
Cy = torch.unsqueeze(Cy[indy, (k[1]*k[1]*k[1])//2], 1).numpy()
Cz = torch.unsqueeze(Cz[indz, (k[1]*k[1]*k[1])//2], 1).numpy()
C = np.concatenate((Cx, Cy, Cz), axis=1)
#print(C.shape)
#sys.exit()
ind_nan = torch.unique((Yu == -1).nonzero()[:, 0])
ind = np.setdiff1d(range(Yu.shape[0]), ind_nan)
#print(ind_nan.shape, ind.shape)

#X, Y, C = Xu[ind].numpy(), Yu[ind, (k**3)//2].numpy(), Cu[ind, (k**3)//2].numpy()
X, Y = Xu[ind].numpy(), Yu[ind, (k[1]*k[1]*k[1])//2].numpy()
print(Y.shape)
print(X.shape)
print(C.shape)
#print(X[100,:])
#print(C[100,:])
#print(Y[100])
#sys.exit()
#print(X.shape, Y.shape, C.shape)
#Y = np.expand_dims(Y, axis=1)
#YC = np.concatenate((Y, C), axis=1)
ind_z = np.argsort(C[:,2])
X = X[ind_z, :]
Y = Y[ind_z]
C = C[ind_z,:]
#print(X.shape, Y.shape, C.shape)
#print(C[:100,:])
np.savez("raw_cord"+str(k[1])+"x"+str(k[1])+"x"+str(k[1])+".npz", X=C)
np.savez("raw_XYu"+str(k[1])+"x"+str(k[1])+"x"+str(k[1])+".npz", X=X, Y=Y)