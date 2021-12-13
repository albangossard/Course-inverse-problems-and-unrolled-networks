import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/../')
import numpy as np
import matplotlib.pyplot as plt
import torch, cv2
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

from recon import tikhonov
import metrics
import nufftbindings.pykeops as nufft

nx = ny = 320
Nbatch = 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

phantom = shepp_logan_phantom()
f = torch.zeros(Nbatch, nx, ny).to(device).type(torch.complex64)
f[0] = torch.tensor(resize(phantom, (nx, ny)))


xi = torch.tensor(np.load("data/xi_grid_25.npy")).to(device)
print(xi.shape)
K = xi.shape[0]
# plt.scatter(xi[:,0].cpu(), xi[:,1].cpu(), s=1)
# plt.axis('equal')
# plt.show()

nufft.nufft.set_dims(K, (nx, ny), device, Nb=Nbatch)
nufft.nufft.precompute(xi)


def A(xi, f):
    return nufft.forward(xi, f)/np.sqrt(nx*ny)
def At(xi, y):
    return nufft.adjoint(xi, y)/np.sqrt(nx*ny)


y = A(xi, f)
ftilde = At(xi, y)

psnr = metrics.psnr(f, ftilde)
err = metrics.l2err(f, ftilde)
print("Error={:1.3e}  PSNR={:<4}".format(err.mean(), psnr.mean()))

plt.figure(1)
c=plt.imshow(f[0].real.cpu()); plt.colorbar(c)
plt.figure(2)
c=plt.imshow(ftilde[0].real.cpu()); plt.colorbar(c)
plt.show()

# cv2.imwrite('images/mri/f.png', f[0].real.cpu().numpy()*255)
# cv2.imwrite('images/mri/f_adjoint.png', ftilde[0].real.cpu().numpy()*255)
