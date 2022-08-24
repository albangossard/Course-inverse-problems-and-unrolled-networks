import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/../')
import numpy as np
import matplotlib.pyplot as plt
import torch, cv2
from skimage.transform import resize

from recon import tikhonov
import metrics
import nufftbindings.cufinufft as nufft

nx = ny = 320
Nbatch = 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

im = np.load('images/im_mri.npy')
f = torch.zeros(Nbatch, nx, ny).to(device).type(torch.complex64)
f[0] = torch.tensor(resize(im, (nx, ny)))


xi = torch.tensor(np.load("data/xi_10.npy")).to(device)
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

lamb = 1e-4
nitermax = 600

y = A(xi, f)

hist_psnr = []
def callback(ft):
    psnr = metrics.psnr(f, ft)
    hist_psnr.append(psnr.mean().item())

ftilde, hist_res = tikhonov(A, At, xi, y, lamb, nitermax, callback=callback)
plt.figure(0, figsize=(4,3))
plt.semilogy(hist_res)
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.tight_layout()
plt.savefig('images/mri/hist_res_tikhonov.pdf')

psnr = metrics.psnr(f, ftilde)
err = metrics.l2err(f, ftilde)
print("Error={:1.3e}  PSNR={:<4}".format(err.mean(), psnr.mean()))

plt.figure(1)
c=plt.imshow(f[0].real.cpu()); plt.colorbar(c)
plt.figure(2)
c=plt.imshow(ftilde[0].real.cpu()); plt.colorbar(c)
plt.figure(3, figsize=(4,3))
plt.plot(hist_psnr)
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('PSNR')
plt.tight_layout()
plt.savefig('images/mri/hist_psnr_tikhonov.pdf')
plt.show()

cv2.imwrite('images/mri/f.png', f[0].real.cpu().numpy()*255)
cv2.imwrite('images/mri/f_tikhonov.png', ftilde[0].real.cpu().numpy()*255)
