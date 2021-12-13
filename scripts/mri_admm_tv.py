import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/../')
import numpy as np
import matplotlib.pyplot as plt
import torch, cv2
from skimage.transform import resize

from recon import tikhonov, cg
import metrics
import nufftbindings.cufinufft as nufft
from tv import TV

nx = ny = 320
Nbatch = 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
isDouble = True
dtype = torch.complex128 if isDouble else torch.complex64

im = np.load('images/im_mri.npy')
f = torch.zeros(Nbatch, nx, ny).to(device).type(dtype)
f[0] = torch.tensor(resize(im, (nx, ny)))


lamb = 1e-3
beta = 1e0
nitermaxcg = 50
nitermax = 100
lambinit = 1e-4
gamma1 = 1e-1
gamma2 = 1e-1


xi = torch.tensor(np.load("data/xi_10.npy")).to(device)
print(xi.shape)
K = xi.shape[0]
# plt.scatter(xi[:,0].cpu(), xi[:,1].cpu(), s=1)
# plt.axis('equal')
# plt.show()

nufft.nufft.set_dims(K, (nx, ny), device, Nb=Nbatch, doublePrecision=True)
nufft.nufft.precompute(xi)

tv = TV()


def A(xi, f):
    return nufft.forward(xi, f)/np.sqrt(nx*ny)
def At(xi, y):
    return nufft.adjoint(xi, y)/np.sqrt(nx*ny)

y = A(xi, f)

x0, hist_res = tikhonov(A, At, xi, y, lambinit, nitermaxcg)


x = x0.clone()
z1 = gamma1*A(xi, x)
z2 = gamma2*tv.forward(x)
mu1 = torch.zeros_like(z1)
mu2 = torch.zeros_like(z2)
zeros = torch.zeros_like(z2.abs())
hist_psnr = []

def Cop(x):
    return gamma1**2*At(xi, A(xi, x))+gamma2**2*tv.adjoint(tv.forward(x))

for nit in range(nitermax):
    # min wrt x
    Bt_rhs = gamma1*At(xi, z1-mu1/beta)+gamma2*tv.adjoint(z2-mu2/beta)
    x, _ = cg(Cop, Bt_rhs, nitermaxcg)
    Ax = A(xi, x)
    Tx = tv.forward(x)

    # min wrt z1
    z1 = ( y/gamma1+beta*gamma1*Ax+mu1 )/(beta+1/gamma1**2)

    # min wrt z2
    z2 = gamma2*Tx+mu2/beta
    z2 = torch.sgn(z2)*torch.maximum(torch.abs(z2)-gamma2*lamb/beta, zeros)

    # update multipliers
    mu1 = mu1 + beta*(gamma1*Ax-z1)
    mu2 = mu2 + beta*(gamma2*Tx-z2)


    ftilde = x
    psnr = metrics.psnr(f, ftilde)
    err = metrics.l2err(f, ftilde)
    print("{:<3}/{:<3}  Error={:1.3e}  PSNR={:<4}".format(nit, nitermax, err.mean(), psnr.mean()))

    # plt.figure(1); plt.clf()
    # c=plt.imshow(f[0].real.cpu()); plt.colorbar(c)
    # plt.figure(2); plt.clf()
    # c=plt.imshow(ftilde[0].real.cpu()); plt.colorbar(c)
    # plt.title(str(nit))
    # plt.pause(1e-3)

    hist_psnr.append(psnr.mean().item())

plt.figure(3, figsize=(3,3))
plt.plot(hist_psnr)
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('PSNR')
plt.tight_layout()
plt.savefig('images/mri/hist_psnr_admm.pdf')
plt.show()

cv2.imwrite('images/mri/f_admm.png', ftilde[0].real.cpu().numpy()*255)
