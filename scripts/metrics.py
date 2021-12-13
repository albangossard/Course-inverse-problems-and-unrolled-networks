import numpy as np
import torch

def psnr(f, f_tilde):
    nx = f.shape[-2]
    ny = f.shape[-1]
    if isinstance(f, torch.Tensor):
        if f_tilde.is_complex():
            return -10*torch.log10( ((f-f_tilde).abs()**2).sum(axis=1).sum(axis=1)/(nx*ny*(torch.amax(f.abs(),axis=(1,2))-torch.amin(f.abs(),axis=(1,2)))) )
        else:
            return -10*torch.log10( ((f-f_tilde)**2).sum(axis=1).sum(axis=1).sum(axis=1)/(nx*ny*(torch.amax(f,axis=(1,2,3))-torch.amin(f,axis=(1,2,3)))) )
    else:
        if np.iscomplex():
            return -10*np.log10( (np.abs(f-f_tilde)**2).sum(axis=1).sum(axis=1)/(nx*ny*(np.amax(np.abs(f),axis=(1,2))-np.amin(np.abs(f),axis=(1,2)))) )
        else:
            return -10*np.log10( ((f-f_tilde)**2).sum(axis=1).sum(axis=1).sum(axis=1)/(nx*ny*(np.amax(f,axis=(1,2,3))-np.amin(f,axis=(1,2,3)))) )

def l2err(f, f_tilde):
    if isinstance(f, torch.Tensor):
        return torch.norm((f-f_tilde).view(f.shape[0],-1), dim=-1)/torch.norm(f.view(f.shape[0],-1), dim=-1)
    else:
        return np.linalg.norm((f-f_tilde).reshape(f.shape[0],-1), axis=-1)/np.linalg.norm((f).reshape(f.shape[0],-1), axis=-1)
