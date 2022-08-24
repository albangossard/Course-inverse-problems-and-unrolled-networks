import torch

class TV:
    def __init__(self, resx=1, resy=1):
        self.resx = resx
        self.resy = resy
    def _forward_x(self, f):
        g = torch.zeros_like(f)
        g[...,:-1,:] = (f[...,1:,:]-f[...,:-1,:])/self.resx
        return g
    def _adjoint_x(self, g):
        f = torch.zeros_like(g)
        f[...,0,:]=-g[...,0,:]/self.resx
        f[...,-1,:]=g[...,-2,:]/self.resx
        f[...,1:-1,:] = -(g[...,1:-1,:]-g[...,:-2,:])/self.resx
        return f
    def _forward_y(self, f):
        g = torch.zeros_like(f)
        g[...,:-1] = (f[...,1:]-f[...,:-1])/self.resy
        return g
    def _adjoint_y(self, g):
        f = torch.zeros_like(g)
        f[...,0]=-g[...,0]/self.resy
        f[...,-1]=g[...,-2]/self.resy
        f[...,1:-1] = -(g[...,1:-1]-g[...,:-2])/self.resy
        return f
    def forward(self, f):
        ret = torch.zeros((2,)+f.shape, dtype=f.dtype, device=f.device)
        ret[0] = self._forward_x(f)
        ret[1] = self._forward_y(f)
        return ret
    def adjoint(self, g):
        return self._adjoint_x(g[0])+self._adjoint_y(g[1])
