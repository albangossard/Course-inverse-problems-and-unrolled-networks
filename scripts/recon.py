import torch


def cg(C, b, nitermax, callback=None):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    res_hist = []
    if len(b.shape)==3:
        ps_r = torch.abs(r*r).sum(dim=2).sum(dim=1)
        for k in range(nitermax):
            Cp = C(p)
            alpha = ps_r/(p*torch.conj(Cp)).sum(dim=2).sum(dim=1)
            x = x + alpha.unsqueeze(-1).unsqueeze(-1)*p
            r = r - alpha.unsqueeze(-1).unsqueeze(-1)*Cp
            ps_rp1 = torch.abs(r*r).sum(dim=2).sum(dim=1)
            beta = ps_rp1/ps_r
            p = r+beta.unsqueeze(-1).unsqueeze(-1)*p
            ps_r = ps_rp1
            res_hist.append(ps_r.sum().sqrt().item())
            if callback is not None:
                callback(x)
    else:
        ps_r = torch.abs(r*r).sum(dim=3).sum(dim=2).sum(dim=1)
        for k in range(nitermax):
            Cp = C(p)
            alpha = ps_r/(p*torch.conj(Cp)).sum(dim=3).sum(dim=2).sum(dim=1)
            x = x + alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*p
            r = r - alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*Cp
            ps_rp1 = torch.abs(r*r).sum(dim=3).sum(dim=2).sum(dim=1)
            beta = ps_rp1/ps_r
            p = r+beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*p
            ps_r = ps_rp1
            res_hist.append(ps_r.sum().sqrt().item())
            if callback is not None:
                callback(x)
    return x, res_hist


def tikhonov(A, At, xi, y, lamb, nitermax, callback=None):
    # Solves the linear system (A^T A + \lambda I) x = A^T y  ie Cx=b
    def Cop(x):
        return At(xi, A(xi, x))+lamb*x
    b = At(xi, y)
    return cg(Cop, b, nitermax, callback=callback)
