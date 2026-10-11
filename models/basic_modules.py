import torch

def A_CDP(x, SamplingRate, mask):
    
    imsize = x.shape[2]   
    x = x.repeat(1, SamplingRate, 1, 1)
    x = mask*x
    Ax = torch.zeros_like(x, dtype=torch.complex64).to(x.device)   
    for i in range(SamplingRate):
        Ax[:, i, :, :] = torch.fft.fft2(x[:, i, :, :])*(1/imsize)   
    return Ax   


def At_CDP(Ax, SamplingRate, mask):
    
    B, C, imsize1, imsize2 = Ax.shape  
    Atx = torch.zeros_like(Ax)  
    for i in range(SamplingRate):
        Atx[:, i, :, :] = torch.fft.ifft2(Ax[:, i, :, :])
    mask_ = torch.conj(mask)   
    Atx = torch.sum(mask_ * Atx, axis=1)*imsize1
    return torch.real(Atx).reshape(B, 1, imsize1, imsize2)  



def Poisson_noise_torch(Mx, alpha):
    device = Mx.device
    norm = torch.abs(Mx)  
    alpha = torch.Tensor([alpha]).to(device)  
    B, C, w, h = norm.shape
    intensity_noise = alpha/255*norm*torch.randn(B, C, w, h).to(device)
    y = norm ** 2 + intensity_noise 
    y = y*(y > 0)
    y = torch.sqrt(y+1e-5)
    return y

