import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def _pixel_eps_to_normalized(eps_px: float, device: torch.device):
    eps = (eps_px / 255.0) / IMAGENET_STD.to(device)  
    return eps

def _clamp01_from_normalized(x_norm: torch.Tensor):
    x = x_norm * IMAGENET_STD.to(x_norm.device) + IMAGENET_MEAN.to(x_norm.device)
    x = torch.clamp(x, 0.0, 1.0)
    x_norm = (x - IMAGENET_MEAN.to(x_norm.device)) / IMAGENET_STD.to(x_norm.device)
    return x_norm

@torch.no_grad()
def predict_label(model, x_norm: torch.Tensor) -> torch.Tensor:
    return model(x_norm).argmax(1)

def fgsm(model, x_norm: torch.Tensor, y: torch.Tensor, eps_px: float, targeted: bool=False):
    device = x_norm.device
    x = x_norm.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    if targeted:
        loss = -loss
    model.zero_grad(set_to_none=True)
    loss.backward()
    grad_sign = x.grad.detach().sign()

    eps = _pixel_eps_to_normalized(eps_px, device)
    x_adv = x + eps * grad_sign
    x_adv = _clamp01_from_normalized(x_adv)
    return x_adv.detach()

def pgd(model, x_norm: torch.Tensor, y: torch.Tensor, eps_px: float, alpha_px: float, steps: int, rand_init: bool=True, targeted: bool=False):
    device = x_norm.device
    eps = _pixel_eps_to_normalized(eps_px, device)
    alpha = _pixel_eps_to_normalized(alpha_px, device)
    x0 = x_norm.clone().detach()

    if rand_init:
        x = x0 + torch.empty_like(x0).uniform_(-1, 1) * eps
        x = _clamp01_from_normalized(x)
    else:
        x = x0.clone()

    for _ in range(steps):
        x.requires_grad_(True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        if targeted:
            loss = -loss
        model.zero_grad(set_to_none=True)
        loss.backward()
        grad_sign = x.grad.detach().sign()
        x = x + alpha * grad_sign
        x = torch.max(torch.min(x, x0 + eps), x0 - eps)
        x = _clamp01_from_normalized(x).detach()

    return x
