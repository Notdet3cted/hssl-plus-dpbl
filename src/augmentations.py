import torch

class SignalAugmentations:
    @staticmethod
    def add_noise(x, noise_level=0.01):
        noise = torch.randn_like(x) * noise_level
        return x + noise

    @staticmethod
    def scale(x, scale_factor=0.1):
        scale = torch.empty(x.shape[0], 1, 1).uniform_(1 - scale_factor, 1 + scale_factor).to(x.device)
        return x * scale

    @staticmethod
    def get_views(x):
        v1 = SignalAugmentations.add_noise(x, 0.05)
        v1 = SignalAugmentations.scale(v1, 0.1)
        
        v2 = SignalAugmentations.add_noise(x, 0.1)
        v2 = SignalAugmentations.scale(v2, 0.2)
        
        return v1, v2