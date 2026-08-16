import numpy as np


def apply_darkening(img, factor, noise_sigma=4.0, seed=0):
    """Reduce illumination by multiplying exposure by `factor` in [0, 1].

    Adds sensor-like Gaussian noise whose strength grows as light drops.
    factor = 1.0 returns the image unchanged.
    """
    img = np.asarray(img, dtype=np.float32)
    rng = np.random.default_rng(seed)
    out = img * factor
    if factor < 1.0:
        noise_scale = noise_sigma * (1.0 - factor)
        out += rng.normal(0.0, noise_scale, size=img.shape).astype(np.float32)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)