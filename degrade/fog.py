import numpy as np


def row_depth_profile(shape, horizon_frac=0.62, d_far=150.0):
    """Distance-from-camera per image row (metres).

    Bottom of the frame (y = H-1) is nearest ground, depth grows linearly up to the
    horizon row, everything above the horizon is treated as far field.
    """
    h = shape[0]
    horizon = max(int(horizon_frac * h), 1)
    ys = np.arange(h, dtype=np.float32)
    depth = np.empty(h, dtype=np.float32)
    depth[:horizon] = d_far
    depth[horizon:] = (h - ys[horizon:]) / max(h - horizon, 1) * d_far
    return depth


def apply_fog(img, beta, horizon_frac=0.62, d_far=150.0, atmospheric_light=200.0):
    """Koschmieder attenuation model: I_out = I*t + A*(1-t), t = exp(-beta*d).

    beta = 0 returns the image unchanged. Larger beta -> denser fog.
    """
    img = np.asarray(img, dtype=np.float32)
    depth = row_depth_profile(img.shape, horizon_frac, d_far)
    transmittance = np.exp(-beta * depth)[:, None, None]
    out = img * transmittance + atmospheric_light * (1.0 - transmittance)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)