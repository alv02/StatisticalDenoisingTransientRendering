import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from metrics.flip_loss import HDRFLIPLoss


def compute_ssim(noisy, reference):
    """
    Compute SSIM metric for transient data.

    Args:
        noisy: numpy array of shape (H, W, T, C)
        reference: numpy array of shape (H, W, T, C)

    Returns:
        mean ssim
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to torch and reorder to (T, H, W, C)
    noisy_torch = torch.from_numpy(noisy).permute(2, 0, 1, 3).float().to(device)
    reference_torch = torch.from_numpy(reference).permute(2, 0, 1, 3).float().to(device)

    # Calculate global data range
    global_max = max(noisy_torch.max().item(), reference_torch.max().item())
    global_min = min(noisy_torch.min().item(), reference_torch.min().item())
    global_data_range = global_max - global_min

    def ssim_frame(frame1, frame2, data_range):
        f1 = frame1.detach().cpu().numpy()
        f2 = frame2.detach().cpu().numpy()
        return ssim(f1, f2, channel_axis=2, data_range=data_range)

    ssim_per_frame = []
    for t in range(noisy_torch.shape[0]):
        s = ssim_frame(noisy_torch[t], reference_torch[t], global_data_range)
        ssim_per_frame.append(s)

    ssim_per_frame = np.array(ssim_per_frame)

    return np.mean(ssim_per_frame)


def compute_rmse(noisy, reference):
    """
    Compute RMSE metric for transient data.

    Args:
        noisy: numpy array of shape (H, W, T, C)
        reference: numpy array of shape (H, W, T, C)

    Returns:
        rmse
    """
    mse = np.mean((reference - noisy) ** 2)
    rmse = np.sqrt(mse)

    return rmse


def compute_flip(noisy, reference):
    """
    Compute FLIP metric for transient data using flip_evaluator.

    Args:
        noisy: numpy array of shape (H, W, T, C)
        reference: numpy array of shape (H, W, T, C)
        flip_type: "HDR" or "LDR" (default: "HDR")

    Returns:
        mean flip
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to torch and reorder to (T, H, W, C)
    noisy_torch = torch.from_numpy(noisy).permute(2, 3, 0, 1).float().to(device)
    reference_torch = torch.from_numpy(reference).permute(2, 3, 0, 1).float().to(device)

    hdrflip_loss_fn = HDRFLIPLoss().to(device)

    loss_per_frame = []
    for t in range(noisy_torch.shape[0]):
        loss = hdrflip_loss_fn(
            noisy_torch[t].unsqueeze(0),
            reference_torch[t].unsqueeze(0),
        )
        loss_per_frame.append(loss.item())

    return np.mean(loss_per_frame)
