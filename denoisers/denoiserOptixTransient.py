import drjit as dr
import mitsuba as mi
import numpy as np


def denoiseOptixTransientTemporal(images, albedo, normals):
    """
    OptiX temporal denoising.
    """

    h, w, t, _ = images.shape
    denoiser = mi.OptixDenoiser(
        input_size=(w, h), albedo=True, normals=True, temporal=True
    )

    denoised_results = np.zeros((h, w, t, 3), dtype=np.float32)

    # Tensores (device automático)
    albedo_tensor = dr.auto.ad.TensorXf(albedo.astype(np.float32))
    normals_tensor = dr.auto.ad.TensorXf(normals.astype(np.float32))
    flow_zeros = dr.auto.ad.TensorXf(np.zeros((h, w, 2), dtype=np.float32))

    prev_frame = None

    for i in range(t):
        noisy_tensor = dr.auto.ad.TensorXf(images[:, :, i, :].astype(np.float32))

        if prev_frame is None:
            denoised = denoiser(
                noisy_tensor,
                albedo=albedo_tensor,
                normals=normals_tensor,
                flow=flow_zeros,
                previous_denoised=noisy_tensor,
            )
        else:
            denoised = denoiser(
                noisy_tensor,
                albedo=albedo_tensor,
                normals=normals_tensor,
                flow=flow_zeros,
                previous_denoised=prev_frame,
            )

        denoised_np = np.array(denoised)
        denoised_results[:, :, i, :] = denoised_np
        prev_frame = dr.auto.ad.TensorXf(denoised_np)

    return denoised_results
