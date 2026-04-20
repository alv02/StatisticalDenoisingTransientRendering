"""
    Official implementation of "Statistical Denoising of Transient Rendering"
    Authors: Oscar Pueyo-Ciutad, Alvaro Lopez, Diego Gutierrez
    Main code dev: Alvaro Lopez
    Contact: o.pueyo@unizar.es
    MIT License
"""


import math

import torch
import torch.nn.functional as F
from scipy import stats
from torch import nn

EPSILON = 0.0


def extract_patches_3d(x, kernel_size, padding=0, stride=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)

    channels = x.shape[0]

    x = torch.nn.functional.pad(x, padding)
    # (C, H, W, T)
    x = (
        x.unfold(1, kernel_size[0], stride[0])
        .unfold(2, kernel_size[1], stride[1])
        .unfold(3, kernel_size[2], stride[2])
    )
    # (C, h_dim_out, w_dim_out, t_dim_out, kernel_size[0], kernel_size[1], kernel_size[2])
    x = x.permute(0, 4, 5, 6, 1, 2, 3).reshape(
        channels, kernel_size[0] * kernel_size[1] * kernel_size[2], -1
    )

    # (C, kernel_size[0] * kernel_size[1] * kernel_size[2],h_dim_out * w_dim_out * t_dim_out)
    return x


def get_dim_blocks(dim_in, kernel_size, padding=0, stride=1, dilation=1):
    return (dim_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def extract_tiles_3d(x, kernel_size, stride=1, dilation=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    x = x.contiguous()

    channels, depth, height, width = x.shape[-4:]
    d_blocks = get_dim_blocks(
        depth, kernel_size=kernel_size[0], stride=stride[0], dilation=dilation[0]
    )
    h_blocks = get_dim_blocks(
        height, kernel_size=kernel_size[1], stride=stride[1], dilation=dilation[1]
    )
    w_blocks = get_dim_blocks(
        width, kernel_size=kernel_size[2], stride=stride[2], dilation=dilation[2]
    )
    shape = (
        channels,
        d_blocks,
        h_blocks,
        w_blocks,
        kernel_size[0],
        kernel_size[1],
        kernel_size[2],
    )
    strides = (
        width * height * depth,
        stride[0] * width * height,
        stride[1] * width,
        stride[2],
        dilation[0] * width * height,
        dilation[1] * width,
        dilation[2],
    )

    x = x.as_strided(shape, strides)
    x = x.permute(1, 2, 3, 0, 4, 5, 6)
    return x


def combine_tiles_3d(x, kernel_size, output_shape, padding=0, stride=1, dilation=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    def get_dim_blocks(
        dim_in, dim_kernel_size, dim_padding=0, dim_stride=1, dim_dilation=1
    ):
        dim_out = (
            dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1
        ) // dim_stride + 1
        return dim_out

    channels = x.shape[1]
    h_dim_out, w_dim_out, t_dim_out = output_shape[2:]
    h_dim_inb = get_dim_blocks(
        h_dim_out, kernel_size[0], padding[0], stride[0], dilation[0]
    )
    w_dim_inb = get_dim_blocks(
        w_dim_out, kernel_size[1], padding[1], stride[1], dilation[1]
    )
    t_dim_in = get_dim_blocks(
        t_dim_out, kernel_size[2], padding[2], stride[2], dilation[2]
    )

    x = x.permute(1, 0, 2, 3, 4)
    x = x.view(
        channels,
        h_dim_inb,
        w_dim_inb,
        t_dim_in,
        kernel_size[0],
        kernel_size[1],
        kernel_size[2],
    )
    # (C, h_dim_in, w_dim_in, t_dim_in, kernel_size[0], kernel_size[1], kernel_size[2])

    x = x.permute(0, 4, 1, 5, 6, 2, 3)
    # (C, kernel_size[0], h_dim_in, kernel_size[1], kernel_size[2], w_dim_in, t_dim_in)

    x = x.contiguous().view(
        -1,
        channels * kernel_size[0] * h_dim_inb * kernel_size[1] * kernel_size[2],
        w_dim_inb * t_dim_in,
    )
    # (B, C * kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2], h_dim_in * w_dim_in)

    x = torch.nn.functional.fold(
        x,
        output_size=(w_dim_out, t_dim_out),
        kernel_size=(kernel_size[1], kernel_size[2]),
        padding=(padding[1], padding[2]),
        stride=(stride[1], stride[2]),
        dilation=(dilation[1], dilation[2]),
    )
    # (B, C * kernel_size[0] * d_dim_in, H, W)

    x = x.view(-1, channels * kernel_size[0], h_dim_inb * w_dim_out * t_dim_out)
    # (B, C * kernel_size[0], d_dim_in * H * W)

    x = torch.nn.functional.fold(
        x,
        output_size=(h_dim_out, w_dim_out * t_dim_out),
        kernel_size=(kernel_size[0], 1),
        padding=(padding[0], 0),
        stride=(stride[0], 1),
        dilation=(dilation[0], 1),
    )
    # (B, C, D, H * W)

    x = x.view(channels, h_dim_out, w_dim_out, t_dim_out)
    # (B, C, D, H, W)

    return x


class Tile(nn.Module):
    """
    Creates a tiled tensor for and image
    """

    def __init__(self, spatial_radius, temporal_radius):
        super().__init__()
        self.spatial_radius = spatial_radius
        self.temporal_radius = temporal_radius
        self.final_spatial_tile_size = 16
        self.final_temporal_tile_size = 4
        self.spatial_tile_size = self.final_spatial_tile_size + 2 * self.spatial_radius
        self.temporal_tile_size = (
            self.final_temporal_tile_size + 2 * self.temporal_radius
        )
        self.stride = (
            self.final_spatial_tile_size,
            self.final_spatial_tile_size,
            self.final_temporal_tile_size,
        )
        self.kernel_size = (
            self.spatial_tile_size,
            self.spatial_tile_size,
            self.temporal_tile_size,
        )

    def forward(self, x, padding_value=0):
        """
        Returns tensor (tiles, C, H, W, T)
        """
        c, h, w, t = x.shape
        x_padded = F.pad(
            x,
            (
                self.temporal_radius,  # T left
                self.final_temporal_tile_size - 1 + self.temporal_radius,  # T right
                self.spatial_radius,  # W left
                self.final_spatial_tile_size - 1 + self.spatial_radius,  # W right
                self.spatial_radius,  # H left
                self.final_spatial_tile_size - 1 + self.spatial_radius,  # H right
            ),
            mode="constant",
            value=padding_value,
        )

        tiles = extract_tiles_3d(
            x=x_padded, kernel_size=self.kernel_size, stride=self.stride
        )

        return tiles


class Shift(nn.Module):
    """
    Creates a tensor with the neighbours of each pixel
    """

    def __init__(self, spatial_radius, temporal_radius):
        super().__init__()
        self.spatial_radius = spatial_radius
        self.temporal_radius = temporal_radius
        self.spatial_kernel_size = 2 * self.spatial_radius + 1
        self.temporal_kernel_size = 2 * self.temporal_radius + 1
        self.n_patches = self.spatial_kernel_size**2 * self.temporal_radius
        self.kernel_size = (
            self.spatial_kernel_size,
            self.spatial_kernel_size,
            self.temporal_kernel_size,
        )

    def forward(self, x):
        """
        x (C, H, W, T)
        returns (C, n_patches, H_out * W_out * T_out)
        where H_out = H - 2*radius, W_out = W - 2*radius, T_out = T - 2*radius
        """
        c, h, w, t = x.shape
        patches = extract_patches_3d(x, kernel_size=self.kernel_size)

        return patches


class StatDenoiser(nn.Module):
    def __init__(
        self,
        spatial_radius=5,
        temporal_radius=5,
        alpha=0.1,
        spp=0,
        device="cpu",
    ):
        super(StatDenoiser, self).__init__()

        self.spatial_radius = spatial_radius
        self.temporal_radius = temporal_radius
        self.spatial_kernel_size = 2 * self.spatial_radius + 1
        self.temporal_kernel_size = 2 * self.temporal_radius + 1
        self.alpha = alpha
        self.n_patches = self.spatial_kernel_size**2 * self.temporal_kernel_size
        self.gamma_w = self.compute_gamma_w(spp, spp, alpha)
        self.gamma = self.compute_gamma(spp, spp, alpha)
        self.device = device

        sigma_inv = torch.tensor(
            [0.3, 0.3, 0.3, 50, 50, 50, 10, 10, 10], dtype=torch.float32
        )
        sigma_inv = torch.reshape(sigma_inv, (-1, 1, 1))
        self.register_buffer("sigma_inv", sigma_inv)

        self.shift = Shift(
            spatial_radius=spatial_radius, temporal_radius=temporal_radius
        )
        self.tile = Tile(spatial_radius=spatial_radius, temporal_radius=temporal_radius)

    def compute_gamma_w(self, n_i, n_j, alpha=0.005):
        degrees_of_freedom = n_i + n_j - 2
        gamma_w = stats.t.ppf(1 - alpha / 2, degrees_of_freedom)
        return torch.tensor(gamma_w, dtype=torch.float32)

    def compute_gamma(self, n_i, n_j, alpha=0.005):
        gamma_w = self.compute_gamma_w(n_i, n_j, alpha)
        return 1 / (2 * (gamma_w**2 + 1))

    def compute_w(
        self, estimand_i, estimand_j, estimand_i_variance, estimand_j_variance
    ):
        numerator_sym = (
            2 * ((estimand_i - estimand_j) ** 2)
            + estimand_i_variance
            + estimand_j_variance
        )
        denominator_sym = 2 * (
            (estimand_i - estimand_j) ** 2 + estimand_i_variance + estimand_j_variance
        )

        result = torch.where(
            denominator_sym == 0.0, 0.5, numerator_sym / denominator_sym
        )
        variance_zero = (estimand_i_variance + estimand_j_variance) <= EPSILON
        values_differ = estimand_i != estimand_j
        result = torch.where(
            (variance_zero & values_differ),
            1.0,
            result,
        )
        return result

    def compute_t_statistic(self, w_ij):
        inf_tensor = torch.full_like(w_ij, float("inf"))
        return torch.where(
            w_ij == 1.0,
            inf_tensor,
            torch.sqrt((1 / (2 * (1 - w_ij))) - 1),
        )

    def compute_bilateral_weights(self, guidance):
        shifted_guidance = self.shift(guidance)
        center_idx = self.n_patches // 2
        center_guidance = shifted_guidance[:, center_idx : center_idx + 1, :]

        diff = shifted_guidance - center_guidance
        diff_squared = diff**2
        weighted_diff = diff_squared * self.sigma_inv

        result = weighted_diff.sum(dim=0)
        bilateral_weights = torch.exp_(-0.5 * result)

        return bilateral_weights

    def compute_membership(self, estimands, estimands_variance):
        center_idx = self.n_patches // 2

        shifted_estimands = self.shift(estimands)
        shifted_estimands_variance = self.shift(estimands_variance)

        center_estimands = shifted_estimands[:, center_idx : center_idx + 1, :]
        center_estimands_variance = shifted_estimands_variance[
            :, center_idx : center_idx + 1, :
        ]

        w_ij = self.compute_w(
            center_estimands,
            shifted_estimands,
            center_estimands_variance,
            shifted_estimands_variance,
        )

        membership = ((1 - w_ij) > self.gamma).all(dim=0, keepdim=True).float()
        membership[:, center_idx : center_idx + 1, :] = 1.0

        return membership

    def create_guidance_3d(self, albedo, normals, h, w, t):
        albedo = (
            torch.from_numpy(albedo).to(torch.float32).permute(2, 0, 1).contiguous()
        )
        normals = (
            torch.from_numpy(normals).to(torch.float32).permute(2, 0, 1).contiguous()
        )

        spatial_radius = self.spatial_radius if self.spatial_radius > 0 else 1
        temporal_radius = self.temporal_radius if self.temporal_radius > 0 else 1

        y_coords = torch.linspace(
            -h / (2 * spatial_radius), h / (2 * spatial_radius), h, dtype=torch.float32
        )
        x_coords = torch.linspace(
            -w / (2 * spatial_radius), w / (2 * spatial_radius), w, dtype=torch.float32
        )
        z_coords = torch.linspace(
            -t / (2 * temporal_radius),
            t / (2 * temporal_radius),
            t,
            dtype=torch.float32,
        )
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

        pos_list = []
        for i in range(t):
            z_val = z_coords[i]
            z_grid = torch.full_like(x_grid, z_val)
            pos_frame = torch.stack([x_grid, y_grid, z_grid], dim=0)
            pos_list.append(pos_frame)
        pos = torch.stack(pos_list, dim=-1)

        albedo = albedo.squeeze(0).unsqueeze(-1).expand(-1, -1, -1, t)
        normals = normals.squeeze(0).unsqueeze(-1).expand(-1, -1, -1, t)

        guidance = torch.cat([pos, albedo, normals], dim=0)

        return guidance

    def load_tensors(self, images, albedo, normals, estimands, estimands_variance):
        images = (
            torch.from_numpy(images).to(torch.float32).permute(3, 0, 1, 2).contiguous()
        )

        _, h, w, t = images.shape
        guidance = self.create_guidance_3d(albedo, normals, h, w, t)

        estimands = (
            torch.from_numpy(estimands)
            .to(torch.float32)
            .permute(3, 0, 1, 2)
            .contiguous()
        )
        estimands_variance = (
            torch.from_numpy(estimands_variance)
            .to(torch.float32)
            .permute(3, 0, 1, 2)
            .contiguous()
        )

        guidance = guidance.to(self.device)
        estimands = estimands.to(self.device)
        estimands_variance = estimands_variance.to(self.device)
        images = images.to(self.device)
        self = self.to(self.device)

        self.sigma_inv = self.sigma_inv.to(self.device)

        return images, guidance, estimands, estimands_variance

    def forward(self, images, albedo, normals, estimands, estimands_variance):

        images, guidance, estimands, estimands_variance = self.load_tensors(
            images, albedo, normals, estimands, estimands_variance
        )

        C, H, W, T = images.shape
        # Tile inputs
        tiled_images = self.tile(images)
        tiled_guidance = self.tile(guidance)
        tiled_estimands = self.tile(estimands, 0.0)
        tiled_estimands_variance = self.tile(estimands_variance)

        tiles_y, tiles_x, tiles_t, _, _, _, _ = tiled_images.shape
        tiled_denoised_image = torch.empty(
            (
                tiles_y * tiles_x * tiles_t,
                C,
                self.tile.final_spatial_tile_size,
                self.tile.final_spatial_tile_size,
                self.tile.final_temporal_tile_size,
            )
        )
        tile_index = 0
        for y in range(tiles_y):
            for x in range(tiles_x):
                for t in range(tiles_t):

                    img_tile = tiled_images[y, x, t, :]
                    guidance_tile = tiled_guidance[y, x, t, :]
                    estimands_tile = tiled_estimands[y, x, t, :]
                    var_tile = tiled_estimands_variance[y, x, t, :]

                    # Calcular pesos
                    weights_jbf = self.compute_bilateral_weights(
                        guidance_tile
                    ).unsqueeze(0)
                    membership = self.compute_membership(estimands_tile, var_tile)
                    final_weights = weights_jbf * membership

                    # Obtener vecindario de píxeles de la imagen
                    shifted_image = self.shift(img_tile)

                    sum_weights = torch.sum(final_weights, dim=1)
                    final_weights = final_weights / sum_weights
                    weighted_values = shifted_image * final_weights
                    denoised_image = torch.sum(weighted_values, dim=1).reshape(
                        C,
                        self.tile.final_spatial_tile_size,
                        self.tile.final_spatial_tile_size,
                        self.tile.final_temporal_tile_size,
                    )
                    tiled_denoised_image[tile_index : tile_index + 1] = denoised_image
                    tile_index += 1

        H_OUT = (
            math.ceil(H / self.tile.final_spatial_tile_size)
            * self.tile.final_spatial_tile_size
        )
        W_OUT = (
            math.ceil(W / self.tile.final_spatial_tile_size)
            * self.tile.final_spatial_tile_size
        )
        T_OUT = (
            math.ceil(T / self.tile.final_temporal_tile_size)
            * self.tile.final_temporal_tile_size
        )
        result = combine_tiles_3d(
            tiled_denoised_image,
            (
                self.tile.final_spatial_tile_size,
                self.tile.final_spatial_tile_size,
                self.tile.final_temporal_tile_size,
            ),
            (1, C, H_OUT, W_OUT, T_OUT),
            0,
            self.tile.stride,
        )
        result = result[:, 0:H, 0:W, 0:T]
        result = result.permute(1, 2, 3, 0)
        return result
