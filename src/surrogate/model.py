"""The tilt-delta operator: a FiLM-conditioned FNO + wavelet hybrid.

Predicts a correction on top of the analytic pattern re-embedding, never the
map itself. With the residual head at zero the model *is* the analytic
baseline, so an untrained or out-of-distribution forward pass degrades to
physics rather than to noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

# The solved grid, and the grid every branch actually runs on. The padding is
# roughly an eighth of the domain: an FFT treats its input as periodic and a
# radio map is not, so a non-periodic domain needs room for the discontinuity
# at the wrap to ring into and be cropped away. 80 x 96 is also divisible by
# eight, which is what three levels of stride-2 wavelet decimation need.
MAP_SHAPE = (61, 74)
PADDED_SHAPE = (80, 96)

# The mask head predicts a change to the paths that already exist, not the
# paths themselves: its logit is added to this, signed by the input mask. So an
# untrained head says "the tilt change opened and closed nothing", which is
# right for most tiles and for every small delta.
_MASK_PRIOR_LOGIT = 4.0

# Orthonormal Haar analysis kernels, in the order LL, LH, HL, HH. Each has unit
# norm, so the synthesis filters are the same four kernels and reconstruction
# is exact.
_HAAR = 0.5 * torch.tensor(
    [
        [[1.0, 1.0], [1.0, 1.0]],
        [[1.0, 1.0], [-1.0, -1.0]],
        [[1.0, -1.0], [1.0, -1.0]],
        [[1.0, -1.0], [-1.0, 1.0]],
    ]
).reshape(4, 1, 2, 2)


@dataclass(frozen=True)
class OperatorSpec:
    """How wide, how deep, and which branches are switched on.

    The three ``use_*`` flags exist so the hybrid claim can be measured rather
    than asserted: the branches are already separate, so an ablation is a flag
    rather than a second model.

    Attributes:
        in_channels: Channels in the input stack; see
            :class:`src.surrogate.dataset.TiltPairs`.
        cond_dim: Length of the scalar conditioning vector.
        width: Channels the trunk runs at.
        depth: Number of hybrid blocks.
        modes: Fourier modes kept per axis.
        levels: Wavelet decomposition levels.
        use_fno: Include the spectral branch.
        use_wno: Include the wavelet branch.
        use_spatial_film: Modulate per pixel as well as per channel.
    """

    in_channels: int = 10
    cond_dim: int = 6
    width: int = 32
    depth: int = 4
    modes: int = 16
    levels: int = 3
    use_fno: bool = True
    use_wno: bool = True
    use_spatial_film: bool = True


def haar_forward(x: Tensor, kernel: Tensor) -> Tensor:
    """One level of Haar decomposition, ``[B, C, H, W] -> [B, C, 4, H/2, W/2]``.

    Grouped so each channel is transformed on its own: a wavelet basis is a
    property of the plane, not something to mix channels across.
    """
    channels = x.shape[1]
    bands = nn.functional.conv2d(x, kernel, stride=2, groups=channels)
    return bands.reshape(x.shape[0], channels, 4, *bands.shape[-2:])


def haar_inverse(bands: Tensor, kernel: Tensor) -> Tensor:
    """One level of Haar reconstruction, the exact inverse of :func:`haar_forward`."""
    batch, channels = bands.shape[0], bands.shape[1]
    flat = bands.reshape(batch, channels * 4, *bands.shape[-2:])
    return nn.functional.conv_transpose2d(flat, kernel, stride=2, groups=channels)


class SpectralConv2d(nn.Module):
    """Factorized Fourier layer: per-mode scaling, then a shared channel mixing.

    A dense ``[C, C, m, m]`` weight is the standard form (Li et al., ICLR 2021)
    and would be 99% of this model's parameters, against a few thousand
    training pairs. Factorizing it into a per-mode diagonal and one channel
    mixing -- the F-FNO construction (Tran et al., ICLR 2023) -- cuts that by
    more than an order of magnitude at no reported accuracy cost.

    Both corners of the half-spectrum are kept, as FNO2d does: ``rfft2`` halves
    only the last axis, so the low modes of the other one live at both ends.
    """

    def __init__(self, width: int, modes: int) -> None:
        """Build the two corners' weights."""
        super().__init__()
        self.modes = modes
        scale = 1.0 / width
        for corner in ("lower", "upper"):
            self.register_parameter(
                f"diag_{corner}",
                nn.Parameter(scale * torch.randn(width, modes, modes, dtype=torch.cfloat)),
            )
            self.register_parameter(
                f"mix_{corner}",
                nn.Parameter(scale * torch.randn(width, width, dtype=torch.cfloat)),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Filter ``[B, C, H, W]`` through the kept modes and back."""
        height, width = x.shape[-2:]
        spectrum = torch.fft.rfft2(x)
        out = torch.zeros_like(spectrum)
        modes = min(self.modes, height // 2, spectrum.shape[-1])

        for corner, rows in (("lower", slice(None, modes)), ("upper", slice(-modes, None))):
            diag = getattr(self, f"diag_{corner}")[:, :modes, :modes]
            scaled = spectrum[..., rows, :modes] * diag
            out[..., rows, :modes] = torch.einsum(
                "bckl,dc->bdkl", scaled, getattr(self, f"mix_{corner}")
            )
        return torch.fft.irfft2(out, s=(height, width))


class WaveletConv2d(nn.Module):
    """Wavelet layer: multi-level Haar, a learned mixing per subband, inverse.

    Multi-level deliberately. One level with a per-subband channel mixing sees
    a 2x2 block and nothing else -- a weaker operator than a 3x3 convolution,
    and with none of the across-scale structure a wavelet operator exists for
    (Tripura and Chakraborty, CMAME 2023). Three levels reach eight tiles.

    Haar rather than a smoother Daubechies basis because its discontinuity
    matches what the residual is made of: shadow boundaries cast by building
    footprints. Its decimation is shift-variant, which costs little here --
    the scene is fixed, so no feature ever moves relative to the lattice.
    """

    def __init__(self, width: int, levels: int) -> None:
        """Build one mixing per detail level plus one for the coarse band."""
        super().__init__()
        self.levels = levels
        self.register_buffer("kernel", _HAAR.repeat(width, 1, 1, 1))
        scale = 1.0 / width
        # Three detail subbands per level, sharing one weight tensor.
        self.detail = nn.ParameterList(
            nn.Parameter(scale * torch.randn(3, width, width)) for _ in range(levels)
        )
        self.coarse = nn.Parameter(scale * torch.randn(width, width))

    def forward(self, x: Tensor) -> Tensor:
        """Decompose ``[B, C, H, W]``, mix every subband, reconstruct."""
        details = []
        coarse = x
        for _ in range(self.levels):
            bands = haar_forward(coarse, self.kernel)
            coarse, detail = bands[:, :, 0], bands[:, :, 1:]
            details.append(detail)

        coarse = torch.einsum("bchw,dc->bdhw", coarse, self.coarse)
        for level in reversed(range(self.levels)):
            mixed = torch.einsum("bcshw,sdc->bdshw", details[level], self.detail[level])
            coarse = haar_inverse(torch.cat([coarse.unsqueeze(2), mixed], dim=2), self.kernel)
        return coarse


class Film(nn.Module):
    """Per-block feature-wise modulation, global and optionally per tile.

    Global modulation alone (Perez et al., AAAI 2018) cannot express what it is
    conditioning on here: tilting changes gain as a function of each tile's own
    elevation angle, lifting the map near the new boresight and dropping it
    away from it. One affine pair for the whole plane has no way to say that,
    so the spatial half reads the angle-off-boresight maps directly.

    Both producing layers start at zero, so an untrained block modulates by
    ``gamma = 1, beta = 0`` and the whole trunk is the identity.
    """

    def __init__(self, width: int, cond_dim: int, embed_dim: int, spatial: bool) -> None:
        """Build the scalar head and, when asked, the per-tile one."""
        super().__init__()
        self.scalar = nn.Sequential(
            nn.Linear(cond_dim, 2 * width), nn.GELU(), nn.Linear(2 * width, 2 * width)
        )
        nn.init.zeros_(self.scalar[-1].weight)
        nn.init.zeros_(self.scalar[-1].bias)

        self.spatial: nn.Conv2d | None = None
        if spatial:
            self.spatial = nn.Conv2d(embed_dim, 2 * width, kernel_size=1)
            nn.init.zeros_(self.spatial.weight)
            nn.init.zeros_(self.spatial.bias)

    def forward(self, x: Tensor, cond: Tensor, embed: Tensor) -> Tensor:
        """Modulate ``[B, C, H, W]`` by the conditioning vector and maps."""
        gamma, beta = self.scalar(cond)[:, :, None, None].chunk(2, dim=1)
        if self.spatial is not None:
            gamma_map, beta_map = self.spatial(embed).chunk(2, dim=1)
            gamma, beta = gamma + gamma_map, beta + beta_map
        return (1.0 + gamma) * x + beta


class HybridBlock(nn.Module):
    """Spectral, wavelet and pointwise paths, summed, modulated, activated."""

    def __init__(self, spec: OperatorSpec, embed_dim: int) -> None:
        """Build whichever branches ``spec`` switches on."""
        super().__init__()
        self.fno = SpectralConv2d(spec.width, spec.modes) if spec.use_fno else None
        self.wno = WaveletConv2d(spec.width, spec.levels) if spec.use_wno else None
        self.pointwise = nn.Conv2d(spec.width, spec.width, kernel_size=1)
        self.film = Film(spec.width, spec.cond_dim, embed_dim, spec.use_spatial_film)

    def forward(self, x: Tensor, cond: Tensor, embed: Tensor) -> Tensor:
        """Run one block over ``[B, C, H, W]``."""
        out = self.pointwise(x)
        if self.fno is not None:
            out = out + self.fno(x)
        if self.wno is not None:
            out = out + self.wno(x)
        return x + nn.functional.gelu(self.film(out, cond, embed))


class TiltOperator(nn.Module):
    """Map a radio map and a tilt change to the correction the ray tracer would make.

    Returns a residual in dB over the analytic pattern re-embedding, and a
    logit for whether each tile has a path after the change. The caller adds
    the residual to ``rsrp_cur + delta_a``; see
    :class:`src.surrogate.evaluator.SurrogateEvaluator`.

    Attributes:
        spec: The configuration this was built from.
    """

    def __init__(self, spec: OperatorSpec | None = None) -> None:
        """Build the lift, the blocks and the two heads."""
        super().__init__()
        self.spec = spec or OperatorSpec()
        embed_dim = 16

        self.lift = nn.Conv2d(self.spec.in_channels, self.spec.width, kernel_size=1)
        self.embed = nn.Sequential(nn.Conv2d(2, embed_dim, kernel_size=3, padding=1), nn.GELU())
        self.blocks = nn.ModuleList(
            HybridBlock(self.spec, embed_dim) for _ in range(self.spec.depth)
        )
        self.project = nn.Conv2d(self.spec.width, 2, kernel_size=1)
        # Start at the analytic baseline: no residual, no change of coverage.
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    @property
    def n_parameters(self) -> int:
        """Trainable parameter count in real floats, for the ablation table.

        A complex parameter is two trained floats but one element, so counting
        ``numel`` alone would report the spectral branch at half its true size
        -- which is the branch whose size the ablation is about.
        """
        return sum(
            p.numel() * (2 if p.is_complex() else 1) for p in self.parameters() if p.requires_grad
        )

    def forward(
        self, x: Tensor, cond: Tensor, cond_map: Tensor, has_path: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Predict the residual and the coverage logit.

        Args:
            x: Input stack, ``[B, in_channels, 61, 74]``.
            cond: Scalar conditioning, ``[B, cond_dim]``.
            cond_map: Angle off boresight before and after, ``[B, 2, 61, 74]``.
            has_path: Coverage of the input map, ``[B, 1, 61, 74]``, in ``{0, 1}``.

        Returns:
            The residual in dB and the post-change path logit, each
            ``[B, 1, 61, 74]``.
        """
        out = self.lift(pad(x))
        embed = self.embed(pad(cond_map))
        for block in self.blocks:
            out = block(out, cond, embed)

        residual, logit = crop(self.project(out)).chunk(2, dim=1)
        return residual, logit + _MASK_PRIOR_LOGIT * (2.0 * has_path - 1.0)


def pad(x: Tensor) -> Tensor:
    """Reflect-pad the solved grid out to :data:`PADDED_SHAPE`."""
    rows, cols = PADDED_SHAPE[0] - MAP_SHAPE[0], PADDED_SHAPE[1] - MAP_SHAPE[1]
    top, left = rows // 2, cols // 2
    return nn.functional.pad(x, (left, cols - left, top, rows - top), mode="reflect")


def crop(x: Tensor) -> Tensor:
    """Undo :func:`pad`."""
    rows, cols = PADDED_SHAPE[0] - MAP_SHAPE[0], PADDED_SHAPE[1] - MAP_SHAPE[1]
    top, left = rows // 2, cols // 2
    return x[..., top : top + MAP_SHAPE[0], left : left + MAP_SHAPE[1]]
