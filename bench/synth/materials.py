"""Procedural solid textures and surface materials.

Every texture is a function of the **world-space hit point**, not of image
coordinates. That is the property the whole synthesizer rests on: the left and
right cameras see the same 3-D point, so they sample the same texture value,
so the only photometric difference between the two views is the one physics
puts there (view-dependent specular highlights, plus whatever sensor noise the
rules ask for). An image-space texture would leak a constant offset into every
matching cost and quietly make the data easier than reality.

Noise is value noise over an integer lattice, hashed rather than tabulated, so
it is deterministic for a given seed, needs no precomputed permutation table,
and evaluates on arbitrarily large point sets in one vectorised pass.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Texture kinds a material may use. Ordered roughly from "easy to match"
# (high-frequency, aperiodic, lots of unique structure) to "adversarial"
# (periodic or near-flat, where a correlation matcher has nothing to lock on).
TEXTURE_KINDS = (
    "gravel",    # single-octave high-frequency noise: richest signal
    "noise",     # fBm, natural-looking, broad spectrum
    "marble",    # fBm-warped sinusoid: smooth gradients plus veins
    "speckle",   # sparse hard blobs on a flat ground: isolated features
    "checker",   # periodic, axis-aligned: the classic ambiguity trap
    "stripes",   # periodic in one axis: ambiguous *along* the epipolar line
    "grid",      # thin lines on flat fill: strong edges, empty interiors
    "flat",      # untextured: matching is impossible without a prior
)

#: Frequency of each pattern's fundamental, relative to a material's
#: ``scale``. The multi-octave kinds start coarse because their octaves supply
#: the fine detail; the single-octave kinds put everything at the fundamental.
#: Without this, asking for "detail every 6 pixels" would give fBm whose
#: fourth octave sits below the Nyquist limit and aliases -- and aliasing is
#: the one kind of image noise that is genuinely different in the two views,
#: so it corrupts the ground truth rather than merely making matching harder.
_KIND_FREQUENCY = {
    "gravel": 1.0,
    "noise": 0.22,
    "marble": 0.30,
    "speckle": 0.9,
    "checker": 1.0,
    "stripes": 1.0,
    "grid": 1.0,
    "flat": 1.0,
}

_U32 = np.uint32


def _hash01(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    """Hash integer lattice coordinates to floats in ``[0, 1)``.

    A three-round integer bit mix, in the spirit of the murmur3 finalizer.
    Unsigned wraparound is the mechanism here, not an error, hence the
    suppressed overflow warnings.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        h = (ix.astype(_U32) * _U32(0x9E3779B1)) ^ (iy.astype(_U32) * _U32(0x85EBCA77))
        h = h ^ (iz.astype(_U32) * _U32(0xC2B2AE3D))
        h = h ^ (_U32(seed & 0xFFFFFFFF) * _U32(0x27D4EB2F))
        h = h ^ (h >> _U32(15))
        h = h * _U32(0x2545F491)
        h = h ^ (h >> _U32(13))
        h = h * _U32(0x3B4D7F2B)
        h = h ^ (h >> _U32(16))
    return (h >> _U32(8)).astype(np.float32) / np.float32(1 << 24)


def value_noise(p: np.ndarray, seed: int = 0) -> np.ndarray:
    """Trilinearly interpolated value noise at points ``p`` of shape ``(N, 3)``.

    Returns ``(N,)`` in ``[0, 1]``. Interpolation uses the smoothstep weight
    ``3f^2 - 2f^3`` so the field is C1 and has no visible lattice creases.
    """
    p = np.asarray(p, dtype=np.float32)
    i = np.floor(p)
    f = p - i
    w = f * f * (3.0 - 2.0 * f)
    ix = i[:, 0].astype(np.int64)
    iy = i[:, 1].astype(np.int64)
    iz = i[:, 2].astype(np.int64)

    c = {}
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                c[dx, dy, dz] = _hash01(ix + dx, iy + dy, iz + dz, seed)

    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
    x00 = c[0, 0, 0] + (c[1, 0, 0] - c[0, 0, 0]) * wx
    x10 = c[0, 1, 0] + (c[1, 1, 0] - c[0, 1, 0]) * wx
    x01 = c[0, 0, 1] + (c[1, 0, 1] - c[0, 0, 1]) * wx
    x11 = c[0, 1, 1] + (c[1, 1, 1] - c[0, 1, 1]) * wx
    y0 = x00 + (x10 - x00) * wy
    y1 = x01 + (x11 - x01) * wy
    return (y0 + (y1 - y0) * wz).astype(np.float32)


def fbm(
    p: np.ndarray, seed: int = 0, octaves: int = 4, gain: float = 0.55
) -> np.ndarray:
    """Fractional Brownian motion: octaves of value noise at doubling frequency.

    Summing octaves of a ``[0, 1]`` field concentrates the result around 0.5 --
    the same averaging that makes any sum of independent variables narrow. Left
    alone that halves the usable contrast of every fBm surface, so the result
    is stretched back out to roughly fill ``[0, 1]`` before it is returned.
    """
    total = np.zeros(len(p), dtype=np.float32)
    amp, norm, freq = 1.0, 0.0, 1.0
    for o in range(max(1, octaves)):
        total += amp * value_noise(p * freq, seed + 101 * o)
        norm += amp
        amp *= gain
        freq *= 2.0
    v = total / max(norm, 1e-6)
    # Independent octaves add in quadrature but are normalised by their plain
    # sum, so the result has only ``sqrt(sum g^2i) / sum g^i`` of one octave's
    # spread -- about 0.59 at four octaves, i.e. nearly half the contrast is
    # lost to the averaging alone. Undo exactly that ratio, so the stretch
    # stays correct for any octave count instead of being tuned for one.
    amps = [gain**o for o in range(max(1, octaves))]
    stretch = sum(amps) / max(float(np.sqrt(sum(a * a for a in amps))), 1e-6)
    return np.clip(0.5 + (v - 0.5) * stretch * 1.2, 0.0, 1.0).astype(np.float32)


@dataclass
class Material:
    """How one surface looks and how it scatters light.

    ``contrast`` scales the texture deviation from its mean, which is the
    single knob that turns a richly textured surface into a flat one. It is
    what the ``texture_richness`` rule ultimately drives.
    """

    kind: str = "noise"
    color_a: tuple[float, float, float] = (0.25, 0.26, 0.30)
    color_b: tuple[float, float, float] = (0.80, 0.78, 0.72)
    scale: float = 4.0           # texture cycles per world unit
    contrast: float = 1.0        # 0 = flat mean colour, 1 = full range
    specular: float = 0.15       # Blinn-Phong highlight strength
    shininess: float = 48.0
    reflectivity: float = 0.0    # single-bounce mirror fraction
    #: Fine surface grain added on top of whatever the pattern does. Real
    #: surfaces -- paint, paper, wood, concrete, fabric -- all carry detail
    #: down to the pixel, and it is a large part of why real images are so
    #: much more matchable than a smoothly shaded render. Without it the
    #: broad patterns alone leave big interiors locally flat, and a
    #: correlation window that lands inside one has nothing to lock onto.
    micro: float = 0.22
    micro_scale: float = 40.0    # cycles per world unit, set near pixel scale
    seed: int = 0

    def albedo(self, p: np.ndarray) -> np.ndarray:
        """Diffuse colour at world points ``p`` ``(N, 3)`` -> ``(N, 3)``."""
        t = self._pattern(p)
        if self.micro > 0.0 and self.kind != "flat":
            grain = value_noise(
                np.asarray(p, dtype=np.float32) * np.float32(self.micro_scale),
                self.seed + 7919,
            )
            t = np.clip(t + (grain - 0.5) * (2.0 * self.micro), 0.0, 1.0)
        a = np.asarray(self.color_a, dtype=np.float32)
        b = np.asarray(self.color_b, dtype=np.float32)
        mean = 0.5 * (a + b)
        c = a[None, :] + (b - a)[None, :] * t[:, None]
        # Pull toward the mean colour rather than toward grey, so lowering
        # contrast changes the texture without changing the scene tone.
        return (mean[None, :] + (c - mean[None, :]) * self.contrast).astype(np.float32)

    def _pattern(self, p: np.ndarray) -> np.ndarray:
        """The scalar pattern in ``[0, 1]`` before colour mapping."""
        k, s = self.kind, self.seed
        freq = np.float32(self.scale) * np.float32(_KIND_FREQUENCY.get(k, 1.0))
        q = np.asarray(p, dtype=np.float32) * freq

        if k == "flat":
            return np.full(len(q), 0.5, dtype=np.float32)
        if k == "gravel":
            return value_noise(q, s)
        if k == "noise":
            return fbm(q, s, octaves=4)
        if k == "marble":
            veins = np.sin((q[:, 0] + q[:, 2]) * 1.6 + 6.0 * fbm(q * 0.5, s, 3))
            return (0.5 + 0.5 * veins).astype(np.float32)
        if k == "speckle":
            n = value_noise(q, s)
            return (n > 0.62).astype(np.float32) * np.float32(0.9) + np.float32(0.05)
        if k == "checker":
            cell = np.floor(q).astype(np.int64)
            return ((cell[:, 0] + cell[:, 1] + cell[:, 2]) & 1).astype(np.float32)
        if k == "stripes":
            return (0.5 + 0.5 * np.sin(q[:, 0] * np.float32(np.pi))).astype(np.float32)
        if k == "grid":
            f = np.abs((q % 1.0) - 0.5)
            line = (f[:, 0] > 0.44) | (f[:, 1] > 0.44) | (f[:, 2] > 0.44)
            return line.astype(np.float32)
        raise ValueError(
            f"unknown texture kind {k!r}; expected one of {TEXTURE_KINDS}"
        )


@dataclass
class Light:
    """A point light, or a directional one when ``directional`` is set.

    Directional lights model the sun: the shadow ray travels to infinity along
    ``-position`` rather than toward a finite point.
    """

    position: tuple[float, float, float] = (3.0, 4.0, -1.0)
    color: tuple[float, float, float] = (1.0, 0.97, 0.92)
    intensity: float = 1.0
    directional: bool = False
    casts_shadow: bool = True
    #: Softened inverse-square constant: brightness is scaled by
    #: ``1 / (1 + falloff * d^2)``. Sized to the scene, because a value tuned
    #: for a 2 m tabletop leaves a 30 m corridor almost black, and both have
    #: to come out of the same renderer with usable contrast.
    falloff: float = 0.01


#: Luminance weights, for guaranteeing that a texture's two colours differ in
#: brightness and not only in hue. Most matchers work on intensity, so two
#: equally bright colours are an invisible texture as far as they are
#: concerned, however striking the pattern looks to a human.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def sample_material(
    rng: np.random.Generator,
    *,
    kinds: tuple[str, ...] = TEXTURE_KINDS,
    richness: float = 1.0,
    scale: float = 8.0,
    micro_scale: float = 40.0,
    specular: float = 0.18,
    reflective: bool = False,
    palette_warmth: float = 0.5,
    min_luma_contrast: float = 0.30,
) -> Material:
    """Draw a random material consistent with the given rules.

    ``scale`` is in cycles per world unit and is expected to come from
    :func:`bench.synth.environments.texture_scale_for`, which derives it from
    the camera so that detail lands at a useful *pixel* frequency.

    ``richness`` in ``[0, 1]`` maps to texture contrast: at 0 every surface is
    its own flat mean colour, which is the pathological case for any matcher
    relying on photometric similarity alone.
    """
    kind = str(rng.choice(np.array(kinds, dtype=object)))

    # Pick the two colours by *brightness first*, one dark and one light, then
    # give each a hue. Sampling two colours freely and hoping they differ is
    # how a texture ends up invisible: offsets that look generous in RGB can
    # still land on two equally bright shades, and most matching costs see
    # only brightness. Separating luminance explicitly makes the pattern's
    # strength a property the rules control rather than a dice roll.
    dark = float(rng.uniform(0.05, 0.34))
    light = float(rng.uniform(max(dark + min_luma_contrast, 0.52), 0.95))
    if rng.random() < 0.5:
        dark, light = light, dark

    def tinted(target_luma: float) -> np.ndarray:
        hue = rng.uniform(0.3, 1.0, size=3).astype(np.float32)
        # Warm palettes bias red and green up and blue down; cool the reverse.
        hue *= np.array(
            [0.6 + 0.8 * palette_warmth, 1.0, 0.6 + 0.8 * (1.0 - palette_warmth)],
            dtype=np.float32,
        )
        hue *= target_luma / max(float(hue @ _LUMA), 1e-6)
        return np.clip(hue, 0.02, 1.0).astype(np.float32)

    color_a, color_b = tinted(dark), tinted(light)

    contrast = float(np.clip(richness * rng.uniform(0.8, 1.15), 0.0, 1.3))
    return Material(
        kind=kind,
        color_a=tuple(float(v) for v in color_a),
        color_b=tuple(float(v) for v in color_b),
        scale=float(scale),
        contrast=contrast,
        specular=float(np.clip(rng.uniform(0.4, 1.6) * specular, 0.0, 1.0)),
        shininess=float(rng.uniform(16.0, 120.0)),
        reflectivity=float(rng.uniform(0.18, 0.45)) if reflective else 0.0,
        micro=float(np.clip(rng.uniform(0.12, 0.30) * richness, 0.0, 0.5)),
        micro_scale=float(micro_scale),
        seed=int(rng.integers(0, 2**31 - 1)),
    )
