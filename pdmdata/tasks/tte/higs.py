"""Model-independent HIGS corruption inspired by PC-MambaSDE's public code.

The published paper and public implementation differ. This version explicitly
follows the public burst and timestamp rules, with RUL always in cycle units.
Positive ``noise_rul_factor`` uses labels only for benchmark simulation; it is
not an unlabeled deployment transform. No target is returned as a model feature.
"""
import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class HigsSpec:
    sensor_drop: float = 0.5
    burst_probability: float = 0.05
    burst_length_mean: float = 5.0
    jitter_scale: float = 0.1
    base_noise_sigma: float = 0.01
    noise_rul_factor: float = 0.5
    rul_cap: float = 125.0

    def __post_init__(self):
        values = asdict(self)
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("HIGS parameters must be finite")
        if not 0 <= self.sensor_drop <= 1 or not 0 <= self.burst_probability <= 1:
            raise ValueError("HIGS probabilities must be between zero and one")
        if min(self.jitter_scale, self.base_noise_sigma, self.noise_rul_factor) < 0:
            raise ValueError("HIGS noise parameters must be nonnegative")
        if self.rul_cap <= 0 or self.burst_length_mean <= 0:
            raise ValueError("HIGS cap and burst length must be positive")


def higs_corrupt(values, *, spec=HigsSpec(), rng=None, target_cycles=None):
    """Return values, binary mask, and normalized local timestamps for one window.

    The same RNG seed gives exactly the same observation to every model. Sensor
    and burst loss are simulated without discarding samples or imputing data.
    Normalized RUL inputs are not silently converted: callers pass cycle units.
    """
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2 or not min(x.shape) or not np.isfinite(x).all():
        raise ValueError("HIGS expects a finite nonempty time-by-sensor matrix")
    if spec.noise_rul_factor:
        if target_cycles is None or not np.isfinite(target_cycles) or target_cycles < 0:
            raise ValueError("Label-dependent benchmark noise requires nonnegative RUL cycles")
        degradation = 1 - np.clip(float(target_cycles), 0, spec.rul_cap) / spec.rul_cap
    else:
        degradation = 0.0
    rng = np.random.default_rng() if rng is None else rng
    length, width = x.shape
    mask = (rng.random(x.shape) >= spec.sensor_drop).astype(np.float32)
    step = 0
    while step < length:
        if rng.random() < spec.burst_probability:
            end = min(length, step + max(1, int(rng.poisson(spec.burst_length_mean))))
            if rng.random() < .5:
                mask[step:end] = 0
            else:
                columns = rng.choice(width, width // 2, replace=False)
                mask[step:end, columns] = 0
            step = end
        else:
            step += 1
    sigma = spec.base_noise_sigma * (1 + spec.noise_rul_factor * degradation)
    noisy = x + rng.normal(0, sigma, x.shape)
    time = np.sort(np.arange(length, dtype=np.float64) + rng.normal(0, spec.jitter_scale, length))
    time = np.maximum(time, 0) / length
    return (noisy * mask).astype(np.float32), mask, time.astype(np.float32)
