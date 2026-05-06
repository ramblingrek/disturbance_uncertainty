# Interpreter Agent Class
# This takes the interpreter value from the field
#  and applies some adjustments to it based on 
#  random, bias, and confidence

# interpreter_agent.py

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict

import numpy as np


@dataclass
class InterpreterAgent:
    """
    Simulated human interpreter that perturbs base probability-of-loss
    values from the InterpreterField according to three mechanisms:

      1) Per-sample consistency noise (Gaussian, mean 0).
      2) Per-interpreter directional bias (Gaussian, mean 0, drawn once).
      3) Per-interpreter confidence adjustment (Gaussian, mean 0, drawn once)
         that either:
             - pulls all probabilities towards 0.5 (low confidence, negative),
             - or pushes them towards extremes (0 or 1) (high confidence, positive).

    Config dictionary structure (example):

        interpreter_cfg = {
            "id": "interpreter_1",
            "consistency_std_pct": 5.0,   # +/- ~5 pct points (1σ) per sample
            "bias_std_pct": 3.0,          # +/- ~3 pct points (1σ) per interpreter
            "confidence_std": 0.5,        # N(0, confidence_std), per interpreter
            "seed": 12345                 # optional for reproducibility
        }
    """

    interpreter_id: str
    consistency_std_pct: float = 0.0   # per-sample noise (pct points)
    bias_std_pct: float = 0.0          # per-interpreter bias (pct points)
    confidence_std: float = 0.0        # N(0, confidence_std), per interpreter
    seed: Optional[int] = None

    # Internal RNG and latent traits
    rng: np.random.Generator = field(init=False, repr=False)
    directional_bias: float = field(init=False)      # in probability units [-0.5, 0.5]
    confidence_scalar: float = field(init=False)     # in [-1, 1]; <0 → towards 0.5, >0 → towards extremes

    def __post_init__(self) -> None:
        # Seed RNG
        if self.seed is None:
            base_seed = int(time.time()) ^ (hash(self.interpreter_id) & 0xFFFFFFFF)
        else:
            base_seed = int(self.seed)
        self.rng = np.random.default_rng(base_seed)

        # 1) Draw a single directional bias for this interpreter (in 0..1 units)
        sigma_bias = max(self.bias_std_pct, 0.0) / 100.0
        if sigma_bias > 0:
            bias = self.rng.normal(loc=0.0, scale=sigma_bias)
        else:
            bias = 0.0
        # Clip to a reasonable range
        self.directional_bias = float(np.clip(bias, -0.5, 0.5))

        # 2) Draw a single confidence scalar for this interpreter
        #    - negative: move all probs towards 0.5
        #    - positive: move all probs towards extremes (0 or 1)
        # if self.confidence_std > 0:
        #     c_raw = self.rng.normal(loc=0.0, scale=self.confidence_std)
        #     # Clamp into [-1, 1] so effect is interpretable as a fraction of the distance
        #     self.confidence_scalar = float(np.clip(c_raw, -1.0, 1.0))
        # else:
        #     self.confidence_scalar = 0.0

        # changing this so the confidence scalar is just set by thte 
        #  user rather than drawn. 
        if self.confidence_std != 0:
            self.confidence_scalar = self.confidence_std
        else:
            self.confidence_scalar = 0.0
        


    
    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def transform_probabilities(
        self,
        probs_pct: Sequence[float],
    ) -> np.ndarray:
        """
        Take a sequence of base probabilities from the InterpreterField
        (0–100%) and return a NumPy array of adjusted probabilities (0–100%)
        after applying:

          1) Per-sample consistency noise.
          2) Interpreter-level directional bias.
          3) Interpreter-level confidence adjustment.

        All math is done in [0,1]; outputs are clipped to [0,1] and
        returned as 0–100%.
        """
        # Convert to 0..1
        probs = np.asarray(probs_pct, dtype=float) / 100.0

        # 1) Per-sample consistency noise
        sigma_consistency = max(self.consistency_std_pct, 0.0) / 100.0
        if sigma_consistency > 0:
            noise = self.rng.normal(loc=0.0, scale=sigma_consistency, size=probs.shape)
        else:
            noise = 0.0
        p1 = probs + noise

        # 2) Add interpreter-level directional bias (same for all samples)
        p2 = p1 + self.directional_bias

        # 3) Interpreter-level confidence adjustment (same scalar for all samples)
        p3 = p2.copy()
        c = self.confidence_scalar

        if c < 0:  # less confident → move towards 0.5
            alpha = abs(c)  # 0..1
            p3 = p2 + alpha * (0.5 - p2)

        elif c > 0:  # more confident → push to extremes
            alpha = c  # 0..1
            # Below 0.5 → move towards 0
            mask_low = p2 < 0.5
            # Above 0.5 → move towards 1
            mask_high = p2 > 0.5

            if np.any(mask_low):
                p3[mask_low] = p2[mask_low] + alpha * (0.0 - p2[mask_low])

            if np.any(mask_high):
                p3[mask_high] = p2[mask_high] + alpha * (1.0 - p2[mask_high])
            # Exactly 0.5 remains unchanged

        # Clip and return 0–100
        p3 = np.clip(p3, 0.0, 1.0)
        return p3 * 100.0



@dataclass
class InterpreterGroup:
    """
    Container for multiple InterpreterAgent objects.
    """

    interpreters: Dict[str, InterpreterAgent] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Dict) -> "InterpreterGroup":
        """
        Build from a config like:

            cfg = {
              "interpreters": [
                 {
                   "id": "int_1",
                   "consistency_std_pct": 5.0,
                   "bias_std_pct": 3.0,
                   "confidence_std": 0.5,
                   "seed": 123,
                 },
                 {
                   "id": "int_2",
                   "consistency_std_pct": 10.0,
                   "bias_std_pct": 0.0,
                   "confidence_std": 0.2,
                 }
              ]
            }
        """
        interp_dict: Dict[str, InterpreterAgent] = {}
        for icfg in cfg.get("interpreters", []):
            agent = InterpreterAgent(
                interpreter_id=icfg.get("id", "unnamed"),
                consistency_std_pct=icfg.get("consistency_std_pct", 0.0),
                bias_std_pct=icfg.get("bias_std_pct", 0.0),
                confidence_std=icfg.get("confidence_std", 0.0),
                seed=icfg.get("seed", None),
            )
            interp_dict[agent.interpreter_id] = agent
        return cls(interpreters=interp_dict)

    def get(self, interpreter_id: str) -> InterpreterAgent:
        return self.interpreters[interpreter_id]

    def apply_to_all(
        self,
        probs_pct: Sequence[float],
    ) -> Dict[str, np.ndarray]:
        """
        Apply each interpreter to the same base probabilities, returning a
        dict mapping interpreter_id -> adjusted probs.
        """
        out: Dict[str, np.ndarray] = {}
        for iid, agent in self.interpreters.items():
            out[iid] = agent.transform_probabilities(probs_pct)
        return out