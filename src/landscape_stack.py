# landscape_stack.py

from __future__ import annotations

import json
import uuid
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any

from disturbance_landscape import DisturbanceLandscape
from sensor_field import SensorField
from interpreter_field import InterpreterField

# Global registry of stacks
LANDSCAPE_REGISTRY: Dict[str, Dict[str, Any]] = {}


def make_stack_id(cfg: dict,
                  prefix: str = "DL",
                  strategy: str = "hash") -> str:
    """
    Generate a unique ID for a landscape stack.

    strategy = "hash"  → deterministic hash of config
    strategy = "uuid"  → random UUID
    """
    if strategy == "uuid":
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    # default: deterministic hash based on the config contents
    cfg_str = json.dumps(cfg, sort_keys=True)
    digest = hashlib.sha1(cfg_str.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{digest}"


@dataclass
class LandscapeStack:
    """
    A single disturbance landscape stack:

      - base_landscape: DisturbanceLandscape
      - sensor_fields:  dict[name -> SensorField]
      - interpreter_field: InterpreterField
      - stack_id: unique identifier (hash of config or UUID)
      - config: the original stack config dict
    """
    stack_id: str
    base_landscape: DisturbanceLandscape
    sensor_fields: Dict[str, SensorField]
    interpreter_field: InterpreterField
    config: dict = field(repr=False)

    # ----------------------- Construction from config -----------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "LandscapeStack":
        """
        Build a LandscapeStack from a configuration dictionary.

        Expected structure (Python dict):

            {
              "id_prefix": "DL",
              "id_strategy": "hash",
              "explicit_id": null,
              "tif_path": "...",
              "disturbance_config": { ... },   # passed to DisturbanceLandscape
              "sensors": [
                {"name": "Landsat_OLI", "config": {...}},
                {"name": "Sentinel2_MSI", "config": {...}},
                ...
              ],
              "interpreter": {
                "name": "DefaultInterpreter",
                "config": {...}
              }
            }
        """
        # 1. ID selection
        id_prefix   = cfg.get("id_prefix", "DL")
        id_strategy = cfg.get("id_strategy", "hash")
        explicit_id = cfg.get("explicit_id")

        if explicit_id is not None:
            stack_id = explicit_id
        else:
            stack_id = make_stack_id(cfg, prefix=id_prefix, strategy=id_strategy)

        # 2. Base disturbance landscape
        tif_path = cfg["tif_path"]
        dist_cfg = cfg["disturbance_config"]

        base_landscape = DisturbanceLandscape(
            tif_path=tif_path,
            config=dist_cfg,
        )

        # 3. Sensor landscapes (arbitrary number)
        sensors_cfg = cfg.get("sensors", [])
        sensor_fields: Dict[str, SensorField] = {}

        for s_cfg in sensors_cfg:
            s_name = s_cfg["name"]
            s_conf = s_cfg["config"]
            sensor_fields[s_name] = SensorField(
                name=s_name,
                truth=base_landscape,
                config=s_conf,
            )

        # 4. Interpreter field (single)
        interp_cfg = cfg.get("interpreter", {})
        interp_name = interp_cfg.get("name", "Interpreter")
        interp_conf = interp_cfg.get("config", {})

        interpreter_field = InterpreterField(
            name=interp_name,
            truth=base_landscape,   # currently designed to consume DisturbanceLandscape
            config=interp_conf,
        )

        # 5. Construct the stack object
        stack = cls(
            stack_id=stack_id,
            base_landscape=base_landscape,
            sensor_fields=sensor_fields,
            interpreter_field=interpreter_field,
            config=cfg,
        )

        # 6. Register this stack in the global registry
        stack.register_in_registry()

        return stack

    # ------------------------ Registry / metadata ---------------------------

    def register_in_registry(self) -> None:
        """
        Store a summary of this stack in the global LANDSCAPE_REGISTRY.
        This keeps a lightweight metadata record keyed by stack_id.
        """

        # Pull whatever "params" attrs you have; if not present, store None.
        base_meta = getattr(self.base_landscape, "config", None)

        sensors_meta = {
            name: getattr(sf, "config", None)
            for name, sf in self.sensor_fields.items()
        }

        interpreter_meta = getattr(self.interpreter_field, "config", None)

        LANDSCAPE_REGISTRY[self.stack_id] = {
            "stack_id": self.stack_id,
            "tif_path": getattr(self.base_landscape, "tif_path", None),
            "base_config": base_meta,
            "sensor_configs": sensors_meta,
            "interpreter_config": interpreter_meta,
            "full_stack_config": self.config,
        }

    # Convenience accessors

    def get_sensor(self, name: str) -> SensorField:
        return self.sensor_fields[name]

    def list_sensors(self) -> list[str]:
        return list(self.sensor_fields.keys())

    def to_metadata(self) -> dict:
        """
        Return a lightweight metadata dict for quick inspection or saving.
        """
        return LANDSCAPE_REGISTRY.get(self.stack_id, {}).copy()