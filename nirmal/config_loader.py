"""
Configuration loader for YAML-based pipeline configuration.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any


logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load, normalize, and validate YAML configuration files."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}

            logger.info(f"Configuration loaded from {self.config_path}")

            self._normalize_config()
            self._validate_config()

            return self.config

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")
            raise

    # --------------------------------------------------
    # 🔧 NORMALIZATION (KEY UPGRADE)
    # --------------------------------------------------
    def _normalize_config(self) -> None:
        """
        Normalize configuration to support flexible YAML formats.
        Converts boolean flags into structured dictionaries.
        """
        pipeline = self.config.get("pipeline", {})
        cleaning = pipeline.get("cleaning", {})

        # normalize remove_duplicates
        cleaning["remove_duplicates"] = self._normalize_flag(
            cleaning.get("remove_duplicates")
        )

        # normalize strip_whitespace
        cleaning["strip_whitespace"] = self._normalize_flag(
            cleaning.get("strip_whitespace")
        )

        pipeline["cleaning"] = cleaning
        self.config["pipeline"] = pipeline

        logger.debug("Configuration normalized successfully")

    def _normalize_flag(self, value: Any) -> Dict[str, bool]:
        """
        Convert boolean or dict flag into standardized dict format.
        """
        if isinstance(value, bool):
            return {"enabled": value}

        if isinstance(value, dict):
            return {"enabled": bool(value.get("enabled", False))}

        return {"enabled": False}

  
    def _validate_config(self) -> None:
        """Validate configuration structure."""
        if "pipeline" not in self.config:
            raise ValueError("Missing required section: 'pipeline'")

        logger.info("Configuration validation completed")


    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        Example: pipeline.cleaning.remove_duplicates.enabled
        """
        keys = key.split(".")
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
