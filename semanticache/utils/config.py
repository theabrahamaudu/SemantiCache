from pathlib import Path
import yaml
from typing import Any, Dict


class Config:
    def __init__(
            self,
            yaml_path: str = "./sem_config/sem_config.yaml",
            cache_path: str = "./sem_cache",
            config_path: str = "./sem_config",
        ) -> None:

        self.yaml_path = yaml_path
        self.cache_path = cache_path
        self.config_path = config_path
    def load_config(self) -> Dict[str, Any]:
        try:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                config: dict = yaml.safe_load(f)

            return config
        except FileNotFoundError:
            self.__create_directories()
            self.__create_config_file()
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                config: dict = yaml.safe_load(f)

                return config

    def __create_directories(self):
        """Create the cache and config directories if they don't exist."""
        cache_dir = Path(self.cache_path)
        config_dir = Path(self.config_path)

        cache_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)

    def __create_config_file(self):
        """Create the config.yaml file with predefined content."""
        config_path = Path(self.yaml_path)

        if not config_path.exists():
            config_content = """cache:
                                    cache_path: ./sem_cache
                                    cache_name: sem_cache_index
                                    cache_size: 100
                                    cache_ttl: 3600
                                    similarity_threshold: 0.1
                                    """
            config_path.write_text(config_content, encoding="utf-8")