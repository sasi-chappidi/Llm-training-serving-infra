import yaml


def load_yaml(path: str):
    """Load YAML configuration from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)