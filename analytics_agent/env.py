from pathlib import Path


class DotEnvLoader:

    def __init__(self, env_path: Path) -> None:
        self.env_path = env_path

    def load(self) -> None:
        import os

        if not self.env_path.exists():
            return

        with self.env_path.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
