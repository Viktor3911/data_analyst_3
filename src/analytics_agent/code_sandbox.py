import ast
import logging
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path


ALLOWED_IMPORT_ROOTS = {"pandas", "numpy", "matplotlib", "seaborn", "scipy", "sklearn", "math", "statistics", "json"}
BLOCKED_CALLS = {
    "open",
    "eval",
    "exec",
    "compile",
    "input",
    "__import__",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "breakpoint",
    "help",
    "exit",
    "quit",
}
BLOCKED_ATTRIBUTES = {
    "environ",
    "system",
    "popen",
    "remove",
    "unlink",
    "rmdir",
    "rename",
    "replace",
    "chmod",
    "chown",
    "mkdir",
    "makedirs",
    "write_text",
    "write_bytes",
    "read_text",
    "read_bytes",
}
PROTECTED_NAMES = {"DATA_PATH", "DATASET_FORMAT", "ARTIFACT_DIR", "save_current_plot"}


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodeExecutionResult:
    stdout: str
    stderr: str
    artifacts: list[Path]
    return_code: int

    @property
    def ok(self) -> bool:
        return self.return_code == 0


class UnsafeCodeError(ValueError):
    pass


class CodeSandbox:
    def __init__(self, run_dir: Path, timeout_seconds: int) -> None:
        self.run_dir = run_dir
        self.timeout_seconds = timeout_seconds
        self.artifact_dir = run_dir / "artifacts"

    def execute(self, code: str, dataset_path: Path) -> CodeExecutionResult:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._validate(code)
        logger.info("Sandbox execution started: dataset=%s run_dir=%s", dataset_path.name, self.run_dir)

        script_path = self.run_dir / "agent_step.py"
        script_path.write_text(self._build_script(code, dataset_path), encoding="utf-8")

        try:
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=self.run_dir,
                env=self._safe_environment(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            logger.warning("Sandbox execution timed out after %s seconds", self.timeout_seconds)
            return CodeExecutionResult(
                stdout=error.stdout or "",
                stderr=f"Python tool timed out after {self.timeout_seconds} seconds.",
                artifacts=self._list_artifacts(),
                return_code=124,
            )

        result = CodeExecutionResult(
            stdout=completed.stdout[-6000:],
            stderr=completed.stderr[-3000:],
            artifacts=self._list_artifacts(),
            return_code=completed.returncode,
        )
        logger.info(
            "Sandbox execution finished: return_code=%s artifacts=%s",
            result.return_code,
            [artifact.name for artifact in result.artifacts],
        )
        return result

    def _validate(self, code: str) -> None:
        try:
            tree = ast.parse(code)
        except SyntaxError as error:
            raise UnsafeCodeError(f"Generated code has syntax error: {error}") from error

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._validate_import(node)
            if isinstance(node, ast.Call):
                self._validate_call(node)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                self._validate_assignment(node)
            if isinstance(node, ast.Attribute) and node.attr in BLOCKED_ATTRIBUTES:
                raise UnsafeCodeError(f"Blocked unsafe attribute access: {node.attr}")

    def _validate_import(self, node: ast.Import | ast.ImportFrom) -> None:
        module_names = []
        if isinstance(node, ast.Import):
            module_names = [alias.name for alias in node.names]
        elif node.module:
            module_names = [node.module]

        for module_name in module_names:
            root = module_name.split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                raise UnsafeCodeError(f"Blocked unsafe import: {module_name}")

    def _validate_call(self, node: ast.Call) -> None:
        function = node.func
        if isinstance(function, ast.Name) and function.id in BLOCKED_CALLS:
            raise UnsafeCodeError(f"Blocked unsafe function call: {function.id}")
        if isinstance(function, ast.Attribute) and function.attr in BLOCKED_ATTRIBUTES:
            raise UnsafeCodeError(f"Blocked unsafe method call: {function.attr}")

    def _validate_assignment(self, node: ast.Assign | ast.AnnAssign | ast.AugAssign) -> None:
        targets = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        else:
            targets = [node.target]

        for target in targets:
            for name in self._assigned_names(target):
                if name in PROTECTED_NAMES:
                    raise UnsafeCodeError(f"Blocked reassignment of protected name: {name}")

    def _assigned_names(self, target: ast.AST) -> list[str]:
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            names: list[str] = []
            for item in target.elts:
                names.extend(self._assigned_names(item))
            return names
        return []

    def _build_script(self, code: str, dataset_path: Path) -> str:
        header = f'''
import matplotlib
matplotlib.use("Agg")

DATA_PATH = r"{dataset_path}"
DATASET_FORMAT = "{dataset_path.suffix.lower().lstrip('.')}"
ARTIFACT_DIR = r"{self.artifact_dir}"

from pathlib import Path as _SafePath
import re as _safe_re

def save_current_plot(file_name="chart.png"):
    import matplotlib.pyplot as _safe_plt

    safe_name = _safe_re.sub(r"[^A-Za-z0-9_.-]", "_", str(file_name)).strip("._")
    if not safe_name:
        safe_name = "chart.png"
    if not safe_name.lower().endswith((".png", ".jpg", ".jpeg")):
        safe_name += ".png"
    target_path = _SafePath(ARTIFACT_DIR) / safe_name
    _safe_plt.savefig(target_path, bbox_inches="tight")
    print(f"Artifact saved: {{target_path.name}}")
    return str(target_path)
'''
        return textwrap.dedent(header).strip() + "\n\n" + code.strip() + "\n"

    def _safe_environment(self) -> dict[str, str]:
        keys_to_keep = ["PATH", "SYSTEMROOT", "TEMP", "TMP"]
        env = {key: value for key, value in os.environ.items() if key.upper() in keys_to_keep}
        env["PYTHONIOENCODING"] = "utf-8"
        env["MPLBACKEND"] = "Agg"
        mpl_config_dir = self.run_dir / "mpl_config"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        env["MPLCONFIGDIR"] = str(mpl_config_dir)
        return env

    def _list_artifacts(self) -> list[Path]:
        if not self.artifact_dir.exists():
            return []
        return sorted(path for path in self.artifact_dir.iterdir() if path.is_file())
