from pathlib import Path

import pytest

from analytics_agent.code_sandbox import CodeSandbox, UnsafeCodeError


def test_sandbox_save_current_plot_creates_artifact(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data.csv"
    dataset_path.write_text("x,y\n1,2\n2,4\n", encoding="utf-8")
    sandbox = CodeSandbox(run_dir=tmp_path / "run", timeout_seconds=10)

    result = sandbox.execute(
        """
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(DATA_PATH)
plt.figure()
plt.plot(df['x'], df['y'])
save_current_plot('line_chart.png')
print('График сохранен')
""".strip(),
        dataset_path,
    )

    assert result.ok
    assert "График сохранен" in result.stdout
    assert [artifact.name for artifact in result.artifacts] == ["line_chart.png"]


def test_sandbox_blocks_reassignment_of_data_path(tmp_path: Path) -> None:
    dataset_path = tmp_path / "data.csv"
    dataset_path.write_text("x\n1\n", encoding="utf-8")
    sandbox = CodeSandbox(run_dir=tmp_path / "run", timeout_seconds=10)

    with pytest.raises(UnsafeCodeError, match="DATA_PATH"):
        sandbox.execute("DATA_PATH = '/tmp/other.csv'", dataset_path)