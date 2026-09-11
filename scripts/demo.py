"""One-command Demo mode（Phase 7）：零 API key 跑通整个 SubPilot。

- 播种 data/schedule.json + 示例手册（复用 app.demo_seed，不重复实现）；
- 以 SUBPILOT_LLM_PROVIDER=demo 启动 uvicorn（确定性 provider，无外部调用）。

用法：
    python scripts/demo.py            # 播种 + 启动 http://127.0.0.1:8000
    python scripts/demo.py --seed-only   # 只播种（Docker / 手动预热用）
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.demo_seed import seed_documents, seed_schedule  # noqa: E402

TZ = "America/New_York"


def main() -> None:
    seed_only = "--seed-only" in sys.argv
    seed_schedule()
    seed_documents()
    if seed_only:
        return
    env = {
        **os.environ,
        "SUBPILOT_LLM_PROVIDER": "demo",
        "SUBPILOT_TIMEZONE": TZ,
    }
    print("Demo mode — open http://127.0.0.1:8000 and ask about the handbook, "
          'e.g. "What is the bathroom pass policy?"')
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"],
        cwd=ROOT, env=env, check=True,
    )


if __name__ == "__main__":
    main()
