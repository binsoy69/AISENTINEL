#!/usr/bin/env python3
"""Build the Windows one-folder editable demo dashboard EXE with PyInstaller."""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "programs" / "editable_dashboard_exe.spec"
DIST_DIR = REPO_ROOT / "dist" / "AISENTINEL Editable Dashboard"
EXTERNAL_CONFIG_TEMPLATE = REPO_ROOT / "programs" / "editable_dashboard_exe.ini"


def main() -> int:
    try:
        import PyInstaller.__main__  # noqa: F401
    except ImportError:
        print("PyInstaller is not installed. Run: python -m pip install pyinstaller")
        return 1

    env = os.environ.copy()
    env["AISENTINEL_REPO_ROOT"] = str(REPO_ROOT)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--clean",
            "--noconfirm",
            str(SPEC_PATH),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        return result.returncode

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EXTERNAL_CONFIG_TEMPLATE, DIST_DIR / "editable_dashboard.ini")
    (DIST_DIR / "data" / "editable_dashboard").mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 78)
    print("  AISENTINEL Editable Dashboard EXE build complete")
    print(f"  Folder : {DIST_DIR}")
    print(f"  EXE    : {DIST_DIR / 'AISENTINEL Editable Dashboard.exe'}")
    print(f"  Config : {DIST_DIR / 'editable_dashboard.ini'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
