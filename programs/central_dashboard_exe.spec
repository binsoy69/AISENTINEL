# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path


repo_root = Path(os.environ.get("AISENTINEL_REPO_ROOT", Path.cwd())).resolve(strict=False)
runtime_root = repo_root / "runtime"
entrypoint = repo_root / "programs" / "central_dashboard_exe.py"

if not entrypoint.exists():
    raise SystemExit(f"AISENTINEL central dashboard entrypoint not found: {entrypoint}")


a = Analysis(
    [str(entrypoint)],
    pathex=[str(runtime_root)],
    binaries=[],
    datas=[
        (
            str(runtime_root / "central_dashboard" / "central_service" / "templates"),
            "central_dashboard/central_service/templates",
        ),
        (
            str(runtime_root / "central_dashboard" / "central_service" / "static"),
            "central_dashboard/central_service/static",
        ),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AISENTINEL Central Dashboard",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AISENTINEL Central Dashboard",
)
