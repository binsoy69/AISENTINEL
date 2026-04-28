#!/usr/bin/env python3
"""Interactive file/model selection helpers for Pi test programs."""

from __future__ import annotations

import os
from pathlib import Path

from front_node_pi_model_paths import (
    HAND_MODEL_CANDIDATES,
    OBJECT_MODEL_CANDIDATES,
    POSE_MODEL_CANDIDATES,
    REPO_ROOT,
    SCRIPT_DIR,
)


VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm")
MODEL_SUFFIXES = (".hef",)
FILE_DIALOG_REQUEST = "__AISENTINEL_FILE_DIALOG__"


def display_path(value):
    path = Path(value)
    try:
        resolved = path.resolve()
        return str(resolved.relative_to(REPO_ROOT))
    except (OSError, ValueError):
        return str(value)


def rglob_files(root, suffixes):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        ],
        key=lambda path: str(path).lower(),
    )


def dedupe_paths(paths):
    seen = set()
    unique = []
    for value in paths:
        path = Path(value)
        try:
            key = str(path.resolve()).lower() if path.exists() else str(path).lower()
        except OSError:
            key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def discover_videos():
    found = []
    roots = (
        REPO_ROOT / "test-videos",
        REPO_ROOT / "runtime" / "front_node_pi" / "data" / "session_uploads",
        SCRIPT_DIR,
        REPO_ROOT / "tests",
        REPO_ROOT,
    )
    for root in roots:
        found.extend(rglob_files(root, VIDEO_SUFFIXES))
    return dedupe_paths(found)


def discover_hef_models(extra_candidates=()):
    found = list(extra_candidates)
    found.extend(rglob_files(REPO_ROOT / "models", MODEL_SUFFIXES))
    found.extend(rglob_files(REPO_ROOT, MODEL_SUFFIXES))
    return [path for path in dedupe_paths(found) if Path(path).is_file()]


def discover_pose_models():
    models = list(POSE_MODEL_CANDIDATES)
    models.extend(path for path in discover_hef_models() if "pose" in path.name.lower())
    return [path for path in dedupe_paths(models) if Path(path).is_file()]


def discover_hand_models():
    models = list(HAND_MODEL_CANDIDATES)
    models.extend(path for path in discover_hef_models() if "hand" in path.name.lower())
    return [path for path in dedupe_paths(models) if Path(path).is_file()]


def discover_object_models():
    models = list(OBJECT_MODEL_CANDIDATES)
    models.extend(
        path
        for path in discover_hef_models()
        if "pose" not in path.name.lower() and "hand" not in path.name.lower()
    )
    return [path for path in dedupe_paths(models) if Path(path).is_file()]


def _input_choice(prompt):
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"


def prompt_numbered_path(title, options, manual_prompt, allow_dialog=False):
    options = list(options)

    print()
    print(title)
    if options:
        for idx, option in enumerate(options, start=1):
            default_marker = " [default]" if idx == 1 else ""
            print(f"  {idx}. {display_path(option)}{default_marker}")
    else:
        print("  No repository matches found.")

    print("  M. Enter path manually")
    if allow_dialog:
        print("  F. Open file dialog")
    print("  Q. Quit")

    default_hint = " [1]" if options else ""
    valid_choices = (
        "a listed number, M, F, or Q"
        if allow_dialog
        else "a listed number, M, or Q"
    )
    while True:
        choice = _input_choice(f"Select option{default_hint}: ")
        if not choice and options:
            return str(options[0])

        lowered = choice.lower()
        if lowered in ("q", "quit", "exit"):
            return None
        if allow_dialog and lowered in ("f", "file", "dialog"):
            return FILE_DIALOG_REQUEST
        if lowered in ("m", "manual", "path", "p"):
            value = _input_choice(f"{manual_prompt}: ").strip().strip('"').strip("'")
            if value:
                return value
            print("  Please enter a path.")
            continue

        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(options):
                return str(options[index - 1])

        print(f"  Choose {valid_choices}.")


def prompt_existing_file(title, options, manual_prompt, allow_dialog=False):
    while True:
        selected = prompt_numbered_path(title, options, manual_prompt, allow_dialog)
        if selected is None or selected == FILE_DIALOG_REQUEST:
            return selected
        if os.path.isfile(selected):
            return selected
        print(f"[ERROR] File not found: {selected}")


def select_video(video_arg=None, dialog_func=None):
    if video_arg:
        return video_arg

    selected = prompt_existing_file(
        "Choose video",
        discover_videos(),
        "Enter video file path",
        allow_dialog=dialog_func is not None,
    )
    if selected == FILE_DIALOG_REQUEST:
        return dialog_func() if dialog_func is not None else None
    return selected


def select_video_dialog():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        print(f"[WARN] File dialog is unavailable: {exc}")
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select video file",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.webm"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path or None


def select_input_source(input_arg=None, dialog_func=None):
    if input_arg:
        return input_arg

    options = ["0"]
    options.extend(discover_videos())
    selected = prompt_numbered_path(
        "Choose input source",
        options,
        "Enter camera index, /dev/video* path, or video file path",
        allow_dialog=dialog_func is not None,
    )
    if selected == FILE_DIALOG_REQUEST:
        return dialog_func() if dialog_func is not None else None
    return selected


def select_pose_model(model_arg=None):
    if model_arg:
        return model_arg
    return prompt_existing_file(
        "Choose pose HEF model",
        discover_pose_models(),
        "Enter pose HEF model path",
    )


def select_hand_model(model_arg=None):
    if model_arg:
        return model_arg
    return prompt_existing_file(
        "Choose hand HEF model",
        discover_hand_models(),
        "Enter hand HEF model path",
    )


def select_object_model(model_arg=None):
    if model_arg:
        return model_arg
    return prompt_existing_file(
        "Choose phone / cheat-sheet object HEF model",
        discover_object_models(),
        "Enter object HEF model path",
    )


def select_generic_hef(model_arg=None, extra_candidates=()):
    if model_arg:
        return model_arg
    return prompt_existing_file(
        "Choose HEF model",
        discover_hef_models(extra_candidates),
        "Enter HEF model path",
    )
