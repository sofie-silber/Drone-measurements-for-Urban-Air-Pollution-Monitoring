#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launch the EXISTING calibration script in working Anaconda from Spyder (F5).

No change is made to Spyder's interpreter or to the calibration script.
No packages are installed. PyTorch is imported only by the Anaconda process.
"""

from pathlib import Path
import os
import subprocess

# These two paths come from the user's working Anaconda installation and traceback.
ANACONDA_PYTHON = Path(r"C:\Users\xy-84\anaconda3\python.exe")
CALIBRATION_SCRIPT = Path(r"C:\Users\xy-84\Downloads\unbenannt0.py")


def run_streamed(command, working_dir, environment):
    """Show output as it arrives in the Spyder IPython console."""
    with subprocess.Popen(
        command,
        cwd=str(working_dir),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        result = process.wait()
    if result != 0:
        raise RuntimeError(
            f"Anaconda process ended with exit code {result}. "
            "The error above comes from the Anaconda process."
        )


def main():
    if not ANACONDA_PYTHON.is_file():
        raise FileNotFoundError(
            "The expected Anaconda Python executable was not found: "
            f"{ANACONDA_PYTHON}"
        )
    if not CALIBRATION_SCRIPT.is_file():
        raise FileNotFoundError(
            "The calibration file shown in your traceback was not found: "
            f"{CALIBRATION_SCRIPT}\n"
            "Edit only CALIBRATION_SCRIPT at the top of this launcher if you renamed it."
        )

    # Do not inherit Spyder runtime's Python import paths in the child process.
    env = os.environ.copy()
    for var in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE"):
        env.pop(var, None)
    root = ANACONDA_PYTHON.parent
    env["PATH"] = os.pathsep.join(
        [str(root), str(root / "Library" / "bin"),
         str(root / "Scripts"), env.get("PATH", "")]
    )
    env["PYTHONIOENCODING"] = "utf-8"

    print("Using existing Anaconda Python:", ANACONDA_PYTHON, flush=True)
    print("Running unchanged calibration file:", CALIBRATION_SCRIPT, flush=True)
    print("Checking PyTorch import from Anaconda...", flush=True)
    run_streamed(
        [str(ANACONDA_PYTHON), "-u", "-c",
         "import sys, torch; print('Python:', sys.executable); "
         "print('PyTorch:', torch.__version__); "
         "print('Torch file:', torch.__file__)"],
        working_dir=CALIBRATION_SCRIPT.parent,
        environment=env,
    )
    print("Running Mandria calibration...", flush=True)
    run_streamed(
        [str(ANACONDA_PYTHON), "-u", str(CALIBRATION_SCRIPT)],
        working_dir=CALIBRATION_SCRIPT.parent,
        environment=env,
    )
    print("Calibration completed.", flush=True)


if __name__ == "__main__":
    main()
