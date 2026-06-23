"""
install_dependencies.py
━━━━━━━━━━━━━━━━━━━━━━━
Installs all Python packages required by the NETWATCH-IDS application.
Run this script once before launching ids_simplified.py.

Usage:
    python install_dependencies.py

Requirements:
    Python 3.8+  |  pip  |  internet connection
"""

import subprocess
import sys

# ── All required packages ─────────────────────────────────────
# Format: (pip_package_name, import_name_for_verification)
DEPENDENCIES = [
    ("numpy",               "numpy"),
    ("pandas",              "pandas"),
    ("matplotlib",          "matplotlib"),
    ("scikit-learn",        "sklearn"),
    ("tk",                  None),       # Tkinter — usually bundled with Python
]

# ── Helpers ───────────────────────────────────────────────────
def pip_install(package: str) -> bool:
    """Run pip install for a single package. Returns True on success."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", package],
        capture_output=True, text=True
    )
    return result.returncode == 0

def check_import(module: str) -> bool:
    """Try importing a module to confirm it is available."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True
    )
    return result.returncode == 0

# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 54)
    print("  NETWATCH-IDS  —  Dependency Installer")
    print("=" * 54)
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Interpreter: {sys.executable}")
    print("=" * 54)

    failed = []

    for pip_name, import_name in DEPENDENCIES:

        # Special case: tkinter ships with Python, not via pip
        if pip_name == "tk":
            if check_import("tkinter"):
                print(f"  [OK]     tkinter  (bundled with Python)")
            else:
                print(f"  [WARN]   tkinter not found.")
                print("           On Ubuntu/Debian: sudo apt install python3-tk")
                print("           On Fedora:         sudo dnf install python3-tkinter")
                print("           On macOS:          reinstall Python from python.org")
                failed.append("tkinter")
            continue

        print(f"  Installing {pip_name}...", end=" ", flush=True)
        ok = pip_install(pip_name)

        if ok and import_name and check_import(import_name):
            print("OK")
        elif ok:
            print("OK (install reported success)")
        else:
            print("FAILED")
            failed.append(pip_name)

    # ── Summary ───────────────────────────────────────────────
    print("=" * 54)
    if not failed:
        print("  All dependencies installed successfully.")
        print("  You can now run:  python ids_simplified.py")
    else:
        print("  The following packages could not be installed:")
        for pkg in failed:
            print(f"    • {pkg}")
        print("\n  Try installing them manually:")
        for pkg in failed:
            if pkg != "tkinter":
                print(f"    pip install {pkg}")
    print("=" * 54)

if __name__ == "__main__":
    main()
