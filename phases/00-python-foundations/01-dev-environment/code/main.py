import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_PACKAGES = ["numpy", "pandas", "matplotlib", "scikit-learn", "scipy"]
MINIMUM_PYTHON = (3, 12)


def python_report():
    version = sys.version_info
    return {
        "version": f"{version.major}.{version.minor}.{version.micro}",
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "platform": f"{platform.system()} {platform.machine()}",
        "meets_minimum": (version.major, version.minor) >= MINIMUM_PYTHON,
    }


def virtual_env_report():
    inside_venv = sys.prefix != sys.base_prefix
    return {
        "inside_venv": inside_venv,
        "env_prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
    }


def package_report(package_names):
    report = {}
    for name in package_names:
        try:
            report[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report[name] = None
    return report


def find_project_root(start):
    for folder in [start, *start.parents]:
        if (folder / "pyproject.toml").exists():
            return folder
    return None


def read_pinned_requirements(requirements_path):
    pins = {}
    if not requirements_path.exists():
        return pins
    for line in requirements_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        pins[name.strip().lower()] = version.strip()
    return pins


def uv_version():
    uv_path = shutil.which("uv")
    if uv_path is None:
        return None
    result = subprocess.run([uv_path, "--version"], capture_output=True, text=True, check=False)
    return result.stdout.strip() or None


def print_section(title):
    print()
    print(title)
    print("-" * len(title))


print_section("Python interpreter")
python_info = python_report()
for key, value in python_info.items():
    print(f"  {key:<15} {value}")

print_section("Virtual environment")
venv_info = virtual_env_report()
for key, value in venv_info.items():
    print(f"  {key:<15} {value}")
if not venv_info["inside_venv"]:
    print("  warning: you are using the system Python. Run scripts with `uv run`.")

print_section("Project")
project_root = find_project_root(Path(__file__).resolve().parent)
print(f"  project_root    {project_root}")
print(f"  uv              {uv_version()}")

print_section("Installed packages vs requirements.txt")
installed = package_report(REQUIRED_PACKAGES)
pinned = read_pinned_requirements(project_root / "requirements.txt") if project_root else {}
all_ok = True
for name, version in installed.items():
    expected = pinned.get(name)
    if version is None:
        status = "MISSING"
        all_ok = False
    elif expected is not None and expected != version:
        status = f"differs from pin {expected}"
    else:
        status = "ok"
    print(f"  {name:<14} {str(version):<10} {status}")

print_section("Use It: a real computation with the installed stack")
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, size=(50, 1))
y = 2.0 * x[:, 0] + 1.0 + rng.normal(0, 0.1, size=50)
model = LinearRegression().fit(x, y)
print(f"  fitted slope     {model.coef_[0]:.3f} (true 2.0)")
print(f"  fitted intercept {model.intercept_:.3f} (true 1.0)")

print()
ready = all_ok and python_info["meets_minimum"] and venv_info["inside_venv"]
print("Environment ready." if ready else "Environment needs attention (see above).")
