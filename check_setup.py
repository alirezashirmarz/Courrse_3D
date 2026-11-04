"""Run once after installing:  python check_setup.py"""
import importlib, sys

print("Python", sys.version.split()[0])
ok = True
for name, need in [("numpy", True), ("scipy", True), ("matplotlib", True), ("PIL", True), ("open3d", True),
                   ("laspy", False), ("DracoPy", False), ("sklearn", False)]:
    try:
        m = importlib.import_module(name)
        print(f"  [ok]       {name:12s} {getattr(m, '__version__', '')}")
    except Exception:
        print(f"  [{'MISSING' if need else 'optional'}] {name}")
        ok &= not need
print("\nAll required packages found - you're ready!" if ok else "\nInstall the missing packages: pip install -r requirements.txt")
