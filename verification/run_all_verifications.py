"""
Master Verification Runner
Executes all verification audit scripts to reproduce all numerical claims
and logs in the verification/ directory.
"""

import subprocess
import sys
import os

SCRIPTS = [
    "verification/verify_detection.py",
    "verification/verify_ncap.py",
    "verification/verify_hic.py",
    "verification/verify_thermo.py",
    "verification/verify_dynamics.py",
    "verification/verify_ml_models.py",
    "verification/verify_driver_monitoring.py"
]

def main():
    print("=" * 80)
    print("  ADAS PEDESTRIAN AEB: MASTER VERIFICATION SUITE")
    print("=" * 80)
    failed = []

    for script in SCRIPTS:
        print(f"\n---> Running {script}...")
        res = subprocess.run([sys.executable, script], env={**os.environ, "PYTHONPATH": "."})
        if res.returncode != 0:
            print(f"FAILED: {script}")
            failed.append(script)
        else:
            print(f"PASSED: {script}")

    print("\n" + "=" * 80)
    if not failed:
        print("  ALL VERIFICATION SCRIPTS EXECUTED SUCCESSFULLY!")
    else:
        print(f"  FAILED SCRIPTS ({len(failed)}): {failed}")
    print("=" * 80)

if __name__ == "__main__":
    main()
