"""
run_all.py -- End-to-end execution of the Instamart Customer Intelligence & Support Copilot.
Runs Part A, Part B, Part C, and Part D sequentially and verifies all outputs.
"""
import sys
import subprocess
import time

def run_step(step_name, command):
    print(f"\n{'='*70}\nSTARTING: {step_name}\nCommand: {command}\n{'='*70}")
    t0 = time.time()
    res = subprocess.run(command, shell=True)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"FAILED: {step_name} with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"COMPLETED: {step_name} in {elapsed:.2f}s\n")

def main():
    print("Executing complete Instamart Copilot pipeline...")
    run_step("Generate Synthetic Data", "python generate_data.py")
    run_step("Part A - Data Foundation", "python src/part_a.py")
    run_step("Part A - Checkpoint Validation", "python validate.py")
    run_step("Part B - Customer Segments & Lapse Prediction", "python src/part_b.py")
    run_step("Part C - Support Ticket Triage", "python src/part_c.py")
    run_step("Part D - Policy Assistant & Refund Agent", "python src/part_d.py")
    print("\n" + "="*70)
    print("ALL PARTS COMPLETED SUCCESSFULLY!")
    print("="*70)

if __name__ == "__main__":
    main()
