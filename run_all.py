"""
run_all.py -- End-to-end execution of the Instamart Customer Intelligence & Support Copilot.
Executes the complete system in sequential order:
  Step 1: Data Foundation (Clean, Store & SQL Queries)
  Step 2: Customer Intelligence (Segmentation & Lapse Prediction)
  Step 3: Support Ticket Triage (PII Guardrails & Dynamic Routing)
  Step 4: Policy Knowledge Assistant & Autonomous Refund Agent
  Step 5: Unified One-Input Copilot Verification
"""
import sys
import subprocess
import time

def run_step(step_number, step_title, command):
    print(f"\n{'='*70}\n[STEP {step_number}] STARTING: {step_title}\nCommand: {command}\n{'='*70}")
    t0 = time.time()
    res = subprocess.run(command, shell=True)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n[STEP {step_number}] FAILED: {step_title} with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[STEP {step_number}] COMPLETED: {step_title} in {elapsed:.2f}s\n")

def main():
    print("=" * 70)
    print("INSTAMART CUSTOMER INTELLIGENCE & SUPPORT COPILOT: FULL PIPELINE")
    print("=" * 70)

    # Deterministic Data Generation
    run_step(0, "Deterministic Synthetic Data Generation", "python generate_data.py")

    # Step 1: Data Foundation
    run_step(1, "Data Foundation (Clean, Store, SQL Analytics)", "python 01_data_foundation.py")
    run_step("1.Val", "Data Foundation Checkpoint Validation", "python validate.py")

    # Step 2: Customer Intelligence
    run_step(2, "Customer Intelligence (Segments & Lapse Prediction)", "python 02_customer_intelligence.py")

    # Step 3: Support Ticket Triage
    run_step(3, "Support Ticket Triage (PII Guardrails & Routing)", "python 03_support_triage.py")

    # Step 4: Policy Assistant & Refund Agent
    run_step(4, "Policy Assistant & Refund Agent (RAG & ReAct)", "python 04_policy_refund_agent.py")

    # Step 5: One-Input Copilot Verification
    run_step(5, "Unified One-Input Copilot Verification", 'python 05_one_input_copilot.py --query "Customer C0082 profile and churn risk"')

    print("\n" + "=" * 70)
    print("ALL 5 STEPS COMPLETED SUCCESSFULLY IN ORDER!")
    print("=" * 70)

if __name__ == "__main__":
    main()
