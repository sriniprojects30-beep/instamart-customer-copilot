"""
03_support_triage.py -- Step 3: Support Ticket Triage with Privacy Guardrails
Performs multi-pattern PII sanitization (phone, email, card, UPI), few-shot LLM ticket classification,
token budget enforcement, and customer-segment-aware ticket queue routing.
"""
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.part_c import main as run_step_3

def main():
    print("\n" + "=" * 70)
    print("STEP 3: SUPPORT TICKET TRIAGE (PII Guardrails & Dynamic Routing)")
    print("=" * 70)
    run_step_3()
    print("\nStep 3 completed successfully.")

if __name__ == "__main__":
    main()
