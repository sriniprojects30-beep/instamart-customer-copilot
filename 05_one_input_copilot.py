"""
05_one_input_copilot.py -- Step 5: Unified One-Input Support Copilot
Universal Omnibox interface that seamlessly disambiguates user intent and routes any query to:
1. Customer 360 & Lapse Intelligence
2. Autonomous ReAct Refund Resolution
3. Policy RAG Knowledge Base
4. Privacy-Safe Support Ticket Triage
"""
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from one_input_support import main as run_step_5

def main():
    print("\n" + "=" * 70)
    print("STEP 5: UNIFIED ONE-INPUT SUPPORT COPILOT (Omnibox)")
    print("=" * 70)
    run_step_5()

if __name__ == "__main__":
    main()
