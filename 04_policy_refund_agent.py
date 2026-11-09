"""
04_policy_refund_agent.py -- Step 4: Policy Knowledge Assistant and Autonomous Refund Agent
Indexes store policy documents using TF-IDF RAG retrieval and runs an autonomous multi-turn ReAct
refund agent with tool execution, safety guards (token/time limits), crash-resumability, and ledger logging.
"""
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.part_d import main as run_step_4

def main():
    print("\n" + "=" * 70)
    print("STEP 4: POLICY ASSISTANT & REFUND AGENT (RAG & ReAct Autonomous Agent)")
    print("=" * 70)
    run_step_4()
    print("\nStep 4 completed successfully.")

if __name__ == "__main__":
    main()
