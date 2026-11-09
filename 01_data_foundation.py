"""
01_data_foundation.py -- Step 1: Data Foundation (Clean, Store & Query)
Executes data auditing, standardisation, SQLite golden store creation,
and analytical SQL queries for Instamart order intelligence.
"""
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.part_a import main as run_step_1

def main():
    print("\n" + "=" * 70)
    print("STEP 1: DATA FOUNDATION (Clean, Store & SQL Queries)")
    print("=" * 70)
    run_step_1()
    print("\nStep 1 completed successfully.")

if __name__ == "__main__":
    main()
