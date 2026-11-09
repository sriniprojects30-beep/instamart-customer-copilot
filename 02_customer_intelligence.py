"""
02_customer_intelligence.py -- Step 2: Customer Segments and Lapse Prediction
Computes RFM metrics, performs K-Means clustering, PCA dimensionality reduction,
and trains a leak-free Random Forest lapse prediction model to identify VIP retention targets.
"""
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.part_b import main as run_step_2

def main():
    print("\n" + "=" * 70)
    print("STEP 2: CUSTOMER INTELLIGENCE (Segmentation & Lapse Prediction)")
    print("=" * 70)
    run_step_2()
    print("\nStep 2 completed successfully.")

if __name__ == "__main__":
    main()
