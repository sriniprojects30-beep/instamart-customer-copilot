"""
test_suite.py -- Comprehensive QA and Regression Test Suite for Instamart Copilot.
Validates all 5 steps in order:
  Step 1: Data Foundation (Clean, Store & SQL Analytics)
  Step 2: Customer Intelligence (Segments & Lapse Prediction)
  Step 3: Support Ticket Triage (PII Guardrails & Routing)
  Step 4: Policy Knowledge Assistant & ReAct Refund Agent
  Step 5: Unified One-Input Copilot Omnibox
"""
import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import mock_llm
from src import part_a, part_b, part_c, part_d
import one_input_support

def test_step_1_data_foundation():
    print("[TEST] Step 1: Data Foundation Validation...")
    audit, cust_raw, prod_raw, ord_raw, tick_raw = part_a.audit_raw_data()
    assert audit["raw_orders_rows"] == 13084
    assert audit["exact_duplicates"] == 119
    assert audit["pending_rows"] == 656
    assert audit["distinct_order_ids"] == 12309
    assert audit["non_numeric_amounts"] == 120
    assert audit["negative_delivery_times"] == 106
    assert audit["blank_delivery_times"] == 2541
    assert audit["blank_ages"] == 61
    assert audit["ages_outside_range"] == 6
    assert audit["distinct_city_spellings"] == 24
    assert audit["distinct_gender_spellings"] == 10
    assert audit["blank_genders"] == 30

    # Cleaned checks
    cust_clean = pd.read_csv("data/clean/customers.csv")
    ord_clean = pd.read_csv("data/clean/orders.csv")
    assert len(cust_clean) == 1200
    assert cust_clean["city"].nunique() == 6
    assert set(cust_clean["gender"].unique()) == {"male", "female", "unknown"}
    assert cust_clean["age"].isna().sum() == 0
    assert cust_clean["age"].min() >= 18 and cust_clean["age"].max() <= 60

    assert len(ord_clean) == 12309
    assert ord_clean["order_id"].nunique() == 12309
    assert (ord_clean["status"] == "completed").sum() == 11110
    assert (ord_clean["status"] == "cancelled").sum() == 1199
    assert "pending" not in ord_clean["status"].values
    assert ord_clean["order_amount"].isna().sum() == 0

    comp_orders = ord_clean[ord_clean["status"] == "completed"]
    assert comp_orders["delivery_minutes"].isna().sum() == 0
    assert round(comp_orders["order_amount"].sum(), 2) == 2998668.80

    # SQL checks
    sql_results = part_a.run_sql_queries()
    assert len(sql_results["Q1"]) == 30
    assert len(sql_results["Q2"]) == 30
    assert round(sql_results["Q2"]["revenue"].sum(), 2) == 2998668.80
    assert len(sql_results["Q3"]) == 11
    assert len(sql_results["Q4"]) == 25
    assert len(sql_results["Q5"]) == 550
    assert (sql_results["Q6"]["spend_rank"] != sql_results["Q6"]["spend_row_number"]).sum() == 8
    assert len(sql_results["Q7"]) == 183
    assert round(sql_results["Q7"]["cumulative_revenue"].iloc[-1], 2) == 2998668.80
    print("  -> Step 1 PASSED successfully.")

def test_step_2_customer_intelligence():
    print("[TEST] Step 2: Customer Intelligence Validation...")
    seg_df = pd.read_csv("outputs/customer_segments.csv")
    assert len(seg_df) == 1175
    assert seg_df["is_premium"].sum() == 138
    assert set(seg_df["segment_id"].unique()) == {0, 1, 2, 3}

    # Verify model artifact
    import joblib
    model = joblib.load("outputs/lapse_model.joblib")
    assert model is not None

    ret_list = pd.read_csv("outputs/retention_list.csv")
    assert len(ret_list) == 10
    assert list(ret_list.columns) == ["customer_id", "lapse_probability", "actually_lapsed"]
    print("  -> Step 2 PASSED successfully.")

def test_step_3_support_triage():
    print("[TEST] Step 3: Support Ticket Triage Validation...")
    triage_results = pd.read_csv("outputs/triage_results.csv")
    assert len(triage_results) == 164
    assert set(triage_results["route"].unique()) == {"support_queue", "security_team", "human_review", "premium_desk"}
    assert set(triage_results["priority"].unique()) == {"NORMAL", "HIGH"}

    budget_demo = pd.read_csv("outputs/triage_budget_demo.csv")
    assert len(budget_demo) == 164
    assert (budget_demo["status"] == "budget_exhausted").sum() > 0
    print("  -> Step 3 PASSED successfully.")

def test_step_4_policy_refund_agent():
    print("[TEST] Step 4: Policy Assistant & Refund Agent Validation...")
    qa_eval = part_d.evaluate_policy_qa()
    assert qa_eval["overlap_10"]["doc_hits"] == 12
    assert qa_eval["overlap_10"]["fact_hits"] == 12
    assert qa_eval["overlap_10"]["fallback_hits"] == 2
    assert qa_eval["overlap_0"]["fact_hits"] < 12

    # Scenarios verification
    runs_file = os.path.join("outputs", "agent_runs.jsonl")
    with open(runs_file, "r", encoding="utf-8") as f:
        runs = [json.loads(line) for line in f]
    assert len(runs) >= 7

    # Check S1 refund
    s1 = next(r for r in runs if r["scenario_id"] == "S1")
    assert s1["status"] == "REFUND"
    assert s1["llm_calls"] == 5

    # Check S7 fallback
    s7 = next(r for r in runs if r["scenario_id"] == "S7")
    assert s7["status"] == "FALLBACK_MAX_ITERATIONS"
    assert s7["llm_calls"] == 6

    # Test crash and resume
    sc_df = pd.read_csv("data/raw/agent_scenarios.csv")
    s1_msg = sc_df.iloc[0]["message"]
    crash_id = "test_qa_crash"
    try:
        part_d.run_agent(s1_msg, "S1", run_id=crash_id, crash_after_tool_step=2)
    except RuntimeError:
        pass
    resumed = part_d.resume_agent(crash_id)
    assert resumed["status"] == "REFUND"
    assert resumed["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    print("  -> Step 4 PASSED successfully.")

def test_step_5_one_input_copilot():
    print("[TEST] Step 5: Unified One-Input Copilot Validation...")
    # Test 1: Customer Intelligence Disambiguation
    res_cust = one_input_support.handle_one_input("C0082")
    assert res_cust["intent"] == "CUSTOMER_INTELLIGENCE"
    assert "data" in res_cust
    assert res_cust["data"]["customer_id"] == "C0082"

    # Test 2: Policy Assistant RAG Disambiguation
    res_policy = one_input_support.handle_one_input("What is the refund window for dairy items?")
    assert res_policy["intent"] == "POLICY_ASSISTANT_RAG"
    assert len(res_policy["data"]["retrieved_sources"]) > 0

    # Test 3: Support Ticket Triage Disambiguation
    res_triage = one_input_support.handle_one_input("Contact 9876543210 my payment was debited twice")
    assert res_triage["intent"] == "SUPPORT_TICKET_TRIAGE"
    assert "9876543210" not in res_triage["data"]["redacted_text"]
    assert "[PHONE]" in res_triage["data"]["redacted_text"]

    # Test 4: Autonomous Refund Agent Disambiguation
    sc_df = pd.read_csv("data/raw/agent_scenarios.csv")
    s1_msg = sc_df.iloc[0]["message"]
    res_agent = one_input_support.handle_one_input(s1_msg)
    assert res_agent["intent"] == "AUTONOMOUS_REFUND_AGENT"
    assert res_agent["data"]["status"] == "REFUND"
    assert res_agent["data"]["refund_amount_inr"] > 0
    print("  -> Step 5 PASSED successfully.")

def main():
    print("==================================================")
    print("STARTING FULL END-TO-END QA REGRESSION TEST SUITE")
    print("==================================================")
    test_step_1_data_foundation()
    test_step_2_customer_intelligence()
    test_step_3_support_triage()
    test_step_4_policy_refund_agent()
    test_step_5_one_input_copilot()
    print("==================================================")
    print("ALL 5 STEPS PASSED QA VALIDATION WITH 100% SUCCESS!")
    print("==================================================")

if __name__ == "__main__":
    main()
