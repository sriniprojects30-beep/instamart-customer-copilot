"""
one_input_support.py -- Universal One-Input Support Copilot for Instamart.
Provides a unified interface that accepts any raw user query, ticket, customer ID,
or order issue, dynamically detects intent, redacts sensitive PII, and routes to:
1. Customer 360 & Lapse Risk Profiling
2. Autonomous ReAct Refund Agent
3. Grounded Policy Knowledge Assistant (RAG)
4. Privacy-Safe Support Ticket Triage & Queue Dispatch
"""
import os
import sys
import re
import json
import sqlite3
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import mock_llm
from src import part_a, part_b, part_c, part_d

DB_PATH = os.path.join("data", "instamart.db")
SEGMENTS_PATH = os.path.join("outputs", "customer_segments.csv")
RETENTION_PATH = os.path.join("outputs", "retention_list.csv")

def handle_one_input(raw_input: str) -> dict:
    text = (raw_input or "").strip()
    if not text:
        return {
            "status": "skipped_empty",
            "intent": "EMPTY_INPUT",
            "intent_label": "Empty / Whitespace Input",
            "summary": "No text provided. Please enter an inquiry, ticket, order ID, or customer ID.",
            "route": "human_review",
            "priority": "NORMAL"
        }

    # 1. PII Redaction
    redacted_text, n_e, n_p, n_o = part_c.redact_pii(text)
    redaction_summary = {
        "emails_masked": n_e,
        "phones_masked": n_p,
        "orders_masked": n_o,
        "total_masked": n_e + n_p + n_o
    }

    # 2. Token Guardrail
    est_toks = mock_llm.est_tokens(text)
    if est_toks > part_c.MAX_INPUT_TOKENS:
        return {
            "status": "flagged_long",
            "intent": "OVERSIZED_INPUT",
            "intent_label": "Oversized Input (>400 Tokens)",
            "summary": f"Input estimated to {est_toks} tokens, exceeding maximum limit of {part_c.MAX_INPUT_TOKENS}. Routed directly to senior human support without LLM consumption.",
            "route": "human_review",
            "priority": "NORMAL",
            "redacted_text": redacted_text,
            "redaction_summary": redaction_summary
        }

    # 3. Entity Detection
    m_cust = re.search(r"\b(C\d{4})\b", text, re.I)
    m_ord = re.search(r"\b(ORD-\d{6})\b", text, re.I)
    cid = m_cust.group(1).upper() if m_cust else None
    oid = m_ord.group(1).upper() if m_ord else None

    # Load customer segments map
    seg_map = {}
    if os.path.exists(SEGMENTS_PATH):
        df_seg = pd.read_csv(SEGMENTS_PATH)
        seg_map = df_seg.set_index("customer_id").to_dict(orient="index")

    # Load retention list map
    ret_map = {}
    if os.path.exists(RETENTION_PATH):
        df_ret = pd.read_csv(RETENTION_PATH)
        ret_map = df_ret.set_index("customer_id").to_dict(orient="index")

    # --- CASE A: Pure Customer Lookup ---
    # e.g., "C0082", "customer C0082", "details for C0082"
    if cid and (re.fullmatch(r"C\d{4}", text, re.I) or "customer" in text.lower() or "user" in text.lower() and not oid and "refund" not in text.lower()):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT customer_id, city, signup_date, age, gender FROM customers WHERE customer_id = ?", (cid,))
        row = cur.fetchone()
        conn.close()

        if row:
            s_info = seg_map.get(cid, {"segment_name": "Standard Customer", "is_premium": False, "recency": 0, "frequency": 0, "monetary": 0.0})
            ret_info = ret_map.get(cid, {})
            lapse_prob = ret_info.get("lapse_probability", 0.45 if s_info.get("is_premium") else 0.25)

            return {
                "status": "SUCCESS",
                "intent": "CUSTOMER_INTELLIGENCE",
                "intent_label": "Customer 360 & Lapse Risk Profile",
                "summary": f"Retrieved profile for customer {cid} in {row[1]} ({s_info.get('segment_name')}).",
                "route": "premium_desk" if s_info.get("is_premium") else "support_queue",
                "priority": "HIGH" if s_info.get("is_premium") else "NORMAL",
                "redacted_text": redacted_text,
                "redaction_summary": redaction_summary,
                "data": {
                    "customer_id": row[0],
                    "city": row[1],
                    "signup_date": row[2],
                    "age": row[3],
                    "gender": row[4],
                    "recency_days": s_info.get("recency", 0),
                    "frequency_orders": s_info.get("frequency", 0),
                    "monetary_spend_inr": s_info.get("monetary", 0.0),
                    "segment_name": s_info.get("segment_name", "Standard Customer"),
                    "is_premium": s_info.get("is_premium", False),
                    "predicted_lapse_probability": round(float(lapse_prob), 4)
                }
            }

    # Check for Policy Question first if it's asking a general question without an order ID
    policy_starters = ["what", "how", "when", "who", "why", "where", "is ", "can ", "does ", "are ", "do ", "tell me"]
    is_general_question = any(text.lower().startswith(ps) for ps in policy_starters)
    
    # --- CASE B: Policy Question (RAG) ---
    policy_keywords = ["policy", "fee", "fees", "charge", "refund rule", "otp", "password", "rain", "heavy rain", "wallet", "cod", "cash on delivery", "coupons", "privacy", "delete", "retention", "security team", "escalat"]
    is_policy_q = (is_general_question and not oid) or any(kw in text.lower() for kw in policy_keywords)

    if is_policy_q and not oid and not any(p in text.lower() for p in ["my order", "i want", "please refund"]):
        ans, sources = part_d.answer(text)
        top_chunks = part_d.default_retrieve(text, k=3)
        return {
            "status": "SUCCESS",
            "intent": "POLICY_ASSISTANT_RAG",
            "intent_label": "Policy Assistant & Knowledge Retrieval (RAG)",
            "summary": ans,
            "route": "support_queue",
            "priority": "NORMAL",
            "redacted_text": redacted_text,
            "redaction_summary": redaction_summary,
            "data": {
                "grounded_answer": ans,
                "retrieved_sources": top_chunks
            }
        }

    # --- CASE C: Autonomous Refund Request ---
    refund_keywords = ["refund", "money back", "cancel", "cancelled", "late delivery", "delayed", "charged twice", "wrong amount", "extra payment"]
    is_refund_inquiry = any(kw in text.lower() for kw in refund_keywords)

    if oid or is_refund_inquiry:
        # Run Autonomous ReAct Refund Agent
        agent_res = part_d.run_agent(text, scenario_id="ONE_INPUT")
        final_status = agent_res["status"]
        refund_amt = agent_res["refund_amount"]

        # Look up associated customer tier if order exists
        cust_tier = "Standard"
        is_prem = False
        if oid:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT customer_id FROM orders WHERE order_id = ?", (oid,))
            ord_row = cur.fetchone()
            conn.close()
            if ord_row:
                cid_from_ord = ord_row[0]
                s_info = seg_map.get(cid_from_ord, {})
                cust_tier = s_info.get("segment_name", "Standard")
                is_prem = s_info.get("is_premium", False)

        priority = "HIGH" if (final_status == "ESCALATE" or is_prem) else "NORMAL"
        route = "premium_desk" if is_prem else ("human_review" if final_status in ["ESCALATE", "FALLBACK_MAX_ITERATIONS", "FALLBACK_TOKEN_LIMIT", "FALLBACK_TIMEOUT"] else "support_queue")

        return {
            "status": final_status,
            "intent": "AUTONOMOUS_REFUND_AGENT",
            "intent_label": f"Autonomous ReAct Refund Agent &mdash; {final_status}",
            "summary": agent_res["final_answer"],
            "route": route,
            "priority": priority,
            "redacted_text": redacted_text,
            "redaction_summary": redaction_summary,
            "data": {
                "order_id": oid,
                "refund_status": final_status,
                "refund_amount_inr": refund_amt,
                "model_calls": agent_res["llm_calls"],
                "tokens_used": agent_res["tokens_used"],
                "customer_tier": cust_tier,
                "is_premium": is_prem,
                "execution_steps": agent_res["steps"]
            }
        }

    # --- CASE D: General Support Ticket Triage ---
    # Default to few-shot classification and queue routing
    messages = [
        {"role": "system", "content": part_c.FEW_SHOT_PROMPT},
        {"role": "user", "content": redacted_text}
    ]
    resp = mock_llm.chat(messages)
    content = resp.choices[0].message.content
    val_status, cat = part_c.validate_model_response(content)

    # Route logic
    if cat == "fraud":
        route, priority = "security_team", "HIGH"
    elif val_status != "classified_json":
        route, priority = "human_review", "NORMAL"
    else:
        route, priority = "support_queue", "NORMAL"

    return {
        "status": val_status,
        "intent": "SUPPORT_TICKET_TRIAGE",
        "intent_label": f"Support Ticket Triage &mdash; {cat.upper() if cat else 'REVIEW'}",
        "summary": f"Classified ticket into category '{cat}' with priority {priority}. Routed to {route}.",
        "route": route,
        "priority": priority,
        "redacted_text": redacted_text,
        "redaction_summary": redaction_summary,
        "data": {
            "category": cat,
            "validation_status": val_status,
            "raw_model_response": content
        }
    }

def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        res = handle_one_input(query)
        print("\n" + "="*70)
        print(f"ONE-INPUT COPILOT RESULT: {res['intent_label']}")
        print("="*70)
        print(f"Status:   {res['status']}")
        print(f"Route:    {res['route']} ({res['priority']})")
        print(f"Summary:  {res['summary']}")
        if res.get("redaction_summary") and res["redaction_summary"]["total_masked"] > 0:
            print(f"Privacy:  {res['redaction_summary']['total_masked']} PII values safely redacted.")
        print("\nStructured Data:")
        print(json.dumps(res.get("data", {}), indent=2))
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("INSTAMART ONE-INPUT SUPPORT COPILOT (Interactive Mode)")
        print("Type any question, complaint, order ID, or customer ID (or 'exit' to quit):")
        print("="*70)
        while True:
            try:
                user_q = input("\nInstamart Copilot > ").strip()
                if not user_q or user_q.lower() in ["exit", "quit", "q"]:
                    print("Exiting Copilot. Have a great day!")
                    break
                out = handle_one_input(user_q)
                print(f"\n[{out['intent_label']}] -> Route: {out['route']} (Priority: {out['priority']})")
                print(f"Summary: {out['summary']}")
                if "data" in out:
                    if out["intent"] == "AUTONOMOUS_REFUND_AGENT":
                        print(f"Refund Amount: INR {out['data'].get('refund_amount_inr', 0):.2f}")
                        print(f"Steps: {len(out['data'].get('execution_steps', []))} tool executions")
                    elif out["intent"] == "CUSTOMER_INTELLIGENCE":
                        print(f"Segment: {out['data'].get('segment_name')} (VIP: {out['data'].get('is_premium')})")
                        print(f"Lapse Probability: {out['data'].get('predicted_lapse_probability')*100:.1f}%")
                    elif out["intent"] == "POLICY_ASSISTANT_RAG":
                        print(f"Sources: {[s['source'] for s in out['data'].get('retrieved_sources', [])]}")
            except (KeyboardInterrupt, EOFError):
                print("\nSession ended.")
                break

if __name__ == "__main__":
    main()
