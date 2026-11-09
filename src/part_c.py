"""
src/part_c.py -- Part C: Support Ticket Triage with Guardrails
Processes support tickets with input guardrails (empty/long), redacts PII,
evaluates zero-shot and few-shot LLM prompts with resilient retry mechanisms,
performs strict schema/keyword validation, demonstrates token budgets, and
routes tickets dynamically according to customer segmentation flags.
"""
import os
import sys
import re
import json
import random
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mock_llm
import pandas as pd

TICKETS_PATH = os.path.join("data", "raw", "tickets.csv")
SEGMENTS_PATH = os.path.join("outputs", "customer_segments.csv")
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Guardrail constants
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
ALLOWED_CATEGORIES = ["billing", "delivery", "technical", "fraud"]
ALLOWED_CATS_SET = set(ALLOWED_CATEGORIES)

# Regex patterns for redaction
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_PATTERN = re.compile(r"ORD-\d{6}")

# System Prompts
ZERO_SHOT_PROMPT = """You are a customer support ticket classification system.
CLASSIFY the incoming support ticket into exactly one of the following four allowed categories:
- billing
- delivery
- technical
- fraud

You must respond strictly in JSON format with keys "category" and "reason":
{"category": "<category>", "reason": "<reason>"}
Do not include any conversational filler or markdown code fences."""

FEW_SHOT_PROMPT = """You are a customer support ticket classification system.
CLASSIFY the incoming support ticket into exactly one of the following four allowed categories:
- billing
- delivery
- technical
- fraud

You must respond strictly in JSON format with keys "category" and "reason":
{"category": "<category>", "reason": "<reason>"}

EXAMPLES:
Ticket: An unauthorized charge appeared on my card statement. Please look into this immediately.
Response: {"category": "fraud", "reason": "unauthorized card transaction"}

Ticket: The delivery partner is delayed by 45 minutes and is not answering calls.
Response: {"category": "delivery", "reason": "rider delay"}"""


def redact_pii(text: str):
    """
    Applies regex substitutions strictly in order:
    1. Email -> [EMAIL]
    2. Phone -> [PHONE]
    3. Order ID -> [ORDER_ID]
    Returns redacted text and counts of replacements.
    """
    t, n_e = EMAIL_PATTERN.subn("[EMAIL]", text)
    t, n_p = PHONE_PATTERN.subn("[PHONE]", t)
    t, n_o = ORDER_PATTERN.subn("[ORDER_ID]", t)
    return t, n_e, n_p, n_o


def validate_redaction(redacted_texts):
    """Verifies that no PII patterns match in the redacted texts."""
    leaks = {
        "email": sum(len(EMAIL_PATTERN.findall(t)) for t in redacted_texts),
        "phone": sum(len(PHONE_PATTERN.findall(t)) for t in redacted_texts),
        "order_id": sum(len(ORDER_PATTERN.findall(t)) for t in redacted_texts)
    }
    assert leaks["email"] == 0 and leaks["phone"] == 0 and leaks["order_id"] == 0, f"PII leak detected: {leaks}"
    return leaks


def validate_model_response(content: str):
    """
    Validates model output:
    1. Valid JSON with category in allowed set -> ('classified_json', category)
    2. If not valid JSON, extract words via re.findall(r"[a-z]+", text.lower()) -> exactly 1 allowed category -> ('classified_keyword', category)
    3. Otherwise -> ('flagged_invalid_output', None)
    """
    # 1. JSON parse
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            cat = str(data.get("category", "")).lower().strip()
            if cat in ALLOWED_CATS_SET:
                return "classified_json", cat
    except Exception:
        pass

    # 2. Keyword fallback extraction
    words = set(re.findall(r"[a-z]+", (content or "").lower()))
    matches = words & ALLOWED_CATS_SET
    if len(matches) == 1:
        return "classified_keyword", list(matches)[0]

    # 3. Invalid output
    return "flagged_invalid_output", None


def run_triage_pipeline(prompt_text: str, token_budget=None, silent=False):
    """
    Executes support triage pipeline over tickets.csv.
    """
    mock_llm.reset()
    tickets_df = pd.read_csv(TICKETS_PATH, keep_default_na=False)

    results = []
    running_tokens = 0
    transient_errors = 0
    retry_logs = []

    for idx, row in tickets_df.iterrows():
        tid = row["ticket_id"]
        cid = row["customer_id"]
        raw_text = row["text"]
        true_cat = row["true_category"]

        # Guardrail (a): Skipped empty
        if not raw_text.strip():
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "skipped_empty",
                "category": "", "true_category": true_cat, "tokens_used": 0
            })
            continue

        # Guardrail (b): Flagged long
        if mock_llm.est_tokens(raw_text) > MAX_INPUT_TOKENS:
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "flagged_long",
                "category": "", "true_category": true_cat, "tokens_used": 0
            })
            continue

        # Token budget enforcement
        if token_budget is not None and running_tokens >= token_budget:
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "budget_exhausted",
                "category": "", "true_category": true_cat, "tokens_used": 0
            })
            continue

        # Redact PII
        redacted_text, _, _, _ = redact_pii(raw_text)

        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": redacted_text}
        ]

        # Resilient model call with exponential backoff and jitter
        resp = None
        for n in range(MAX_RETRIES):
            try:
                resp = mock_llm.chat(messages)
                break
            except mock_llm.TransientLLMError as e:
                transient_errors += 1
                wait_sec = BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15)
                log_msg = f"[RETRY] Ticket {tid}: retry {n}, wait={wait_sec:.4f}s ({e})"
                retry_logs.append(log_msg)
                if not silent:
                    print(log_msg)
                time.sleep(wait_sec)

        if resp is None:
            resp = mock_llm.chat(messages)

        call_tokens = resp.usage.total_tokens
        running_tokens += call_tokens

        content = resp.choices[0].message.content
        status, cat = validate_model_response(content)

        results.append({
            "ticket_id": tid,
            "customer_id": cid,
            "status": status,
            "category": cat if cat is not None else "",
            "true_category": true_cat,
            "tokens_used": call_tokens
        })

    df_results = pd.DataFrame(results)
    return df_results, transient_errors, retry_logs, running_tokens


def apply_routing(df_triage: pd.DataFrame):
    """
    Routes tickets according to specification:
    - skipped_empty, flagged_long, flagged_invalid_output, budget_exhausted -> route human_review, priority NORMAL
    - category == 'fraud' -> route security_team, priority HIGH
    - customer is_premium == True -> route premium_desk, priority HIGH
    - otherwise -> route support_queue, priority NORMAL
    """
    segments_df = pd.read_csv(SEGMENTS_PATH) if os.path.exists(SEGMENTS_PATH) else pd.DataFrame(columns=["customer_id", "segment_name", "is_premium"])
    seg_map = segments_df.set_index("customer_id").to_dict(orient="index")

    routes = []
    priorities = []
    seg_names = []

    for _, row in df_triage.iterrows():
        cid = row["customer_id"]
        status = row["status"]
        cat = row["category"]

        seg_info = seg_map.get(cid, {"segment_name": "", "is_premium": False})
        is_prem = bool(seg_info.get("is_premium", False))
        s_name = str(seg_info.get("segment_name", ""))
        seg_names.append(s_name if status in ["classified_json", "classified_keyword"] else "")

        if status in ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]:
            routes.append("human_review")
            priorities.append("NORMAL")
        elif cat == "fraud":
            routes.append("security_team")
            priorities.append("HIGH")
        elif is_prem:
            routes.append("premium_desk")
            priorities.append("HIGH")
        else:
            routes.append("support_queue")
            priorities.append("NORMAL")

    df_triage["segment_name"] = seg_names
    df_triage["priority"] = priorities
    df_triage["route"] = routes

    final_cols = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
    return df_triage[final_cols]


def main():
    print("=== PART C: SUPPORT TICKET TRIAGE WITH GUARDRAILS ===")
    tickets_df = pd.read_csv(TICKETS_PATH, keep_default_na=False)

    # Redaction audit
    tot_e, tot_p, tot_o = 0, 0, 0
    all_redacted = []
    for txt in tickets_df["text"]:
        r_txt, ne, np_, no = redact_pii(txt)
        tot_e += ne
        tot_p += np_
        tot_o += no
        all_redacted.append(r_txt)

    validate_redaction(all_redacted)
    print(f"Redaction substitutions: emails={tot_e}, phones={tot_p}, orders={tot_o}, total={tot_e + tot_p + tot_o}")
    print("Re-application verification passed with 0 leaks.")

    # 1. Zero-shot run
    print("\nExecuting Zero-shot pipeline...")
    df_zero, errs_z, logs_z, tok_z = run_triage_pipeline(ZERO_SHOT_PROMPT, silent=True)
    class_z = df_zero[df_zero["status"].isin(["classified_json", "classified_keyword"])]
    acc_z = (class_z["category"] == class_z["true_category"]).mean()
    print(f"Zero-shot: Classified={len(class_z)}, Acc={acc_z*100:.2f}%, Total tokens={tok_z}, Tokens/ticket={tok_z/len(class_z):.2f}")

    # 2. Few-shot run
    print("\nExecuting Few-shot pipeline...")
    df_few, errs_f, logs_f, tok_f = run_triage_pipeline(FEW_SHOT_PROMPT, silent=True)
    class_f = df_few[df_few["status"].isin(["classified_json", "classified_keyword"])]
    acc_f = (class_f["category"] == class_f["true_category"]).mean()
    print(f"Few-shot: Classified={len(class_f)}, Acc={acc_f*100:.2f}%, Total tokens={tok_f}, Tokens/ticket={tok_f/len(class_f):.2f}")

    # 3. Budget demo run (zero-shot, budget 6000)
    print("\nExecuting Budget Demo (TOKEN_BUDGET=6000)...")
    df_budget, _, _, _ = run_triage_pipeline(ZERO_SHOT_PROMPT, token_budget=6000, silent=True)
    budget_csv = os.path.join(OUTPUTS_DIR, "triage_budget_demo.csv")
    df_budget.to_csv(budget_csv, index=False)
    exhausted_count = int((df_budget["status"] == "budget_exhausted").sum())
    print(f"Saved {budget_csv}. Tickets budget_exhausted: {exhausted_count}")

    # 4. Final few-shot run with routing (budget 100000)
    print("\nExecuting Final Triage & Routing (Few-shot, budget=100000)...")
    df_final_raw, _, _, _ = run_triage_pipeline(FEW_SHOT_PROMPT, token_budget=100000, silent=True)
    df_routed = apply_routing(df_final_raw)
    final_csv = os.path.join(OUTPUTS_DIR, "triage_results.csv")
    df_routed.to_csv(final_csv, index=False)
    print(f"Saved final triage results to {final_csv} ({len(df_routed)} rows).")
    print("Routing breakdown:")
    print(df_routed["route"].value_counts())


if __name__ == "__main__":
    main()
