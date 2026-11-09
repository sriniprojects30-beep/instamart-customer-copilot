"""
src/part_d.py -- Part D: Policy RAG Assistant and Refund Agent
Chunks store policy documents, implements TF-IDF retrieval, grounds answers via
RAG, implements refund agent tools (DB queries, policy search, refund calculations,
and ledger commits), runs the ReAct agent across 7 test scenarios, enforces safe
fallback limits, and demonstrates checkpointing and resumability after failure.
"""
import os
import sys
import glob
import re
import json
import time
import sqlite3
from datetime import datetime, timezone
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mock_llm

POL_DIR = os.path.join("data", "policies")
DB_PATH = os.path.join("data", "instamart.db")
OUTPUTS_DIR = "outputs"
LEDGER_PATH = os.path.join(OUTPUTS_DIR, "refund_ledger.jsonl")
RUNS_PATH = os.path.join(OUTPUTS_DIR, "agent_runs.jsonl")
STATE_DIR = "state"

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)


# --- 1. Document Chunking & Retrieval ---

def chunk_documents(overlap=10):
    """
    Reads 5 policy text files and chunks them into 40-word chunks with specified overlap.
    """
    policy_files = sorted(glob.glob(os.path.join(POL_DIR, "*.txt")))
    chunks = []
    chunk_counts = {}
    for pf in policy_files:
        fname = os.path.basename(pf)
        with open(pf, "r", encoding="utf-8") as f:
            words = f.read().strip().split()
        step = 40 - overlap
        start = 0
        doc_chunks = []
        while True:
            chunk_words = words[start : start + 40]
            doc_chunks.append({
                "source": fname,
                "text": " ".join(chunk_words)
            })
            if start + 40 >= len(words):
                break
            start += step
        chunk_counts[fname] = len(doc_chunks)
        chunks.extend(doc_chunks)
    return chunks, chunk_counts


def create_retriever(chunks):
    """Fits TfidfVectorizer on chunk texts and returns a retrieval function."""
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer().fit(texts)
    chunk_vectors = vectorizer.transform(texts)

    def retrieve(query, k=3):
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, chunk_vectors).flatten()
        top_idx = sims.argsort()[::-1][:k]
        return [
            {"source": chunks[i]["source"], "score": float(sims[i]), "text": chunks[i]["text"]}
            for i in top_idx
        ]

    return retrieve


# Default retriever with overlap=10
DEFAULT_CHUNKS, DEFAULT_COUNTS = chunk_documents(overlap=10)
default_retrieve = create_retriever(DEFAULT_CHUNKS)


def answer(question, retrieve_fn=default_retrieve):
    """
    Retrieves 3 chunks and asks the POLICY ASSISTANT model for a grounded answer.
    """
    retrieved = retrieve_fn(question, k=3)
    context_str = "\n".join(f"[{r['source']}]\n{r['text']}" for r in retrieved)
    messages = [
        {
            "role": "system",
            "content": "POLICY ASSISTANT. Answer the question based solely on the provided context. If the answer cannot be found in the context, state that you could not find that information in the policy documents."
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context_str}\nQUESTION:\n{question}"
        }
    ]
    resp = mock_llm.chat(messages)
    return resp.choices[0].message.content, [r["source"] for r in retrieved]


def evaluate_policy_qa():
    """
    Evaluates retrieval and grounding over policy_eval.csv with overlap=10 and overlap=0.
    """
    eval_df = pd.read_csv(os.path.join("data", "raw", "policy_eval.csv"))

    # Overlap 10
    doc_hits_10 = 0
    fact_hits_10 = 0
    fallback_hits_10 = 0

    for _, row in eval_df.iterrows():
        q = row["question"]
        exp_doc = row["expected_doc"]
        exp_fact = row["expected_fact"]

        ans, sources = answer(q, retrieve_fn=default_retrieve)
        if exp_doc == "NONE":
            if ans.startswith("I could not find"):
                fallback_hits_10 += 1
        else:
            if exp_doc in sources:
                doc_hits_10 += 1
            if exp_fact in ans:
                fact_hits_10 += 1

    # Overlap 0
    chunks_0, _ = chunk_documents(overlap=0)
    retrieve_0 = create_retriever(chunks_0)
    fact_hits_0 = 0
    failed_questions_0 = []

    for _, row in eval_df.iterrows():
        q = row["question"]
        exp_doc = row["expected_doc"]
        exp_fact = row["expected_fact"]

        if exp_doc != "NONE":
            ans, sources = answer(q, retrieve_fn=retrieve_0)
            if exp_fact in ans:
                fact_hits_0 += 1
            else:
                failed_questions_0.append({
                    "question": q,
                    "expected_fact": exp_fact,
                    "model_answer": ans
                })

    return {
        "overlap_10": {
            "doc_hits": doc_hits_10,
            "fact_hits": fact_hits_10,
            "fallback_hits": fallback_hits_10
        },
        "overlap_0": {
            "fact_hits": fact_hits_0,
            "failed_questions": failed_questions_0
        }
    }


# --- 2. Tool Definitions ---

def get_order(order_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return {
            "order_id": row[0],
            "status": row[1],
            "order_amount": float(row[2]),
            "delivery_minutes": int(row[3]) if row[3] is not None and not pd.isna(row[3]) else None
        }
    return {"error": "order not found"}


def search_policy(query):
    return {"results": default_retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": float(order_amount)}
    return {"refund_inr": 0.0}


def issue_refund(order_id, refund_inr):
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "tool": "issue_refund",
        "order_id": order_id,
        "refund_inr": float(refund_inr)
    }
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": float(refund_inr)}


def escalate_to_human(order_id, refund_inr):
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "tool": "escalate_to_human",
        "order_id": order_id,
        "refund_inr": float(refund_inr)
    }
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": float(refund_inr)}


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Reads order details including status, amount, and delivery time from database.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}
    },
    {
        "name": "search_policy",
        "description": "Searches store policy documents using semantic TF-IDF retrieval.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    },
    {
        "name": "calculate_refund",
        "description": "Calculates refund eligibility and amount based on order amount and delivery duration.",
        "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "integer"}}, "required": ["order_amount", "delivery_minutes"]}
    },
    {
        "name": "issue_refund",
        "description": "Issues an approved refund up to INR 1000 and appends transaction to ledger.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}
    },
    {
        "name": "escalate_to_human",
        "description": "Escalates refund requests above INR 1000 to a human support lead and logs to ledger.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}
    }
]


# --- 3. ReAct Agent Loop & Safety Controls ---

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10, run_id=None, crash_after_tool_step=None):
    """
    Executes ReAct agent loop with tool calls, safety limits, and checkpointing.
    """
    if run_id is None:
        run_id = f"run_{scenario_id}_{int(time.time()*1000)}"

    start_time = time.monotonic()
    messages = [
        {"role": "system", "content": "You are a customer refund support AGENT. Use the available tools to verify orders, check policy, and resolve refund requests."},
        {"role": "user", "content": user_message}
    ]

    tokens_used = 0
    llm_calls = 0
    steps = []
    tools_executed = []

    for it in range(max_iterations):
        # 1. Timeout Check
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_TIMEOUT",
                "final_answer": "FALLBACK_TIMEOUT: Operation exceeded wall-clock time limit. Handing over to human support.",
                "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
                "refund_amount": 0.0, "tools_executed": tools_executed
            }

        # 2. Model Call
        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        call_tokens = resp.usage.total_tokens
        tokens_used += call_tokens

        # 3. Token Limit Check (stops before executing requested tool)
        if tokens_used > token_limit:
            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "FALLBACK_TOKEN_LIMIT: Token budget exceeded. Handing over to human support.",
                "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
                "refund_amount": 0.0, "tools_executed": tools_executed
            }

        msg = resp.choices[0].message

        # 4. Check if Final Answer reached
        if not msg.tool_calls:
            final_content = msg.content or ""
            status = final_content.split(":")[0].strip()

            # Determine refund amount from executed steps
            refund_amt = 0.0
            for step in steps:
                if step["tool_name"] in ["issue_refund", "escalate_to_human"]:
                    refund_amt = float(step["observation"].get("refund_inr", 0.0))

            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": status,
                "final_answer": final_content, "steps": steps, "llm_calls": llm_calls,
                "tokens_used": tokens_used, "refund_amount": refund_amt,
                "tools_executed": tools_executed
            }

        # 5. Handle Tool Call
        tc = msg.tool_calls[0]
        t_id = tc.id
        t_name = tc.name
        t_args = tc.arguments

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": t_id, "name": t_name, "arguments": t_args}]
        })

        fn = TOOL_MAP.get(t_name)
        obs = fn(**t_args) if fn else {"error": f"Tool {t_name} not found"}
        tools_executed.append(t_name)

        messages.append({
            "role": "tool",
            "tool_call_id": t_id,
            "name": t_name,
            "content": json.dumps(obs)
        })

        steps.append({
            "step": len(steps) + 1,
            "tool_name": t_name,
            "arguments": t_args,
            "observation": obs
        })

        # Save checkpoint to state/
        checkpoint = {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "messages": messages,
            "tokens_used": tokens_used,
            "llm_calls": llm_calls,
            "step_number": len(steps),
            "tools_executed": list(tools_executed)
        }
        with open(os.path.join(STATE_DIR, f"{run_id}.json"), "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)

        if crash_after_tool_step and len(tools_executed) == crash_after_tool_step:
            raise RuntimeError(f"Simulated crash after tool step {len(tools_executed)}")

    # 6. Max Iterations Exceeded
    return {
        "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "FALLBACK_MAX_ITERATIONS: Maximum iterations reached without resolution. Escalating case to human agent.",
        "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
        "refund_amount": 0.0, "tools_executed": tools_executed
    }


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    """
    Resumes an agent execution from saved checkpoint in state/.
    """
    ckpt_file = os.path.join(STATE_DIR, f"{run_id}.json")
    with open(ckpt_file, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    start_time = time.monotonic()
    scenario_id = checkpoint["scenario_id"]
    messages = checkpoint["messages"]
    tokens_used = checkpoint["tokens_used"]
    llm_calls = checkpoint["llm_calls"]
    tools_executed = list(checkpoint["tools_executed"])
    steps = []

    # Reconstruct steps from messages
    for i in range(len(messages)):
        if messages[i]["role"] == "assistant" and messages[i].get("tool_calls"):
            tc = messages[i]["tool_calls"][0]
            if i + 1 < len(messages) and messages[i+1]["role"] == "tool":
                obs = json.loads(messages[i+1]["content"])
                steps.append({
                    "step": len(steps) + 1,
                    "tool_name": tc["name"],
                    "arguments": tc["arguments"],
                    "observation": obs
                })

    remaining_iter = max_iterations - llm_calls
    for it in range(remaining_iter):
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_TIMEOUT",
                "final_answer": "FALLBACK_TIMEOUT: Operation exceeded wall-clock time limit. Handing over to human support.",
                "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
                "refund_amount": 0.0, "tools_executed": tools_executed
            }

        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens

        if tokens_used > token_limit:
            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "FALLBACK_TOKEN_LIMIT: Token budget exceeded. Handing over to human support.",
                "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
                "refund_amount": 0.0, "tools_executed": tools_executed
            }

        msg = resp.choices[0].message
        if not msg.tool_calls:
            final_content = msg.content or ""
            status = final_content.split(":")[0].strip()
            refund_amt = 0.0
            for step in steps:
                if step["tool_name"] in ["issue_refund", "escalate_to_human"]:
                    refund_amt = float(step["observation"].get("refund_inr", 0.0))

            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": status,
                "final_answer": final_content, "steps": steps, "llm_calls": llm_calls,
                "tokens_used": tokens_used, "refund_amount": refund_amt,
                "tools_executed": tools_executed
            }

        tc = msg.tool_calls[0]
        t_id = tc.id
        t_name = tc.name
        t_args = tc.arguments

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": t_id, "name": t_name, "arguments": t_args}]
        })

        fn = TOOL_MAP.get(t_name)
        obs = fn(**t_args) if fn else {"error": f"Tool {t_name} not found"}
        tools_executed.append(t_name)

        messages.append({
            "role": "tool",
            "tool_call_id": t_id,
            "name": t_name,
            "content": json.dumps(obs)
        })

        steps.append({
            "step": len(steps) + 1,
            "tool_name": t_name,
            "arguments": t_args,
            "observation": obs
        })

        checkpoint = {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "messages": messages,
            "tokens_used": tokens_used,
            "llm_calls": llm_calls,
            "step_number": len(steps),
            "tools_executed": list(tools_executed)
        }
        with open(os.path.join(STATE_DIR, f"{run_id}.json"), "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)

    return {
        "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "FALLBACK_MAX_ITERATIONS: Maximum iterations reached without resolution.",
        "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
        "refund_amount": 0.0, "tools_executed": tools_executed
    }


def main():
    print("=== PART D: POLICY RAG ASSISTANT AND REFUND AGENT ===")
    
    # 1. Chunking verification
    print(f"Total chunks created with overlap=10: {len(DEFAULT_CHUNKS)} (counts: {DEFAULT_COUNTS})")
    
    # 2. Retrieval check
    top_refund = default_retrieve("refund policy late delivery", k=3)
    print("Top 3 retrieved for 'refund policy late delivery':")
    for r in top_refund:
        print(f"  {r['source']} (score: {r['score']:.4f})")

    # 3. Policy evaluation
    print("\nEvaluating policy QA against policy_eval.csv...")
    qa_eval = evaluate_policy_qa()
    print(f"Overlap 10: Doc Hits={qa_eval['overlap_10']['doc_hits']}/12, Fact Hits={qa_eval['overlap_10']['fact_hits']}/12, Fallback Hits={qa_eval['overlap_10']['fallback_hits']}/2")
    print(f"Overlap 0: Fact Hits={qa_eval['overlap_0']['fact_hits']}/12 (Failed: {len(qa_eval['overlap_0']['failed_questions'])})")

    # 4. Clean ledger before run
    if os.path.exists(LEDGER_PATH):
        os.remove(LEDGER_PATH)
    if os.path.exists(RUNS_PATH):
        os.remove(RUNS_PATH)

    # 5. Run 7 scenarios
    print("\nExecuting 7 Agent Scenarios...")
    scenarios_df = pd.read_csv(os.path.join("data", "raw", "agent_scenarios.csv"))
    scenario_results = []

    for _, row in scenarios_df.iterrows():
        sid = row["scenario_id"]
        msg = row["message"]
        exp_out = row["expected_outcome"]
        exp_amt = float(row["expected_refund_inr"])

        res = run_agent(msg, sid)
        scenario_results.append(res)

        # Write to outputs/agent_runs.jsonl
        run_record = {
            "scenario_id": sid,
            "status": res["status"],
            "final_answer": res["final_answer"],
            "llm_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"],
            "steps": res["steps"]
        }
        with open(RUNS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_record) + "\n")

        print(f"Scenario {sid}: Status={res['status']} (Exp: {exp_out}), Refund=INR {res['refund_amount']:.2f} (Exp: INR {exp_amt:.2f}), Calls={res['llm_calls']}, Tokens={res['tokens_used']}")

    # 6. Stop Safely Demonstrations
    print("\nExecuting Stop Safely Demonstrations...")
    s1_msg = scenarios_df.iloc[0]["message"]
    demo_tok = run_agent(s1_msg, "S1_demo_token", token_limit=200)
    print(f"Stop Safely (token_limit=200): Status={demo_tok['status']}, Calls={demo_tok['llm_calls']}")

    demo_timeout = run_agent(s1_msg, "S1_demo_timeout", time_limit_seconds=0)
    print(f"Stop Safely (time_limit_seconds=0): Status={demo_timeout['status']}, Calls={demo_timeout['llm_calls']}")

    # 7. Resumability Demonstration
    print("\nExecuting Crash and Resume Demonstration...")
    crash_id = "demo_s1_crash_resume"
    try:
        run_agent(s1_msg, "S1", run_id=crash_id, crash_after_tool_step=2)
    except RuntimeError as e:
        print(f"Successfully simulated crash: {e}")

    resumed_res = resume_agent(crash_id)
    print(f"Resumed status: {resumed_res['status']}, Refund: INR {resumed_res['refund_amount']:.2f}")
    print(f"Tools executed across both parts: {resumed_res['tools_executed']}")


if __name__ == "__main__":
    main()
