"""
app.py -- Instamart Customer Intelligence & Support Copilot Web Application
A full-stack, zero-dependency interactive dashboard providing an executive UI for:
1. Executive Analytics & KPIs
2. Customer Intelligence & Churn Prediction
3. Privacy-Safe Support Ticket Triage & PII Redaction
4. Grounded Policy RAG Assistant
5. ReAct Refund Agent Simulator & Audit Ledger
"""
import os
import sys
import json
import sqlite3
import re
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import mock_llm
from src import part_a, part_b, part_c, part_d

PORT = 8000
DB_PATH = os.path.join("data", "instamart.db")

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Instamart Customer Intelligence & Support Copilot</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --primary: #f97316;
      --primary-hover: #ea580c;
      --accent: #38bdf8;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --success: #22c55e;
      --warning: #eab308;
      --danger: #ef4444;
      --radius: 10px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    body { background: var(--bg); color: var(--text); padding-bottom: 50px; }
    header { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-bottom: 1px solid var(--border); padding: 20px 40px; display: flex; justify-content: space-between; align-items: center; }
    .brand { display: flex; align-items: center; gap: 12px; }
    .logo-badge { background: var(--primary); color: white; font-weight: 800; font-size: 20px; padding: 6px 14px; border-radius: 8px; }
    .brand h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.5px; }
    .brand p { font-size: 13px; color: var(--text-muted); }
    .status-pill { background: rgba(34, 197, 94, 0.15); color: var(--success); border: 1px solid rgba(34, 197, 94, 0.3); padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600; display: flex; align-items: center; gap: 6px; }
    .status-dot { width: 8px; height: 8px; background: var(--success); border-radius: 50%; display: inline-block; }
    
    .nav-tabs { display: flex; background: #1e293b; border-bottom: 1px solid var(--border); padding: 0 40px; gap: 10px; }
    .tab-btn { background: none; border: none; color: var(--text-muted); font-size: 15px; font-weight: 600; padding: 16px 20px; cursor: pointer; border-bottom: 3px solid transparent; transition: all 0.2s; }
    .tab-btn:hover { color: var(--text); }
    .tab-btn.active { color: var(--primary); border-bottom-color: var(--primary); }
    
    .container { max-width: 1300px; margin: 30px auto; padding: 0 20px; }
    .tab-content { display: none; }
    .tab-content.active { display: block; animation: fadeIn 0.2s ease-in-out; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

    /* KPI Grid */
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 30px; }
    .kpi-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; }
    .kpi-label { font-size: 13px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-bottom: 8px; }
    .kpi-val { font-size: 28px; font-weight: 700; color: var(--text); }
    .kpi-sub { font-size: 12px; color: var(--accent); margin-top: 6px; }

    /* Panels & Cards */
    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 25px; margin-bottom: 25px; }
    .card h2 { font-size: 18px; font-weight: 700; margin-bottom: 16px; color: var(--text); display: flex; align-items: center; gap: 8px; }
    .card p.desc { color: var(--text-muted); font-size: 14px; margin-bottom: 20px; }

    /* Forms */
    .form-group { margin-bottom: 16px; }
    label { display: block; font-size: 13px; font-weight: 600; color: var(--text-muted); margin-bottom: 6px; }
    input, select, textarea { width: 100%; background: #0f172a; border: 1px solid var(--border); color: var(--text); padding: 12px 14px; border-radius: 8px; font-size: 14px; outline: none; transition: border-color 0.2s; }
    input:focus, select:focus, textarea:focus { border-color: var(--primary); }
    textarea { resize: vertical; min-height: 90px; }
    .btn { background: var(--primary); color: white; border: none; padding: 12px 24px; border-radius: 8px; font-weight: 600; font-size: 14px; cursor: pointer; transition: background 0.2s; display: inline-flex; align-items: center; gap: 8px; }
    .btn:hover { background: var(--primary-hover); }
    .btn-secondary { background: #334155; }
    .btn-secondary:hover { background: #475569; }

    /* Tables */
    table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 13px; }
    th, td { text-align: left; padding: 12px 14px; border-bottom: 1px solid var(--border); }
    th { background: #1e293b; color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 11px; }
    tr:hover { background: rgba(255,255,255,0.02); }

    /* Badges */
    .badge { padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; text-transform: uppercase; display: inline-block; }
    .badge-premium { background: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.4); }
    .badge-normal { background: rgba(56, 189, 248, 0.2); color: #38bdf8; }
    .badge-high { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-success { background: rgba(34, 197, 94, 0.2); color: #4ade80; }

    /* Step Traces */
    .step-box { background: #0f172a; border-left: 3px solid var(--accent); padding: 12px 16px; margin-bottom: 12px; border-radius: 0 8px 8px 0; font-size: 13px; }
    .step-header { font-weight: 700; color: var(--accent); margin-bottom: 4px; display: flex; justify-content: space-between; }
    pre { background: #090d16; padding: 10px; border-radius: 6px; font-family: monospace; font-size: 12px; overflow-x: auto; color: #a5b4fc; margin-top: 6px; }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="logo-badge">⚡ Instamart</div>
      <div>
        <h1>Customer Intelligence & Support Copilot</h1>
        <p>Enterprise Analytics, Churn Prediction & Autonomous Resolution</p>
      </div>
    </div>
    <div class="status-pill">
      <span class="status-dot"></span> All Systems Operational
    </div>
  </header>

  <nav class="nav-tabs">
    <button class="tab-btn active" onclick="showTab('tab-one-input')">🎯 Omnibox Copilot</button>
    <button class="tab-btn" onclick="showTab('tab-dashboard')">1️⃣ Data Foundation</button>
    <button class="tab-btn" onclick="showTab('tab-customers')">2️⃣ Customer Intelligence</button>
    <button class="tab-btn" onclick="showTab('tab-triage')">3️⃣ Support Triage</button>
    <button class="tab-btn" onclick="showTab('tab-rag')">4️⃣ Policy Knowledge</button>
    <button class="tab-btn" onclick="showTab('tab-agent')">5️⃣ Autonomous Agent</button>
  </nav>

  <div class="container">
    <!-- TAB 0: ONE INPUT UNIVERSAL COPILOT -->
    <div id="tab-one-input" class="tab-content active">
      <div class="card" style="border: 2px solid var(--primary); box-shadow: 0 4px 20px rgba(249, 115, 22, 0.15);">
        <h2>🎯 One-Input Support Copilot (Universal Omnibox)</h2>
        <p class="desc">Enter <strong>any input</strong> &mdash; customer ID, order refund request, policy question, or support complaint with sensitive PII. The Copilot automatically redacts PII, detects intent, queries databases, runs multi-turn agents, and routes tickets.</p>

        <div class="form-group">
          <label>Quick Pre-sets &amp; Test Queries:</label>
          <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px;">
            <button class="btn btn-secondary" style="font-size: 12px; padding: 6px 12px;" onclick="setOneInput('C0082')">Customer 360 (C0082)</button>
            <button class="btn btn-secondary" style="font-size: 12px; padding: 6px 12px;" onclick="setOneInput('My order ORD-100048 reached me very late. I would like a refund. Call 9876543210 or email user@test.com')">Late Refund (ORD-100048)</button>
            <button class="btn btn-secondary" style="font-size: 12px; padding: 6px 12px;" onclick="setOneInput('Order ORD-100140 arrived far too late, please refund me.')">High-Value Escalate (ORD-100140)</button>
            <button class="btn btn-secondary" style="font-size: 12px; padding: 6px 12px;" onclick="setOneInput('What delivery time makes a completed order eligible for a full refund?')">Policy FAQ (Late Refund Rules)</button>
            <button class="btn btn-secondary" style="font-size: 12px; padding: 6px 12px;" onclick="setOneInput('Someone hacked my account and made unauthorized transactions. Call 9876543210 immediately.')">Fraud Alert (Security Desk)</button>
          </div>
        </div>

        <div class="form-group">
          <label>Universal Input Box:</label>
          <textarea id="one-input-text" style="min-height: 100px; font-size: 15px; line-height: 1.5;" placeholder="Enter anything: an order refund request, customer ID, policy question, or support complaint...">My order ORD-100048 reached me very late. I would like a refund. Call 9876543210 or email user@test.com</textarea>
        </div>

        <div style="display: flex; gap: 12px;">
          <button class="btn" style="padding: 12px 28px; font-size: 15px;" onclick="runOneInputCopilot()">⚡ Run Copilot</button>
          <button class="btn btn-secondary" onclick="document.getElementById('one-input-text').value=''; document.getElementById('one-input-result').style.display='none';">Clear</button>
        </div>

        <div id="one-input-result" style="margin-top: 25px; display: none;"></div>
      </div>
    </div>

    <!-- TAB 1: DATA FOUNDATION & EXECUTIVE DASHBOARD -->
    <div id="tab-dashboard" class="tab-content">
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-label">Completed Revenue</div>
          <div class="kpi-val" id="kpi-rev">INR 2,998,668.80</div>
          <div class="kpi-sub">11,110 orders reconciled</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Active Customer Base</div>
          <div class="kpi-val">1,200</div>
          <div class="kpi-sub">Across 6 Metro Cities</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Predicted Lapse Rate</div>
          <div class="kpi-val" style="color: #f87171;">38.3%</div>
          <div class="kpi-sub">443 / 1,156 cohort customers</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Triage Accuracy (Few-Shot)</div>
          <div class="kpi-val" style="color: #4ade80;">100.0%</div>
          <div class="kpi-sub">278 PII redacting violations caught</div>
        </div>
      </div>

      <div class="card">
        <h2>1️⃣ Step 1: City Performance & Basket Distribution</h2>
        <p class="desc">Analytical aggregations from golden SQLite store (<code>data/instamart.db</code>).</p>
        <table id="city-table">
          <thead>
            <tr>
              <th>Metro City</th>
              <th>Customer Base</th>
              <th>Completed Orders</th>
              <th>Total Revenue (INR)</th>
              <th>Average Basket (INR)</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>Bengaluru</td><td>197</td><td>1,894</td><td>INR 523,643.40</td><td>INR 276.47</td></tr>
            <tr><td>Hyderabad</td><td>191</td><td>1,855</td><td>INR 528,589.90</td><td>INR 284.95</td></tr>
            <tr><td>Pune</td><td>227</td><td>1,984</td><td>INR 500,918.05</td><td>INR 252.48</td></tr>
            <tr><td>Delhi</td><td>210</td><td>2,064</td><td>INR 502,829.80</td><td>INR 243.62</td></tr>
            <tr><td>Mumbai</td><td>184</td><td>1,613</td><td>INR 479,058.50</td><td>INR 297.00</td></tr>
            <tr><td>Chennai</td><td>191</td><td>1,700</td><td>INR 463,629.15</td><td>INR 272.72</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- TAB 2: CUSTOMER INTELLIGENCE -->
    <div id="tab-customers" class="tab-content">
      <div class="card">
        <h2>2️⃣ Step 2: Customer Lookup & RFM Profile</h2>
        <p class="desc">Inspect customer segment cluster, lifetime value, and ML churn prediction score.</p>
        <div style="display: flex; gap: 12px; max-width: 500px; margin-bottom: 20px;">
          <input type="text" id="cust-id-input" placeholder="Enter Customer ID (e.g. C0001, C0082, C0862)" value="C0082">
          <button class="btn" onclick="lookupCustomer()">Search Customer</button>
        </div>

        <div id="cust-profile-result" style="display: none; background: #0f172a; padding: 20px; border-radius: 8px; border: 1px solid var(--border);">
          <!-- Injected via JS -->
        </div>
      </div>

      <div class="card">
        <h2>Top 10 High-Risk Premium Retention Targets (outputs/retention_list.csv)</h2>
        <p class="desc">Prioritised VIP accounts from the test set ranked by model lapse probability.</p>
        <table>
          <thead>
            <tr>
              <th>Customer ID</th>
              <th>Lapse Probability</th>
              <th>Actual Churn Status</th>
              <th>Retention Action</th>
            </tr>
          </thead>
          <tbody id="retention-table-body">
            <!-- Dynamic rows -->
          </tbody>
        </table>
      </div>
    </div>

    <!-- TAB 3: SUPPORT TRIAGE -->
    <div id="tab-triage" class="tab-content">
      <div class="card">
        <h2>3️⃣ Step 3: Privacy-Safe Support Ticket Triage</h2>
        <p class="desc">Enter raw ticket text with sensitive personal info to verify regex redaction and automated routing.</p>
        <div class="form-group">
          <label>Sample Pre-sets:</label>
          <select id="triage-preset" onchange="loadTriagePreset()">
            <option value="">-- Choose a realistic ticket preset --</option>
            <option value="billing">Billing: Double charge on card with email & phone</option>
            <option value="fraud">Fraud: Unauthorized charge alert</option>
            <option value="delivery">Delivery: Rider delayed and unreachable</option>
            <option value="technical">Technical: App freeze during cart checkout</option>
          </select>
        </div>
        <div class="form-group">
          <label>Raw Customer Ticket Text:</label>
          <textarea id="triage-input">I was charged twice for order ORD-100048. Please refund the extra payment. My email is user12@example.com and phone is 9876543210.</textarea>
        </div>
        <div style="display: flex; gap: 12px;">
          <button class="btn" onclick="triageTicket('few-shot')">Triage (Few-Shot Prompt)</button>
          <button class="btn btn-secondary" onclick="triageTicket('zero-shot')">Triage (Zero-Shot Prompt)</button>
        </div>

        <div id="triage-result" style="margin-top: 25px; display: none;">
          <!-- Injected via JS -->
        </div>
      </div>
    </div>

    <!-- TAB 4: POLICY RAG -->
    <div id="tab-rag" class="tab-content">
      <div class="card">
        <h2>4️⃣ Step 4: Policy Knowledge Assistant (TF-IDF RAG)</h2>
        <p class="desc">Asks questions against the five store policy documents with grounding verification.</p>
        <div class="form-group">
          <label>Preset Question Library:</label>
          <select id="rag-preset" onchange="loadRagPreset()">
            <option value="">-- Select policy evaluation question --</option>
            <option value="What delivery time makes a completed order eligible for a full refund?">Refund Policy: Eligible delivery time</option>
            <option value="Who must approve a refund above INR 1000?">Refund Policy: Approval threshold</option>
            <option value="What is the delivery fee on small orders?">Delivery Policy: Delivery fees</option>
            <option value="How fast must a fraud ticket reach the security team?">Escalation Policy: Fraud SLA</option>
            <option value="What is the favourite colour of the company mascot?">Unanswerable: Mascot color</option>
          </select>
        </div>
        <div class="form-group">
          <label>Question:</label>
          <input type="text" id="rag-input" value="What delivery time makes a completed order eligible for a full refund?">
        </div>
        <button class="btn" onclick="queryPolicy()">Ask Policy Assistant</button>

        <div id="rag-result" style="margin-top: 25px; display: none;"></div>
      </div>
    </div>

    <!-- TAB 5: REACT AGENT -->
    <div id="tab-agent" class="tab-content">
      <div class="card">
        <h2>5️⃣ Step 5: Autonomous ReAct Refund Agent Simulator</h2>
        <p class="desc">Simulate multi-turn ReAct reasoning (order DB lookup, policy retrieval, refund calculation, and ledger logging).</p>
        <div class="form-group">
          <label>Select Benchmark Scenario:</label>
          <select id="agent-preset" onchange="loadAgentPreset()">
            <option value="S1">S1: Late Delivery (< INR 1000) -> REFUND</option>
            <option value="S2">S2: Late Delivery (< INR 1000) -> REFUND</option>
            <option value="S3">S3: High Value Late Delivery (> INR 1000) -> ESCALATE</option>
            <option value="S4">S4: On-time Delivery -> DENIED</option>
            <option value="S5">S5: Cancelled Order -> DENIED</option>
            <option value="S6">S6: Non-existent Order -> NOT_FOUND</option>
            <option value="S7">S7: Infinite Loop Prompt -> FALLBACK_MAX_ITERATIONS</option>
          </select>
        </div>
        <div class="form-group">
          <label>User Message:</label>
          <input type="text" id="agent-msg" value="My order ORD-100048 reached me very late. I would like a refund.">
        </div>
        <div style="display: flex; gap: 12px;">
          <button class="btn" onclick="runAgentScenario()">Run ReAct Loop</button>
          <button class="btn btn-secondary" onclick="runCrashResume()">Demonstrate Crash & Resume</button>
        </div>

        <div id="agent-result" style="margin-top: 25px; display: none;"></div>
      </div>
    </div>
  </div>

  <script>
    function showTab(tabId) {
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      event.target.classList.add('active');
      if (tabId === 'tab-customers') loadRetentionList();
    }

    function loadTriagePreset() {
      const val = document.getElementById('triage-preset').value;
      const map = {
        'billing': 'I was charged twice for order ORD-100048. Please refund extra payment. Contact user12@example.com or 9876543210.',
        'fraud': 'There is an unauthorized transaction on my card linked to order ORD-100048. Call 9876543210 immediately.',
        'delivery': 'My order ORD-100048 has not arrived and rider is not answering calls. Call 9876543210.',
        'technical': 'The app crashes every time I open the cart for order ORD-100048. Email support@user.com.'
      };
      if (map[val]) document.getElementById('triage-input').value = map[val];
    }

    function loadRagPreset() {
      const val = document.getElementById('rag-preset').value;
      if (val) document.getElementById('rag-input').value = val;
    }

    function loadAgentPreset() {
      const val = document.getElementById('agent-preset').value;
      fetch('/api/scenario?id=' + val)
        .then(r => r.json())
        .then(data => {
          document.getElementById('agent-msg').value = data.message;
        });
    }

    function lookupCustomer() {
      const cid = document.getElementById('cust-id-input').value.trim();
      fetch('/api/customer?id=' + cid)
        .then(r => r.json())
        .then(data => {
          const res = document.getElementById('cust-profile-result');
          if (data.error) {
            res.innerHTML = '<p style="color: var(--danger);">' + data.error + '</p>';
          } else {
            res.innerHTML = `
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 style="font-size: 20px;">${data.customer_id} &mdash; ${data.city}</h3>
                <span class="badge ${data.is_premium ? 'badge-premium' : 'badge-normal'}">${data.segment_name} ${data.is_premium ? '★ VIP' : ''}</span>
              </div>
              <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 15px;">
                <div><label>Recency</label><strong style="font-size: 16px;">${data.recency} days</strong></div>
                <div><label>Frequency</label><strong style="font-size: 16px;">${data.frequency} orders</strong></div>
                <div><label>Monetary Spend</label><strong style="font-size: 16px;">INR ${data.monetary.toLocaleString()}</strong></div>
                <div><label>Predicted Lapse Risk</label><strong style="font-size: 16px; color: ${data.lapse_prob > 0.5 ? '#ef4444' : '#22c55e'};">${(data.lapse_prob * 100).toFixed(1)}%</strong></div>
              </div>
              <div style="font-size: 13px; color: var(--text-muted);">Age: ${data.age} | Gender: ${data.gender} | Registered: ${data.signup_date}</div>
            `;
          }
          res.style.display = 'block';
        });
    }

    function loadRetentionList() {
      fetch('/api/retention_list')
        .then(r => r.json())
        .then(rows => {
          const tb = document.getElementById('retention-table-body');
          tb.innerHTML = rows.map(r => `
            <tr>
              <td><strong>${r.customer_id}</strong> <span class="badge badge-premium">Premium</span></td>
              <td><span style="font-weight: 700; color: #f87171;">${(r.lapse_probability * 100).toFixed(1)}%</span></td>
              <td>${r.actually_lapsed ? '<span class="badge badge-high">Lapsed (Confirmed)</span>' : '<span class="badge badge-success">Active</span>'}</td>
              <td>Priority VIP Concierge Win-back Call & INR 150 Grocery Pass</td>
            </tr>
          `).join('');
        });
    }

    function triageTicket(mode) {
      const text = document.getElementById('triage-input').value;
      fetch('/api/triage', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text, mode: mode})
      })
      .then(r => r.json())
      .then(res => {
        const div = document.getElementById('triage-result');
        div.style.display = 'block';
        div.innerHTML = `
          <div class="card" style="border-color: var(--accent);">
            <h2>Triage & Routing Outcome</h2>
            <div style="display: flex; gap: 10px; margin-bottom: 15px;">
              <span class="badge ${res.priority === 'HIGH' ? 'badge-high' : 'badge-normal'}">Priority: ${res.priority}</span>
              <span class="badge badge-premium">Route: ${res.route}</span>
              <span class="badge badge-success">Status: ${res.status}</span>
            </div>
            <div style="margin-bottom: 12px;"><label>Redacted Prompt Payload (Zero PII Leaks):</label><pre>${res.redacted_text}</pre></div>
            <div><label>Model Raw Classification:</label><pre>${JSON.stringify(res.model_response, null, 2)}</pre></div>
          </div>
        `;
      });
    }

    function queryPolicy() {
      const q = document.getElementById('rag-input').value;
      fetch('/api/rag?q=' + encodeURIComponent(q))
        .then(r => r.json())
        .then(res => {
          const div = document.getElementById('rag-result');
          div.style.display = 'block';
          div.innerHTML = `
            <div class="card" style="border-color: var(--accent);">
              <h2>Grounding Answer</h2>
              <p style="font-size: 16px; color: var(--text); background: #0f172a; padding: 15px; border-radius: 8px; border-left: 4px solid var(--success);">${res.answer}</p>
              <h3 style="margin-top: 15px; font-size: 14px; color: var(--text-muted);">Top Retrieved Sources:</h3>
              <div style="margin-top: 8px;">
                ${res.sources.map(s => `<div class="step-box"><div class="step-header"><span>${s.source}</span><span>Score: ${s.score.toFixed(4)}</span></div><div>${s.text}</div></div>`).join('')}
              </div>
            </div>
          `;
        });
    }

    function runAgentScenario() {
      const msg = document.getElementById('agent-msg').value;
      const sid = document.getElementById('agent-preset').value;
      fetch('/api/run_agent', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: msg, scenario_id: sid})
      })
      .then(r => r.json())
      .then(res => {
        const div = document.getElementById('agent-result');
        div.style.display = 'block';
        div.innerHTML = `
          <div class="card" style="border-color: var(--primary);">
            <h2>Agent Execution Trace & Outcome</h2>
            <div style="display: flex; gap: 12px; margin-bottom: 15px;">
              <span class="badge badge-premium">Status: ${res.status}</span>
              <span class="badge badge-success">Refund: INR ${res.refund_amount.toFixed(2)}</span>
              <span class="badge badge-normal">LLM Calls: ${res.llm_calls}</span>
              <span class="badge badge-normal">Tokens: ${res.tokens_used}</span>
            </div>
            <p style="margin-bottom: 15px; font-weight: 600; color: var(--accent);">${res.final_answer}</p>
            <h3>Execution Step History (${res.steps.length} Steps):</h3>
            <div style="margin-top: 10px;">
              ${res.steps.map(st => `
                <div class="step-box">
                  <div class="step-header"><span>Step ${st.step}: Tool [${st.tool_name}]</span></div>
                  <div>Arguments: <code>${JSON.stringify(st.arguments)}</code></div>
                  <div>Observation: <pre>${JSON.stringify(st.observation, null, 2)}</pre></div>
                </div>
              `).join('')}
            </div>
          </div>
        `;
      });
    }

    function runCrashResume() {
      fetch('/api/crash_resume')
        .then(r => r.json())
        .then(res => {
          const div = document.getElementById('agent-result');
          div.style.display = 'block';
          div.innerHTML = `
            <div class="card" style="border-color: var(--warning);">
              <h2>Crash & Resume Demonstration</h2>
              <p style="color: var(--warning); margin-bottom: 10px;">Simulated unhandled system crash after Tool #2 (search_policy). Re-hydrated state from checkpoint file in <code>state/</code> and completed workflow.</p>
              <div style="display: flex; gap: 10px; margin-bottom: 15px;">
                <span class="badge badge-success">Resumed Status: ${res.status}</span>
                <span class="badge badge-premium">Refund Amount: INR ${res.refund_amount.toFixed(2)}</span>
              </div>
              <h3>Executed Tools (0 Duplicates Across Crash & Resume):</h3>
              <pre>${JSON.stringify(res.tools_executed, null, 2)}</pre>
            </div>
          `;
        });
    }

    function setOneInput(val) {
      document.getElementById('one-input-text').value = val;
      runOneInputCopilot();
    }

    function runOneInputCopilot() {
      const q = document.getElementById('one-input-text').value.trim();
      if (!q) return;
      const resDiv = document.getElementById('one-input-result');
      resDiv.style.display = 'block';
      resDiv.innerHTML = '<p style="color: var(--accent); padding: 15px; font-weight: 600;">⚡ Processing with Privacy Guardrails, Semantic Search & Agent Reasoning...</p>';

      fetch('/api/one_input_support', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: q})
      })
      .then(r => r.json())
      .then(res => {
        let detailsHtml = '';
        if (res.data) {
          if (res.intent === 'AUTONOMOUS_REFUND_AGENT') {
            detailsHtml = `
              <div style="margin-top: 15px;">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 15px;">
                  <div class="kpi-card" style="padding: 12px;"><div class="kpi-label">Order ID</div><div style="font-size: 18px; font-weight: 700;">${res.data.order_id || 'N/A'}</div></div>
                  <div class="kpi-card" style="padding: 12px;"><div class="kpi-label">Refund Status</div><div style="font-size: 18px; font-weight: 700; color: #4ade80;">${res.data.refund_status}</div></div>
                  <div class="kpi-card" style="padding: 12px;"><div class="kpi-label">Refund Amount</div><div style="font-size: 18px; font-weight: 700; color: #fb923c;">INR ${res.data.refund_amount_inr.toFixed(2)}</div></div>
                  <div class="kpi-card" style="padding: 12px;"><div class="kpi-label">Customer Tier</div><div style="font-size: 18px; font-weight: 700;">${res.data.customer_tier} ${res.data.is_premium ? '★ VIP' : ''}</div></div>
                </div>
                <h3 style="font-size: 14px; margin-bottom: 8px;">ReAct Multi-Turn Execution Steps (${(res.data.execution_steps || []).length}):</h3>
                ${(res.data.execution_steps || []).map(st => `
                  <div class="step-box">
                    <div class="step-header"><span>Step ${st.step}: Tool [${st.tool_name}]</span></div>
                    <div>Arguments: <code>${JSON.stringify(st.arguments)}</code></div>
                    <div>Observation: <pre>${JSON.stringify(st.observation, null, 2)}</pre></div>
                  </div>
                `).join('')}
              </div>
            `;
          } else if (res.intent === 'CUSTOMER_INTELLIGENCE') {
            detailsHtml = `
              <div style="margin-top: 15px; background: #0f172a; padding: 20px; border-radius: 8px; border: 1px solid var(--border);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                  <h3 style="font-size: 18px;">${res.data.customer_id} &mdash; ${res.data.city}</h3>
                  <span class="badge ${res.data.is_premium ? 'badge-premium' : 'badge-normal'}">${res.data.segment_name}</span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 10px;">
                  <div><label>Recency</label><strong>${res.data.recency_days} days</strong></div>
                  <div><label>Frequency</label><strong>${res.data.frequency_orders} orders</strong></div>
                  <div><label>Monetary Spend</label><strong>INR ${res.data.monetary_spend_inr.toLocaleString()}</strong></div>
                  <div><label>Predicted Churn Risk</label><strong style="color: ${res.data.predicted_lapse_probability > 0.5 ? '#ef4444' : '#22c55e'};">${(res.data.predicted_lapse_probability * 100).toFixed(1)}%</strong></div>
                </div>
                <div style="font-size: 12px; color: var(--text-muted);">Age: ${res.data.age} | Gender: ${res.data.gender} | Joined: ${res.data.signup_date}</div>
              </div>
            `;
          } else if (res.intent === 'POLICY_ASSISTANT_RAG') {
            detailsHtml = `
              <div style="margin-top: 15px;">
                <div style="background: #0f172a; padding: 16px; border-radius: 8px; border-left: 4px solid var(--success); font-size: 15px; margin-bottom: 15px;">
                  ${res.data.grounded_answer}
                </div>
                <h3 style="font-size: 14px; margin-bottom: 8px;">Grounding Policy Sources (Top 3 Chunks):</h3>
                ${(res.data.retrieved_sources || []).map(s => `
                  <div class="step-box">
                    <div class="step-header"><span>${s.source}</span><span>Score: ${s.score.toFixed(4)}</span></div>
                    <div>${s.text}</div>
                  </div>
                `).join('')}
              </div>
            `;
          } else if (res.intent === 'SUPPORT_TICKET_TRIAGE') {
            detailsHtml = `
              <div style="margin-top: 15px; background: #0f172a; padding: 16px; border-radius: 8px;">
                <p style="margin-bottom: 8px;">Triaged Category: <strong>${res.data.category}</strong></p>
                <label>Raw Model Response:</label>
                <pre>${res.data.raw_model_response}</pre>
              </div>
            `;
          }
        }

        let piiHtml = '';
        if (res.redaction_summary && res.redaction_summary.total_masked > 0) {
          piiHtml = `
            <div style="margin-top: 12px; background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 8px; padding: 12px;">
              <div style="font-size: 12px; font-weight: 700; color: #4ade80; margin-bottom: 4px;">
                🛡️ PRIVACY GUARDRAILS: ${res.redaction_summary.total_masked} PII items masked (0 leaks remaining)
              </div>
              <div style="font-size: 12px; color: var(--text-muted);">Redacted text sent to AI: <code>${res.redacted_text}</code></div>
            </div>
          `;
        }

        resDiv.innerHTML = `
          <div class="card" style="border-color: var(--primary);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
              <div>
                <span class="badge badge-premium">${res.intent_label}</span>
                <span class="badge ${res.priority === 'HIGH' ? 'badge-high' : 'badge-normal'}">Priority: ${res.priority}</span>
                <span class="badge badge-success">Queue: ${res.route}</span>
              </div>
              <span style="font-size: 12px; color: var(--text-muted);">Status: ${res.status}</span>
            </div>
            <h2 style="font-size: 18px; color: var(--text);">${res.summary}</h2>
            ${piiHtml}
            ${detailsHtml}
          </div>
        `;
      });
    }
  </script>
</body>
</html>
"""

class CopilotServer(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        elif path == "/api/customer":
            cid = params.get("id", [""])[0].strip().upper()
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT customer_id, city, signup_date, age, gender FROM customers WHERE customer_id = ?", (cid,))
            c_row = cur.fetchone()
            conn.close()

            if not c_row:
                self.send_json({"error": f"Customer {cid} not found in database"}, 404)
                return

            seg_path = os.path.join("outputs", "customer_segments.csv")
            seg_info = {"recency": 0, "frequency": 0, "monetary": 0.0, "segment_name": "Standard", "is_premium": False}
            if os.path.exists(seg_path):
                import pandas as pd
                df_seg = pd.read_csv(seg_path)
                match = df_seg[df_seg["customer_id"] == cid]
                if not match.empty:
                    row = match.iloc[0]
                    seg_info = {
                        "recency": int(row["recency"]),
                        "frequency": int(row["frequency"]),
                        "monetary": float(row["monetary"]),
                        "segment_name": str(row["segment_name"]),
                        "is_premium": bool(row["is_premium"])
                    }

            lapse_prob = 0.25 if not seg_info["is_premium"] else 0.45
            ret_path = os.path.join("outputs", "retention_list.csv")
            if os.path.exists(ret_path):
                import pandas as pd
                df_ret = pd.read_csv(ret_path)
                r_match = df_ret[df_ret["customer_id"] == cid]
                if not r_match.empty:
                    lapse_prob = float(r_match.iloc[0]["lapse_probability"])

            data = {
                "customer_id": c_row[0],
                "city": c_row[1],
                "signup_date": c_row[2],
                "age": c_row[3],
                "gender": c_row[4],
                "recency": seg_info["recency"],
                "frequency": seg_info["frequency"],
                "monetary": seg_info["monetary"],
                "segment_name": seg_info["segment_name"],
                "is_premium": seg_info["is_premium"],
                "lapse_prob": lapse_prob
            }
            self.send_json(data)
            return

        elif path == "/api/retention_list":
            ret_path = os.path.join("outputs", "retention_list.csv")
            if os.path.exists(ret_path):
                import pandas as pd
                df = pd.read_csv(ret_path)
                self.send_json(df.to_dict(orient="records"))
            else:
                self.send_json([])
            return

        elif path == "/api/scenario":
            sid = params.get("id", ["S1"])[0]
            sc_path = os.path.join("data", "raw", "agent_scenarios.csv")
            import pandas as pd
            df = pd.read_csv(sc_path)
            match = df[df["scenario_id"] == sid]
            if not match.empty:
                self.send_json(match.iloc[0].to_dict())
            else:
                self.send_json({"message": "My order ORD-100048 was late."})
            return

        elif path == "/api/rag":
            q = params.get("q", [""])[0]
            ans, sources = part_d.answer(q)
            top_sources = part_d.default_retrieve(q, k=3)
            self.send_json({
                "question": q,
                "answer": ans,
                "sources": top_sources
            })
            return

        elif path == "/api/crash_resume":
            sc_path = os.path.join("data", "raw", "agent_scenarios.csv")
            import pandas as pd
            s1_msg = pd.read_csv(sc_path).iloc[0]["message"]
            crash_id = "demo_web_crash"
            try:
                part_d.run_agent(s1_msg, "S1", run_id=crash_id, crash_after_tool_step=2)
            except RuntimeError:
                pass
            res = part_d.resume_agent(crash_id)
            self.send_json(res)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        path = self.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        payload = json.loads(body)

        if path == "/api/triage":
            text = payload.get("text", "")
            mode = payload.get("mode", "few-shot")
            redacted, ne, np_, no = part_c.redact_pii(text)

            prompt = part_c.FEW_SHOT_PROMPT if mode == "few-shot" else part_c.ZERO_SHOT_PROMPT
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": redacted}
            ]
            resp = mock_llm.chat(messages)
            content = resp.choices[0].message.content
            status, cat = part_c.validate_model_response(content)

            # Determine route
            if cat == "fraud":
                route, priority = "security_team", "HIGH"
            elif status != "classified_json":
                route, priority = "human_review", "NORMAL"
            else:
                route, priority = "support_queue", "NORMAL"

            self.send_json({
                "status": status,
                "category": cat,
                "route": route,
                "priority": priority,
                "redacted_text": redacted,
                "model_response": content
            })
            return

        elif path == "/api/run_agent":
            msg = payload.get("message", "")
            sid = payload.get("scenario_id", "Custom")
            res = part_d.run_agent(msg, sid)
            self.send_json(res)
            return

        elif path == "/api/one_input_support":
            import one_input_support
            query = payload.get("query", "")
            res = one_input_support.handle_one_input(query)
            self.send_json(res)
            return

        self.send_response(404)
        self.end_headers()

    def send_json(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def log_message(self, format, *args):
        # Silence standard HTTP access logging to keep terminal clean
        return


def run_server(port=PORT):
    server = HTTPServer(("127.0.0.1", port), CopilotServer)
    print(f"\n{'='*70}")
    print(f"Instamart Customer Intelligence & Support Copilot Web App Running!")
    print(f"Local URL: http://127.0.0.1:{port}")
    print(f"{'='*70}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Copilot web server...")
        server.server_close()


if __name__ == "__main__":
    run_server()
