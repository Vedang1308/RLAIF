import streamlit as st
import pandas as pd
import json
import os
import subprocess
import time
import altair as alt

# Config
LOG_FILE = "logs/metrics.jsonl"
JOB_SCRIPT = "job_launcher.sh"

st.set_page_config(page_title="RLAIF Control Center", layout="wide")
st.title("🚀 RLAIF Training Control Center")

# --- Sidebar: Job Control ---
st.sidebar.header("Cluster Control")

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

if st.sidebar.button("Start Research Job"):
    out = run_command(f"sbatch {JOB_SCRIPT}")
    st.sidebar.success(f"Submitted: {out}")

if st.sidebar.button("Stop All My Jobs"):
    # Cancels all jobs by this user
    user = os.environ.get("USER", "vavaghad")
    out = run_command(f"scancel -u {user}")
    st.sidebar.warning(f"Stopped: {out}")

# Status Check
st.sidebar.subheader("Job Status")
if st.sidebar.button("Refresh Status"):
    labels = run_command("squeue --me --format='%.8i %.9P %.8j %.8u %.2t %.10M %.6D %R'")
    st.sidebar.code(labels if labels else "No active jobs.")
else:
    # Auto-check if we can (careful with lag)
    pass


# --- Main: Dashboard ---
st.header("Training Dashboard")

def load_data():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(), pd.DataFrame()
    
    metrics = []
    samples = []
    
    try:
        with open(LOG_FILE, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("type") == "metrics":
                        metrics.append(data)
                    elif data.get("type") == "sample":
                        samples.append(data)
                except:
                    continue
    except Exception as e:
        pass # Handle read errors gracefully during writes
        
    return pd.DataFrame(metrics), pd.DataFrame(samples)

df_metrics, df_samples = load_data()

# 1. KPI Row (Big Numbers)
kpi1, kpi2, kpi3 = st.columns(3)

if not df_metrics.empty:
    latest = df_metrics.iloc[-1]
    
    # 1. Rewards
    cols = df_metrics.columns
    reward_col = next((c for c in cols if "reward" in c or "score" in c), None)
    loss_col = next((c for c in cols if "loss" in c), None)
    
    current_step = int(latest.get("step", 0))
    current_reward = float(latest[reward_col]) if reward_col else 0.0
    current_loss = float(latest[loss_col]) if loss_col else 0.0
    
    kpi1.metric("Current Step", f"{current_step}", delta=None)
    kpi2.metric("avg Reward", f"{current_reward:.3f}", delta=f"{current_reward:.3f}")
    kpi3.metric("Loss", f"{current_loss:.4f}", delta_color="inverse")
    
    st.markdown("---")
    
    # 2. Charts Row
    c1, c2 = st.columns(2)
    
    if reward_col:
        c1.subheader("Reward History")
        chart_r = alt.Chart(df_metrics).mark_line(color='green').encode(
            x='step', 
            y=alt.Y(reward_col, title='Reward'),
            tooltip=['step', reward_col]
        ).interactive()
        c1.altair_chart(chart_r, use_container_width=True)
        
    kl_col = next((c for c in cols if "kl" in c), None)
    if kl_col:
        c2.subheader("KL Divergence")
        chart_k = alt.Chart(df_metrics).mark_line(color='orange').encode(
            x='step', 
            y=alt.Y(kl_col, title='KL Div'),
            tooltip=['step', kl_col]
        ).interactive()
        c2.altair_chart(chart_k, use_container_width=True)

else:
    st.info("Waiting for metrics... (Job might be starting)")

# 3. Samples Inspector
st.subheader("📝 Model Output Inspector")
if not df_samples.empty:
    recent = df_samples.tail(5)[::-1] # Newest first
    for i, row in recent.iterrows():
        r_val = row.get('reward', 0.0)
        color = "🟢" if r_val > 0.5 else "🔴"
        with st.expander(f"{color} Reward: {r_val:.2f} | Q: {row.get('question', '')[:50]}..."):
            st.markdown(f"**Question:** {row.get('question')}")
            st.markdown(f"**Answer:** {row.get('response')}")
            st.caption(f"Generated at Step {row.get('step', '?')} | {time.ctime(row.get('timestamp', 0))}")
else:
    st.caption("No samples yet.")

# Auto Refresh logic
if st.checkbox("Auto-Refresh (2s)", value=True):
    time.sleep(2)
    st.rerun()

st.button("Manual Refresh")
