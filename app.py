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


# --- Main: Metrics ---
st.header("Training Metrics")

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
        st.error(f"Error reading logs: {e}")
        
    return pd.DataFrame(metrics), pd.DataFrame(samples)

df_metrics, df_samples = load_data()

if not df_metrics.empty:
    # Extract useful columns (like reward, loss)
    # TRL logs often have keys like 'reward', 'ppo/loss/policy', etc.
    # We'll try to find relevant columns dynamically
    
    cols = df_metrics.columns
    
    # 1. Rewards
    reward_cols = [c for c in cols if "reward" in c or "score" in c]
    if reward_cols:
        st.subheader("Rewards")
        chart_r = alt.Chart(df_metrics).mark_line().encode(
            x='step',
            y=reward_cols[0],
            tooltip=['step', reward_cols[0]]
        ).interactive()
        st.altair_chart(chart_r, use_container_width=True)
        
    # 2. Loss
    loss_cols = [c for c in cols if "loss" in c]
    if loss_cols:
        st.subheader("Loss")
        selected_loss = st.selectbox("Select Loss Metric", loss_cols)
        chart_l = alt.Chart(df_metrics).mark_line().encode(
            x='step',
            y=selected_loss,
            tooltip=['step', selected_loss]
        ).interactive()
        st.altair_chart(chart_l, use_container_width=True)

    # 3. KL
    kl_cols = [c for c in cols if "kl" in c]
    if kl_cols:
         st.subheader("KL Divergence")
         chart_k = alt.Chart(df_metrics).mark_line().encode(
            x='step',
            y=kl_cols[0],
            tooltip=['step', kl_cols[0]]
        ).interactive()
         st.altair_chart(chart_k, use_container_width=True)

else:
    st.info("No metrics logged yet. Start a job!")

# --- Main: Samples ---
st.header("Latest Generated Samples")
if not df_samples.empty:
    # Show last 5 samples
    recent = df_samples.tail(10)
    
    for i, row in recent.iterrows():
        with st.expander(f"Sample {i} (Reward: {row.get('reward', 0.0)})"):
            st.markdown(f"**Question:**\n{row.get('question')}")
            st.markdown(f"**Model Answer:**\n{row.get('response')}")
else:
    st.info("No samples generated yet.")

# Auto refresh
st.empty()
import time
# Rerun button usually better than infinite loop in Streamlit
if st.button("Refresh Data"):
    st.rerun()
