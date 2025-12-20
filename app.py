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

import shutil
import signal

# Detect Environment
HAS_SLURM = shutil.which("sbatch") is not None
MODE_LABEL = "Cluster" if HAS_SLURM else "Local"

# --- Sidebar: Job Control ---
st.sidebar.title(f"🎮 Control Panel ({MODE_LABEL})")

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

# Local Process Helpers
PID_FILE = "local_run.pid"
def start_local():
    # Run in background, redirect output to a file so we don't block
    # Using 'nohup' equivalent
    with open("local_log.txt", "w") as out:
        proc = subprocess.Popen(["python3", "train.py", "--mode", "demo"], stdout=out, stderr=out)
        with open(PID_FILE, "w") as f:
            f.write(str(proc.pid))
    return f"Started Local Process (PID {proc.pid})"

def stop_local():
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
            os.remove(PID_FILE)
            return f"Killed PID {pid}"
        except Exception as e:
            return f"Error killing: {e}"
    return "No local PID found."

def check_local_status():
    if os.path.exists(PID_FILE):
        # Check if actually running
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())
            os.kill(pid, 0) # Check existence
            return f"🟢 Running (PID {pid})"
        except OSError:
            return "🔴 Crashed/Stopped (Stale PID)"
    return "⚪ Idle"


# Section 1: Actions
st.sidebar.markdown("### 1. Actions")
col_s1, col_s2 = st.sidebar.columns(2)

with col_s1:
    btn_label = "▶️ Start Job" if HAS_SLURM else "▶️ Start Local"
    if st.button(btn_label, help=f"Start training on {MODE_LABEL}"):
        with st.spinner("Starting..."):
            if HAS_SLURM:
                out = run_command(f"sbatch {JOB_SCRIPT}")
            else:
                out = start_local()
        st.sidebar.success(out)

with col_s2:
    if st.button("🛑 Stop", help="Stop current training run"):
        with st.spinner("Stopping..."):
            if HAS_SLURM:
                user = os.environ.get("USER", "vavaghad")
                out = run_command(f"scancel -u {user}")
            else:
                out = stop_local()
        st.sidebar.warning(out)

# Section 2: Status
st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Status")

if HAS_SLURM:
    # Auto-check status on refresh
    labels = run_command("squeue --me --format='%.8i %.9P %.8j %.2t %.10M'")
    if labels and "JOBID" in labels:
        st.sidebar.info("🟢 Job Active")
        st.sidebar.code(labels)
    else:
        st.sidebar.caption("⚪ No Active Jobs")
else:
    status_msg = check_local_status()
    st.sidebar.info(status_msg)

if st.sidebar.button("🔄 Force Refresh"):
    st.rerun()



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
