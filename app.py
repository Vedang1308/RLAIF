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
import sys

def start_local():
    # Run in background, redirect output to a file so we don't block
    # Using 'nohup' equivalent
    # Use sys.executable to ensure we use the SAME python that runs streamlit
    
    # Authenticate WandB for local run
    env = os.environ.copy()
    env["WANDB_API_KEY"] = "ef2da50d021e41130a9c9d762f7e56c79dbed703"
    env["WANDB_PROJECT"] = "rlaif-research" # Ensure project name is set
    
    with open("local_log.txt", "w") as out:
        proc = subprocess.Popen(
            [sys.executable, "train.py", "--mode", "demo"], 
            stdout=out, 
            stderr=out,
            env=env # Pass auth
        )
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

# Determine state
is_running = False
if HAS_SLURM:
    # Quick check (cached if possible, but for safety we run it)
    # We might rely on the labels from below, but let's do a quick check here
    check = run_command("squeue --me --noheader")
    if check and len(check.strip()) > 0:
        is_running = True
else:
    # Local check
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            is_running = True
        except:
            is_running = False

with col_s1:
    btn_label = "▶️ Start Job" if HAS_SLURM else "▶️ Start Local"
    
    if is_running:
        st.button("⚠️ Running...", disabled=True, help="Job is already active. Stop it first.")
    else:
        if st.button(btn_label, help=f"Start training on {MODE_LABEL}"):
            # 1. Clear old logs on new run
            if os.path.exists(LOG_FILE):
                try:
                    os.remove(LOG_FILE)
                except:
                    pass
            
            # 2. Start Job
            with st.spinner("Starting..."):
                if HAS_SLURM:
                    out = run_command(f"sbatch {JOB_SCRIPT}")
                else:
                    out = start_local()
            # Force reload to update state immediately
            st.sidebar.success(out)
            time.sleep(1)
            st.rerun()

with col_s2:
    if st.button("🛑 Stop", help="Stop current training run"):
        with st.spinner("Stopping..."):
            if HAS_SLURM:
                user = os.environ.get("USER", "vavaghad")
                out = run_command(f"scancel -u {user}")
            else:
                out = stop_local()
        st.sidebar.warning(out)
        time.sleep(1)
        st.rerun()

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

# --- Live Console Logic (Sidebar) ---
def get_log_content():
    # 1. Determine Log File
    log_file = None
    if HAS_SLURM:
        import glob
        files = glob.glob("slurm-*.out")
        if files:
            log_file = max(files, key=os.path.getctime)
    else:
        if os.path.exists("local_log.txt"):
            log_file = "local_log.txt"
    
    if not log_file:
        return "Waiting for logs..."
    
    try:
        with open(log_file, "r") as f:
            f.seek(0, os.SEEK_END)
            filesize = f.tell()
            if filesize < 5000:
                f.seek(0)
                return f.read()
            else:
                f.seek(max(filesize - 5000, 0))
                return f.read()[-5000:] # Last 5k chars roughly
    except Exception as e:
        return f"Error: {e}"

st.sidebar.markdown("---")
with st.sidebar.expander("🖥️ Live Logs", expanded=True):
    st.code(get_log_content(), language="text")



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
        pass 
        
    return pd.DataFrame(metrics), pd.DataFrame(samples)

df_metrics, df_samples = load_data()

# 1. Top-Level KPIs (Always Visible)
kpi1, kpi2, kpi3 = st.columns(3)
if not df_metrics.empty:
    latest = df_metrics.iloc[-1]
    cols = df_metrics.columns
    reward_col = next((c for c in cols if "reward" in c or "score" in c), None)
    loss_col = next((c for c in cols if "loss" in c), None)
    
    current_step = int(latest.get("step", 0))
    current_reward = float(latest[reward_col]) if reward_col else 0.0
    current_loss = float(latest[loss_col]) if loss_col else 0.0
    
    kpi1.metric("Current Step", f"{current_step}")
    kpi2.metric("avg Reward", f"{current_reward:.3f}")
    kpi3.metric("Loss", f"{current_loss:.4f}")
else:
    st.info("Waiting for first metrics...")

st.markdown("---")

# 2. Main Visuals in Tabs
tab1, tab2, tab3 = st.tabs(["📊 Metrics", "📝 Samples", "ℹ️ Help"])

with tab1:
    if not df_metrics.empty:
        c1, c2 = st.columns(2)
        if reward_col:
            c1.subheader("Reward History")
            chart_r = alt.Chart(df_metrics).mark_line(color='green').encode(
                x='step', y=alt.Y(reward_col, title='Reward'), tooltip=['step', reward_col]
            ).interactive()
            c1.altair_chart(chart_r, use_container_width=True)
            
        kl_col = next((c for c in cols if "kl" in c), None)
        if kl_col:
            c2.subheader("KL Divergence")
            chart_k = alt.Chart(df_metrics).mark_line(color='orange').encode(
                x='step', y=alt.Y(kl_col, title='KL Div'), tooltip=['step', kl_col]
            ).interactive()
            c2.altair_chart(chart_k, use_container_width=True)
    else:
        st.write("Charts will appear here once training starts.")

with tab2:
    if not df_samples.empty:
        recent = df_samples.tail(10)[::-1]
        for i, row in recent.iterrows():
            r_val = row.get('reward', 0.0)
            color = "🟢" if r_val > 0.5 else "🔴"
            with st.expander(f"{color} Reward: {r_val:.2f} | Q: {row.get('question', '')[:50]}..."):
                st.markdown(f"**Question:** {row.get('question')}")
                st.markdown(f"**Answer:** {row.get('response')}")
                st.caption(f"Step {row.get('step', '?')}")
    else:
        st.write("No samples yet.")

with tab3:
    st.markdown("""
    ### How to Use
    1. **Start**: Use the Sidebar (left) to click "▶️ Start Local" (Mac) or "Start Job" (Cluster).
    2. **Monitor**:
       - **Live Logs (Sidebar)**: Watch the raw terminal output (downloads, errors).
       - **Metrics (Here)**: Watch the reward go UP and loss go DOWN.
       - **Samples (Tab 2)**: Read the actual Q&A the model is generating.
    3. **Stop**: Click "🛑 Stop" in the sidebar when done.
    """)

# Auto Refresh logic at bottom
if st.checkbox("Auto-Refresh (2s)", value=True):
    time.sleep(2)
    st.rerun()

