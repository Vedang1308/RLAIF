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

# Local Process Helpers
PID_FILE = "local_run.pid"
import sys

def start_local(mode_arg="demo"):
    # Authenticate WandB for local run
    env = os.environ.copy()
    env["WANDB_API_KEY"] = "ef2da50d021e41130a9c9d762f7e56c79dbed703"
    env["WANDB_PROJECT"] = "rlaif-research"
    
    with open("local_log.txt", "w") as out:
        # Pass the selected mode
        proc = subprocess.Popen(
            [sys.executable, "train.py", "--mode", mode_arg], 
            stdout=out, 
            stderr=out,
            env=env
        )
        with open(PID_FILE, "w") as f:
            f.write(str(proc.pid))
    return f"Started Local Process (PID {proc.pid}) in {mode_arg.upper()} mode"

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
    
    # Mode Selector
    if not HAS_SLURM:
        train_mode = st.radio("Training Mode", ["Demo (Fast)", "Full (Research)"], index=0)
        mode_arg = "demo" if "Demo" in train_mode else "research"
    else:
        mode_arg = "research" # Cluster always research
    
    if is_running:
        st.button("⚠️ Running...", disabled=True, help="Job is already active. Stop it first.")
    else:
        if st.button(btn_label, help=f"Start training on {MODE_LABEL}"):
            # 1. Logic: Only Clear Logs if Starting FRESH
            # If we have checkpoints, we want to append/persist history.
            has_checkpoints = False
            if os.path.exists("trainer_output"):
                import glob
                if glob.glob("trainer_output/checkpoint-*"):
                    has_checkpoints = True
            
            if not has_checkpoints and os.path.exists(LOG_FILE):
                try:
                    os.remove(LOG_FILE)
                except:
                    pass
            
            # 2. Start Job
            with st.spinner("Starting..."):
                if HAS_SLURM:
                    out = run_command(f"sbatch {JOB_SCRIPT}")
                else:
                    out = start_local(mode_arg)
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
        st.sidebar.text(labels)
    else:
        st.sidebar.caption("⚪ No Active Jobs")
else:

    # Global Progress Parsing
    # Read the log file to find our custom "GLOBAL PROGRESS" line
    log_file_path = "local_log.txt" if not HAS_SLURM else None # Slurm logic handled by glob/max below, but simple check here
    
    global_step = 0
    total_steps = 1
    pct = 0.0
    
    # Try to find the log file
    real_log_file = None
    if HAS_SLURM:
         import glob
         files = glob.glob("slurm-*.out")
         if files:
             real_log_file = max(files, key=os.path.getctime)
    elif os.path.exists("local_log.txt"):
         real_log_file = "local_log.txt"
         
    if real_log_file and os.path.exists(real_log_file):
        try:
            # tailored for speed: read last 20 lines
            with open(real_log_file, "rb") as f:
                try:
                    f.seek(-2000, os.SEEK_END)
                except:
                    pass # File too small
                lines = f.readlines()
                for line in reversed(lines):
                    line = line.decode("utf-8", errors="ignore")
                    if "GLOBAL PROGRESS:" in line:
                        # Parse: 🌍 GLOBAL PROGRESS: Step 8/42 (19.0%)
                        import re
                        match = re.search(r"Step (\d+)/(\d+)", line)
                        if match:
                            global_step = int(match.group(1))
                            total_steps = int(match.group(2))
                            pct = global_step / total_steps
                        break
        except:
            pass
            
    if global_step > 0:
        st.sidebar.markdown("### 🌍 Global Progress")
        st.sidebar.progress(pct, text=f"Step {global_step} of {total_steps}")
    
    status_msg = check_local_status()
    st.sidebar.info(status_msg)

    # Checkpoint Persistence Check
    if os.path.exists("trainer_output"):
        import glob
        ckpts = glob.glob("trainer_output/checkpoint-*")
        if ckpts:
            latest = max(ckpts, key=os.path.getctime)
            ckpt_name = os.path.basename(latest)
            st.sidebar.success(f"💾 Resume Ready: {ckpt_name}")
        else:
            st.sidebar.caption("💾 No checkpoints yet")
    else:
        st.sidebar.caption("💾 No checkpoints found")

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
st.sidebar.markdown("---")
with st.sidebar.expander("🖥️ Live Logs", expanded=True):
    # Fixed height container for scrolling
    with st.container(height=300):
        st.code(get_log_content(), language="text")



# --- Main: Conference Dashboard ---
# Hero Section
st.markdown("""
<style>
    /* Global clean up */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem;
    }
    
    /* 1. Card Styling for Metrics */
    div[data-testid="stMetric"] {
        background-color: #1e2129;
        border: 1px solid #303540;
        padding: 15px;
        border-radius: 8px;
        color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        height: 100%; /* Force equal height */
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.2);
    }
    
    /* 2. Hero Header */
    .hero-container {
        padding: 20px;
        background-color: #0e1117;
        border-radius: 10px;
        margin-bottom: 20px;
        border-bottom: 2px solid #262730;
    }
    
    /* 3. Status Pulse Animation */
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
        100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
    }
    .status-pulse {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #4CAF50;
        border-radius: 50%;
        animation: pulse-green 2s infinite;
        margin-right: 8px;
    }
    
    /* 4. Chat Slide Up Animation */
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stChatMessage {
        animation: slideUp 0.3s ease-out;
    }
</style>
""", unsafe_allow_html=True)

with st.container():
    # Use the CSS classes we defined
    status_indicator = ""
    is_online = "Running" in check_local_status() or (HAS_SLURM and "Active" in run_command("squeue --me"))
    
    if is_online:
        status_html = """<div style="display: flex; align-items: center; justify-content: flex-end; height: 100%;">
                            <span class="status-pulse"></span>
                            <span style="color: #4CAF50; font-weight: bold; letter-spacing: 1px;">SYSTEM ONLINE</span>
                         </div>"""
    else:
        status_html = """<div style="display: flex; align-items: center; justify-content: flex-end; height: 100%;">
                            <span style="color: #F44336; font-weight: bold; letter-spacing: 1px;">SYSTEM OFFLINE</span>
                         </div>"""

    st.markdown(f"""
        <div class="hero-container">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 style="margin: 0; padding: 0; font-size: 2.2rem;">🚀 RLAIF Control Center</h1>
                    <p style="margin: 5px 0 0 0; opacity: 0.8;">Reinforcement Learning from AI Feedback | Real-Time Monitor</p>
                </div>
                {status_html}
            </div>
        </div>
    """, unsafe_allow_html=True)

def load_data():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(), pd.DataFrame()
    
    metrics = []
    samples = []
    
    try:
        # Fast read of last 2000 lines to avoid lag
        with open(LOG_FILE, "rb") as f:
            try:
                f.seek(-50000, os.SEEK_END)
            except:
                pass
            lines = f.readlines()
            
        for line in lines:
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

# MAIN VERTICAL LAYOUT (Story Mode)
# 1. System Health
st.subheader("📊 System Health")

if not df_metrics.empty:
    latest = df_metrics.iloc[-1]
    cols = df_metrics.columns
    reward_col = next((c for c in cols if "reward" in c or "score" in c), None)
    loss_col = next((c for c in cols if "loss" in c), None)
    kl_col = next((c for c in cols if "kl" in c), None)

    # Technical Metric Cards (Grid Alignment)
    m1, m2, m3 = st.columns(3)
    
    cur_reward = float(latest[reward_col]) if reward_col else 0.0
    cur_loss = float(latest[loss_col]) if loss_col else 0.0
    cur_kl = float(latest.get(kl_col, 0.0)) if kl_col else 0.0
    
    m1.metric("Avg Reward", f"{cur_reward:.3f}", delta=f"{cur_reward - df_metrics.iloc[-2][reward_col]:.3f}" if len(df_metrics)>1 else None)
    m2.metric("Loss", f"{cur_loss:.4f}", delta_color="inverse")
    m3.metric("KL Div", f"{cur_kl:.4f}", delta_color="inverse")

    st.divider()

    # 2. Key Charts (Training Trends)
    st.subheader("📈 Training Trends")
    c1, c2 = st.columns(2) # Side by side charts looks better on full width
    
    # Reward Chart
    if reward_col:
        with c1:
            st.caption("Reward History")
            chart_r = alt.Chart(df_metrics.tail(200)).mark_line(color='#4CAF50').encode(
                x='step', y=alt.Y(reward_col, title='Reward'), tooltip=['step', reward_col]
            ).interactive().properties(height=300) # Available height check
            st.altair_chart(chart_r, use_container_width=True)

    # KL Chart
    if kl_col:
        with c2:
            st.caption("KL Divergence")
            chart_k = alt.Chart(df_metrics.tail(200)).mark_line(color='#FF9800').encode(
                x='step', y=alt.Y(kl_col, title='KL Div'), tooltip=['step', kl_col]
            ).interactive().properties(height=300)
            st.altair_chart(chart_k, use_container_width=True)

else:
    st.info("Waiting for training metrics...")
    for _ in range(3):
        st.markdown("⬜⬜⬜⬜⬜")

st.divider()

# 3. Live Thought Process (Full Width Chat)
st.subheader("💬 Live Thought Process")
st.caption("Real-time samples from the model as it learns.")

container = st.container(height=500)
with container:
    if not df_samples.empty:
        # Show last 5 interactions
        recent = df_samples.tail(5)[::-1]
        for i, row in recent.iterrows():
            r_val = row.get('reward', 0.0)
            icon = "🧠" if r_val > 0 else "💤"
            
            # Render as Chat
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"**Scenario:** {row.get('question')}")
            
            with st.chat_message("assistant", avatar=icon):
                st.markdown(row.get('response'))
                
                # Evaluation Badge
                if r_val > 0.5:
                    st.success(f"✅ High Quality (Score: {r_val:.2f})")
                else:
                    st.warning(f"⚠️ Needs Improvement (Score: {r_val:.2f})")
            
            st.divider()
    else:
        st.write("Initializing Conversation Interface...")


# Auto Refresh logic at bottom
if st.checkbox("Auto-Refresh (1s)", value=True):
    time.sleep(1)
    st.rerun()

