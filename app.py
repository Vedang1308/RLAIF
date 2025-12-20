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
st.title("🚀 RLAIF Training Control Center") # REMOVED (Redundant with Hero)

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


# --- Sidebar Logic Refactored ---

# 1. Global Progress Parsing (Prominent at Top)
log_file_path = "local_log.txt" if not HAS_SLURM else None
global_step = 0
total_steps = 1
pct = 0.0
etr_str = "--:--"

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
        with open(real_log_file, "rb") as f:
            try: f.seek(-2000, os.SEEK_END)
            except: pass
            lines = f.readlines()
            for line in reversed(lines):
                line = line.decode("utf-8", errors="ignore")
                if "GLOBAL PROGRESS:" in line:
                    import re
                    match = re.search(r"Step (\d+)/(\d+)", line)
                    if match:
                        global_step = int(match.group(1))
                        total_steps = int(match.group(2))
                        pct = global_step / total_steps
                        
                        # ETR Calculation attempt
                        # If we have df_metrics loaded later, we can be more precise.
                        # But here, let's use a simple heuristic if widely available
                        # Actually, we can't see df_metrics yet.
                        # Let's placeholder ETR calculation here and do it in Main.
                    break
    except:
        pass

if global_step > 0:
    st.sidebar.markdown("### 🌍 Global Progress")
    st.sidebar.progress(pct, text=f"Step {global_step} of {total_steps} ({pct*100:.1f}%)")

# 2. Controls & Status (Collapsed to clean up)
with st.sidebar.expander("⚙️ Controls & Status", expanded=False):
    st.markdown("#### Actions")
    col_s1, col_s2 = st.columns(2)
    
    # Determine state
    is_running = False
    if HAS_SLURM:
        check = run_command("squeue --me --noheader")
        if check and len(check.strip()) > 0: is_running = True
    else:
        if os.path.exists(PID_FILE):
            try:
                with open(PID_FILE, "r") as f:
                    pid = int(f.read().strip())
                os.kill(pid, 0)
                is_running = True
            except: is_running = False

    with col_s1:
        btn_label = "▶️ Start"
        # Mode Selector
        if not HAS_SLURM:
            train_mode = st.radio("Mode", ["Demo", "Full"], index=0, horizontal=True)
            mode_arg = "demo" if "Demo" in train_mode else "research"
        else:
            mode_arg = "research"
            
        if is_running:
            st.button("Running...", disabled=True)
        else:
            if st.button(btn_label, help=f"Start on {MODE_LABEL}"):
                # Clear Logic
                has_checkpoints = False
                if os.path.exists("trainer_output"):
                    import glob
                    if glob.glob("trainer_output/checkpoint-*"): has_checkpoints = True
                if not has_checkpoints and os.path.exists(LOG_FILE):
                     try: os.remove(LOG_FILE)
                     except: pass
                
                with st.spinner("Starting..."):
                    if HAS_SLURM: out = run_command(f"sbatch {JOB_SCRIPT}")
                    else: out = start_local(mode_arg)
                st.success("Started!")
                time.sleep(1)
                st.rerun()

    with col_s2:
        if st.button("🛑 Stop"):
            with st.spinner("Stopping..."):
                if HAS_SLURM:
                    user = os.environ.get("USER", "vavaghad")
                    out = run_command(f"scancel -u {user}")
                else:
                    out = stop_local()
            st.warning("Stopped")
            time.sleep(1)
            st.rerun()

    # st.markdown("---")
    st.markdown("#### Status")
    status_msg = check_local_status()
    st.caption(f"Local: {status_msg}")
    
    if HAS_SLURM:
        labels = run_command("squeue --me --format='%.8i %.9P %.8j %.2t %.10M'")
        if labels and "JOBID" in labels:
            st.info("🟢 Cluster Job Active")
            st.text(labels)

    # Checkpoints
    if os.path.exists("trainer_output"):
        import glob
        ckpts = glob.glob("trainer_output/checkpoint-*")
        if ckpts:
            latest_c = max(ckpts, key=os.path.getctime)
            st.caption(f"💾 Latest: {os.path.basename(latest_c)}")
        else:
            st.caption("💾 No checkpoints")

if st.sidebar.button("🔄 Refresh"):
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
        # tailored for speed: read last 50 lines
        # and REVERSE them so the latest is at the top (Pseudo Auto-Scroll)
        with open(log_file, "r") as f:
            # simple tail approach
            lines = f.readlines()
            # unique reverse sort
            return "".join(lines[::-1][:50]) 
    except Exception as e:
        return f"Error: {e}"

# Dividers removed for compactness
with st.sidebar.expander("🖥️ Live Logs", expanded=True):
    # Fixed height container for scrolling
    with st.container(height=300):
        st.code(get_log_content(), language="text")



# --- Main: Conference Dashboard ---

# 0. Load Data EARLY for ETR and Hero
def load_data():
    if not os.path.exists(LOG_FILE): return pd.DataFrame(), pd.DataFrame()
    metrics, samples = [], []
    try:
        with open(LOG_FILE, "rb") as f:
            try: f.seek(-50000, os.SEEK_END)
            except: pass
            lines = f.readlines()
        for line in lines:
            try:
                data = json.loads(line)
                if data.get("type") == "metrics": metrics.append(data)
                elif data.get("type") == "sample": samples.append(data)
            except: continue
    except: pass
    return pd.DataFrame(metrics), pd.DataFrame(samples)

df_metrics, df_samples = load_data()

# Calculate ETR
etr_html = ""
if not df_metrics.empty and 'step' in df_metrics.columns and 'timestamp' in df_metrics.columns:
    # Use global_step from sidebar if available, else derive
    if global_step > 0 and total_steps > global_step:
        # Rate calc
        recent = df_metrics.tail(50)
        if len(recent) > 1:
            t_span = recent.iloc[-1]['timestamp'] - recent.iloc[0]['timestamp']
            s_span = recent.iloc[-1]['step'] - recent.iloc[0]['step']
            if s_span > 0:
                sec_per_step = t_span / s_span
                rem_sec = (total_steps - global_step) * sec_per_step
                etr_html = f'<span style="margin-left: 15px; background-color: #333; padding: 4px 8px; border-radius: 4px; border: 1px solid #555; font-size: 0.9em;">⏱️ ETR: {int(rem_sec//60)}m {int(rem_sec%60)}s</span>'

# Hero Section
# Hero Section
st.markdown("""
<style>
    /* Global clean up - Aggressive Compactness */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem; /* No bottom scroll */
    }
    
    /* 1. Card Styling for Metrics */
    div[data-testid="stMetric"] {
        background-color: #1e2129;
        border: 1px solid #303540;
        padding: 10px; /* Reduced padding */
        border-radius: 8px;
        color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        height: 100%; 
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.2);
    }
    
    /* 2. Hero Header - Compact */
    .hero-container {
        padding: 15px; /* Reduced from 20px */
        background-color: #0e1117;
        border-radius: 10px;
        margin-bottom: 10px;
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
        status_html = f'''<div style="display: flex; align-items: center; justify-content: flex-end; height: 100%;">
{etr_html}
<span class="status-pulse" style="margin-left: 15px;"></span>
<span style="color: #4CAF50; font-weight: bold; letter-spacing: 1px;">SYSTEM ONLINE</span>
</div>'''
    else:
        status_html = '''<div style="display: flex; align-items: center; justify-content: flex-end; height: 100%;">
<span style="color: #F44336; font-weight: bold; letter-spacing: 1px;">SYSTEM OFFLINE</span>
</div>'''

    st.markdown(f'''
<div class="hero-container">
<div style="display: flex; justify-content: space-between; align-items: center;">
<div>
<h1 style="margin: 0; padding: 0; font-size: 1.8rem;">🚀 RLAIF Control Center</h1>
<p style="margin: 3px 0 0 0; opacity: 0.8; font-size: 0.9rem;">RL from AI Feedback | Real-Time Monitor</p>
</div>
{status_html}
</div>
</div>
''', unsafe_allow_html=True)

# Data loaded at top for ETR

# MAIN VERTICAL LAYOUT (Story Mode - Compact)
# 1. System Health (No Header, merged under Hero)
# st.markdown("#### 📊 System Health")

if not df_metrics.empty:
    latest = df_metrics.iloc[-1]
    cols = df_metrics.columns
    reward_col = next((c for c in cols if "reward" in c or "score" in c), None)
    loss_col = next((c for c in cols if "loss" in c), None)
    kl_col = next((c for c in cols if "kl" in c), None)

    # Technical Metric Cards (Grid Alignment)
    with st.container():
        m1, m2, m3 = st.columns(3)
        cur_reward = float(latest[reward_col]) if reward_col else 0.0
        cur_loss = float(latest[loss_col]) if loss_col else 0.0
        cur_kl = float(latest.get(kl_col, 0.0)) if kl_col else 0.0
        
        m1.metric("Avg Reward", f"{cur_reward:.3f}", delta=f"{cur_reward - df_metrics.iloc[-2][reward_col]:.3f}" if len(df_metrics)>1 else None)
        m2.metric("Loss", f"{cur_loss:.4f}", delta_color="inverse")
        m3.metric("KL Div", f"{cur_kl:.4f}", delta_color="inverse")

    # Duplicate removed


    # st.divider() # Removed for compactness

    # 2. Key Charts (Training Trends)
    with st.container():
        c1, c2 = st.columns(2) 
        
        # Reward Chart
        if reward_col:
            with c1:
                chart_r = alt.Chart(df_metrics.tail(200)).mark_line(color='#4CAF50').encode(
                    x=alt.X('step', axis=None), 
                    y=alt.Y(reward_col, title=''), tooltip=['step', reward_col]
                ).interactive().properties(height=180)
                st.altair_chart(chart_r, use_container_width=True)

        # KL Chart
        if kl_col:
            with c2:
                chart_k = alt.Chart(df_metrics.tail(200)).mark_line(color='#FF9800').encode(
                    x=alt.X('step', axis=None),
                    y=alt.Y(kl_col, title=''), tooltip=['step', kl_col]
                ).interactive().properties(height=180)
                st.altair_chart(chart_k, use_container_width=True)

else:
    st.info("Waiting for training metrics...")
    # Removed blank placeholders

# st.divider() # Removed for compactness

# 3. Live Thought Process (Full Width Chat)
# 3. Live Thought Process (Full Width Chat)
# st.markdown("#### 💬 Live Thought Process")
# st.caption("Real-time samples from the model as it learns.")

container = st.container(height=280) # Ultra Compact Height
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

