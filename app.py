import streamlit as st
import pandas as pd
import json
import os
import subprocess
import time
import altair as alt
import shutil
import signal
import sys

# --- Config & Setup ---
st.set_page_config(page_title="RLAIF Control Center", layout="wide", page_icon="🚀")

LOG_FILE = "logs/metrics.jsonl"
JOB_SCRIPT = "job_launcher.sh"
PID_FILE = "local_run.pid"

# Default Model IDs (Fallback if local missing)
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_ID = "vedang1308/RLAIF-Qwen" # We can load adapter dynamically if merged missing

# Detect Environment
HAS_SLURM = shutil.which("sbatch") is not None
MODE_LABEL = "Cluster" if HAS_SLURM else "Local"

# --- Styling ---
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background-color: #1e2129; border: 1px solid #303540;
        padding: 10px; border-radius: 8px; color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .hero-container {
        padding: 15px; background-color: #0e1117; border-radius: 10px;
        margin-bottom: 15px; border-bottom: 2px solid #262730;
    }
    .chat-message {
        padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-user { background-color: #2b313e; }
    .chat-bot { background-color: #1e2129; border: 1px solid #4CAF50; }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

def check_local_status():
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f: pid = int(f.read().strip())
            os.kill(pid, 0)
            return f"🟢 Running (PID {pid})"
        except: return "🔴 Stopped"
    return "⚪ Idle"

def start_local(mode_arg="demo"):
    env = os.environ.copy()
    # env["WANDB_API_KEY"] = "..." # Assumed set in shell or unused locally
    with open("local_log.txt", "w") as out:
        proc = subprocess.Popen([sys.executable, "train.py", "--mode", mode_arg], stdout=out, stderr=out, env=env)
        with open(PID_FILE, "w") as f: f.write(str(proc.pid))
    return f"Started PID {proc.pid}"

def stop_local():
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f: pid = int(f.read().strip())
            os.kill(pid, signal.SIGTERM)
            os.remove(PID_FILE)
        except: pass

# --- Model Loading (Cached) ---
@st.cache_resource
def load_chat_model():
    """Smart Loader: Tries Local Merged -> Local Adapter -> Remote Adapter"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")

    # 1. Try Local Merged Model
    if os.path.exists("merged_model") and os.path.exists("merged_model/config.json"):
        print("✅ Found local merged_model. Loading...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("merged_model")
            model = AutoModelForCausalLM.from_pretrained("merged_model", torch_dtype=torch.float16, device_map=device)
            return tokenizer, model, "Local Merged 📂"
        except Exception as e:
            print(f"Failed to load local merged: {e}")

    # 2. Try Remote Adapter (Fallback)
    print(f"☁️ Configuring Remote Adapter: {ADAPTER_ID} + {BASE_MODEL_ID}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float16, device_map=device)
        model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
        return tokenizer, model, f"Remote Adapter ({ADAPTER_ID}) ☁️"
    except Exception as e:
        return None, None, f"Error: {e}"

# --- Sidebar ---
st.sidebar.title("🎮 Control")
app_mode = st.sidebar.radio("Mode", ["📊 Monitor", "💬 Chat / Demo"], index=0)
st.sidebar.divider()

if app_mode == "📊 Monitor":
    # --- MONITOR LOGIC ---
    
    # Refresh Logic
    if st.sidebar.checkbox("Auto-Refresh (2s)", value=True):
        time.sleep(2)
        st.rerun()

    # Controls
    if st.sidebar.button("▶️ Start Training"):
        if HAS_SLURM: run_command(f"sbatch {JOB_SCRIPT}")
        else: start_local("demo")
        st.success("Sent!")
        time.sleep(1)
        st.rerun()

    if st.sidebar.button("🛑 Stop Job"):
        if HAS_SLURM: run_command("scancel -u $USER")
        else: stop_local()
        st.warning("Stopped.")
        time.sleep(1)
        st.rerun()

    status = check_local_status() if not HAS_SLURM else ("🟢 Active" if "JOBID" in run_command("squeue --me") else "⚪ Idle")
    st.sidebar.info(f"Status: {status}")

    # Main Dashboard
    st.markdown('<div class="hero-container"><h1>🚀 Training Monitor</h1></div>', unsafe_allow_html=True)
    
    # Load Logs
    metrics = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            for line in f:
                try: metrics.append(json.loads(line))
                except: pass
    
    if metrics:
        df = pd.DataFrame([m for m in metrics if m.get("type")=="metrics"])
        samples = [m for m in metrics if m.get("type")=="sample"]
        
        if not df.empty:
            # Metrics
            c1, c2, c3 = st.columns(3)
            latest = df.iloc[-1]
            c1.metric("Reward", f"{latest.get('reward',0):.3f}")
            c2.metric("KL Div", f"{latest.get('kl',0):.4f}")
            c3.metric("Loss", f"{latest.get('ppo_loss',0):.4f}")
            
            # Charts
            st.markdown("### 📈 Trends")
            chart = alt.Chart(df.tail(100)).mark_line().encode(x='step', y='reward', tooltip=['step', 'reward']).properties(height=200)
            st.altair_chart(chart, use_container_width=True)
        
        if samples:
            st.markdown("### 🧠 Latest Thoughts")
            last_sample = samples[-1]
            st.code(f"Q: {last_sample.get('question')}\n\nA: {last_sample.get('response')}", language="text")

    else:
        st.info("Waiting for logs... (Start a job to see data)")

else:
    # --- CHAT LOGIC ---
    st.markdown('<div class="hero-container"><h1>💬 RLAIF Chat Demo</h1></div>', unsafe_allow_html=True)
    
    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "I am your RLAIF-Trained Assistant. Ask me a math question!"})

    # Render History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Enter a math problem..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                tokenizer, model, source = load_chat_model()
                
                if model:
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=200, 
                        do_sample=True, 
                        temperature=0.7
                    )
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # Simple cleanup to remove prompt if echo
                    if response.startswith(prompt): 
                        response = response[len(prompt):]
                    
                    st.markdown(response)
                    st.caption(f"Generated by: {source}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error(f"Failed to load model. Source: {source}")
