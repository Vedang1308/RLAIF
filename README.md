# RLAIF / RLVR Training System

A robust system for Reinforcement Learning (PPO) with AI Feedback/Verification, featuring a Streamlit dashboard.

## 🚀 Quick Start

### 1. Supercomputer (Cluster)
Run this on the login node. It sets up the dashboard and prints the secure tunnel command.

```bash
# 1. Update Code
git pull

# 2. Launch Dashboard (Headless)
bash start_dashboard.sh
```
*Follow the on-screen instructions to tunnel port 8502.*

### 2. Local Mac (M1/M2/M3)
Run this on your laptop directly.

```bash
# 1. Install Dependencies
pip install -r requirements.txt
pip install streamlit

# 2. Run Dashboard
streamlit run app.py
```
*Click "Start Local" in the dashboard sidebar.*

## 📂 Key Files
- `app.py`: Streamlit Control Center (UI).
- `train.py`: Main training script (PPO).
- `rewards.py`: Reward logic (Math verification).
- `job_launcher.sh`: SLURM submission script.
