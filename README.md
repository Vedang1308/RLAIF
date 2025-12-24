# RLAIF-Qwen-Nano: Democratizing Alignment via Heuristic RLAIF and Verification

**Artifacts:** [Hugging Face Model](https://huggingface.co/vedang1308/RLAIF-Qwen-Merged) | [Codebase](https://github.com/vedang1308/RLAIF)

---

## 1. Abstract
The current paradigm of Reinforcement Learning from AI Feedback (RLAIF) predominantly relies on massive "Judge Models" (e.g., GPT-4, 70B+ parameters) to distill preferences into smaller models (7B+). This creates a high barrier to entry and high latency. This research proposes and validates a **Heuristic RLAIF + RLVR (Reinforcement Learning from Verifiable Rewards)** pipeline applied to a "Nano-scale" model (`Qwen2.5-0.5B-Instruct`). By replacing the learned Reward Model with a lightweight, programmatic reward function that combines verifiable correctness (Math) with structural heuristics (Chain-of-Thought), we successfully aligned a 0.5B parameter model. We demonstrate that small models can adopt complex formatting and reasoning structures via PPO without significant catastrophic forgetting, achieving a **43.75%** accuracy on GSM8K while adhering to strict formatting constraints, all within an offline, compute-constrained cluster environment.

---

## 2. Quick Start

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
/usr/bin/python3 -m streamlit run app.py
```
*Click "Start Local" in the dashboard sidebar.*

---

## 3. Methodology & System Architecture

### 3.1 Base Model and Efficient Fine-Tuning
We selected `Qwen/Qwen2.5-0.5B-Instruct` as the policy model due to its state-of-the-art performance-to-size ratio. To preserve the pre-trained knowledge and ensure stability, we employed **LoRA (Low-Rank Adaptation)** rather than full fine-tuning.
* **Rank:** 16
* **Alpha:** 32
* **Dropout:** 0.05
* **Target Modules:** `k_proj`, `v_proj`, `q_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

### 3.2 The Hybrid Reward Function (The Core Innovation)
Unlike standard PPO implementations that query a neural network for a scalar reward, we implemented a custom `ProgrammaticRewardModel` class. The reward $R_{total}$ is a weighted sum of two distinct signal sources defined in `rewards.py`:

$$R_{total} = w_1 \cdot R_{verification} + w_2 \cdot R_{heuristic}$$

#### A. Verification Reward ($R_{verification}$) - The "RLVR" Component
This component ensures truthfulness. The system parses the generated text for the final answer and compares it against the GSM8K ground truth.
* **Logic:** String normalization (removing `$`, `,`) followed by float comparison with a tolerance of $1e^{-6}$.
* **Value:** +1.0 for correct, 0.0 for incorrect.

#### B. Heuristic AI Feedback ($R_{heuristic}$) - The "RLAIF" Component
This component guides the *process* of reasoning. Instead of a black-box LLM judge, we encoded high-quality reasoning indicators directly into Python logic:
1.  **Structural Signalling (+0.2):** Presence of "Step 1", "First", or "Therefore".
2.  **Format Compliance (+0.5):** usage of LaTeX `\boxed{}` tagging.
3.  **Depth Penalty/Reward (+0.2):** Positive reinforcement for chain-of-thought length > 100 characters to discourage lazy guessing.

---

## 4. Experimental Results

### Quantitative Analysis
Evaluation was performed on the GSM8K test set using `evaluate.py`.
* **Baseline Accuracy:** 44.05%
* **RLAIF-Trained Accuracy:** 43.75%
* **Format Adherence:** 98% (up from <50% in base model).

While the raw accuracy remained statistically flat, the **utility** of the model increased significantly. The aligned model now reliably outputs answers in a parseable format, making it suitable for automated systems—something the base model failed to do consistently.

### Qualitative Analysis
The dashboard logs reveal that the trained model attempts to derive answers using "First," "Next," "Therefore" structures. In failed cases, the model often hallucinated incorrect intermediate numbers but maintained the correct structure, validating that the PPO successfully optimized for $R_{heuristic}$.

---

## 5. Differentiation from State-of-the-Art

This project diverges from contemporary RLAIF research (e.g., Anthropic’s "Constitutional AI" or Google’s "Starling") in three critical ways:

| Feature | Standard RLAIF Research | **Our Approach (RLAIF-Qwen-Nano)** |
| :--- | :--- | :--- |
| **Model Scale** | 7B - 70B Parameters | **0.5B Parameters** |
| **Judge Mechanism** | LLM-as-a-Judge (e.g., GPT-4) | **Programmatic Heuristics + Verification** |
| **Latency** | High (Inference-heavy scoring) | **Near-Zero (Regex/Python scoring)** |
| **Infrastructure** | H100 Clusters | **Consumer GPU / Free-Tier Cluster** |

---

## 📂 Key Files
- `app.py`: Streamlit Control Center (UI).
- `train.py`: Main training script (PPO).
- `rewards.py`: Reward logic (Math verification).
- `job_launcher.sh`: SLURM submission script.
