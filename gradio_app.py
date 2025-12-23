import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import os

# Config
MERGED_REPO = "vedang1308/RLAIF-Qwen-Merged"
LOCAL_MERGED = "merged_model"
MAX_NEW_TOKENS = 512

print(f"🚀 Initializing Gradio Chat...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device: {device.upper()}")

# Smart Loading (Local -> Remote)
model_path = MERGED_REPO
if os.path.exists(LOCAL_MERGED) and os.path.exists(os.path.join(LOCAL_MERGED, "config.json")):
    print(f"📂 Found local model: {LOCAL_MERGED}")
    model_path = LOCAL_MERGED
else:
    print(f"☁️ Using remote model: {MERGED_REPO}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map=device)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise e

def chat(message, history):
    # Prepare Input
    prompt = ""
    
    # Robust History Parsing (Handle both Tuple and Message formats)
    for item in history:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            # Legacy Gradio: [user, bot]
            user_msg, bot_msg = item
            prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{bot_msg}<|im_end|>\n"
        elif isinstance(item, dict):
            # OpenAI/New Gradio: {"role": "user", "content": ...}
            # We need to reconstruct pairs or just append linearly
            role = item.get("role")
            content = item.get("content")
            if role and content:
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                
    prompt += f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs, 
        streamer=streamer, 
        max_new_tokens=MAX_NEW_TOKENS, 
        do_sample=True, 
        temperature=0.7 
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

# Create UI
css = """
footer {visibility: hidden}
.bubble-wrap { display: flex; flex-direction: column; }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 🤖 RLAIF Qwen Chat\nRunning: `{model_path}`")
    
    chat_interface = gr.ChatInterface(
        fn=chat, 
        chatbot=gr.Chatbot(height=600),
        textbox=gr.Textbox(placeholder="Ask me a math question...", container=False, scale=7),
        examples=["Solve this: 2x + 5 = 15", "Jane has 3 apples and eats one. How many left?", "Explain quantum physics like I am 5."]
    )

if __name__ == "__main__":
    # share=True creates a public link (Good for clusters!)
    # server_name="0.0.0.0" allows access from network
    print("\n🌐 Launching Web Interface...")
    print("   If 'share=True' works, you will get a public URL.")
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
