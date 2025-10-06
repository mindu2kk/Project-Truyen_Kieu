# sft/train_lora.py
# QLoRA cho Llama-3-8B-Instruct (hoặc Vistral-7B). Dùng PEFT + transformers.
import os, json, random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# ====== Config ======
BASE_MODEL   = os.getenv("BASE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
SFT_PATH     = os.getenv("SFT_JSONL", "data/sft_json/instruct.jsonl")
OUTPUT_DIR   = os.getenv("LORA_OUT", "artifacts/lora-kieu")
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", "1"))
GRAD_ACCUM   = int(os.getenv("GRAD_ACCUM", "16"))
LR           = float(os.getenv("LR", "2e-4"))
EPOCHS       = int(os.getenv("EPOCHS", "2"))
CUTOFF_LEN   = int(os.getenv("CUTOFF_LEN", "2048"))
LORA_R       = int(os.getenv("LORA_R", "16"))
LORA_ALPHA   = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))
SEED         = 42

assert Path(SFT_PATH).exists(), "Missing SFT jsonl!"

# ====== Data ======
def format_example(ex: Dict) -> str:
    system = ex.get("system", "Bạn là học giả Truyện Kiều.")
    instr  = ex["instruction"].strip()
    ctx    = ex.get("context","").strip()
    out    = ex["output"].strip()
    # Nhập nhịp giống app: có NGỮ CẢNH nếu có
    prompt = f"[SYSTEM]\n{system}\n\n"
    if ctx:
        prompt += f"[NGỮ CẢNH]\n{ctx}\n\n"
    prompt += f"[NGƯỜI DÙNG]\n{instr}\n\n[HƯỚNG DẪN]\nViết mạch lạc, có mở–thân–kết, 180–260 từ, trích thơ nếu có."
    return prompt, out

class JsonlDataset(Dataset):
    def __init__(self, path):
        self.items = [json.loads(l) for l in open(path, "r", encoding="utf-8") if l.strip()]
        random.seed(SEED); random.shuffle(self.items)
    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]

# ====== Load model (4-bit) ======
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    load_in_4bit=True,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)
peft_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)
model = get_peft_model(model, peft_cfg)

# ====== Collator ======
def collate(batch):
    prompts, outs = zip(*(format_example(ex) for ex in batch))
    texts = [p + "\n\n[TRẢ LỜI]\n" + o for p, o in zip(prompts, outs)]
    toks = tokenizer(list(texts), truncation=True, max_length=CUTOFF_LEN, padding=True, return_tensors="pt")
    toks["labels"] = toks["input_ids"].clone()
    return toks

ds = JsonlDataset(SFT_PATH)

# ====== Train ======
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    save_strategy="epoch",
    bf16=torch.cuda.is_available(),
    optim="paged_adamw_32bit",
    report_to="none",
    seed=SEED
)

trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved LoRA adapter to", OUTPUT_DIR)
