#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────
#  Multi-Agent VQA: Text Agent | Chart Agent | Aggregator Agent
#  Parallelized inference with separate GPU allocation
# ────────────────────────────────────────────────────────────────────
from __future__ import annotations
import argparse, json, os, time
from typing import Any, Dict, List, Tuple
import math
from functools import lru_cache
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

import torch
from PIL import Image

# -------------------------------------------------------------------#
#  Constants                                                          #
# -------------------------------------------------------------------#
MODEL_ID   = "Qwen/Qwen2.5-VL-7B-Instruct" 
MIN_PIXELS = 1280 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

# -------------------------------------------------------------------#
#  Model loader with GPU assignment                                   #
# -------------------------------------------------------------------#
# Import at module level to avoid threading issues
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Thread-safe model cache
import threading
_model_cache = {}
_cache_lock = threading.Lock()

def load_qwen(model_path: str, device_id: int):
    """Load Qwen-VL model with proper GPU assignment and thread-safe caching."""

    # Create cache key
    cache_key = (model_path, device_id)

    # Check cache with lock
    with _cache_lock:
        if cache_key in _model_cache:
            return _model_cache[cache_key]

    # Load model (outside lock to allow parallel loading on different GPUs)
    print(f"[GPU {device_id}] Loading {model_path}...")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    device = f"cuda:{device_id}"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=device,
        trust_remote_code=True,
    )

    model.eval()
    torch.cuda.empty_cache()

    # Cache result with lock
    with _cache_lock:
        _model_cache[cache_key] = (processor, model)

    print(f"[GPU {device_id}] Model loaded successfully")
    return processor, model

# -------------------------------------------------------------------#
#  Image preparation                                                  #
# -------------------------------------------------------------------#
def _resize(img: Image.Image) -> Image.Image:
    w, h = img.size
    p = w * h
    if MIN_PIXELS <= p <= MAX_PIXELS:
        return img
    tgt_p = max(min(p, MAX_PIXELS), MIN_PIXELS)
    scale = (tgt_p / p) ** 0.5
    new_wh = (int(w * scale), int(h * scale))
    return img.resize(new_wh, Image.BICUBIC)

# -------------------------------------------------------------------#
#  Agent inference functions                                          #
# -------------------------------------------------------------------#
def text_agent_inference(image_path: str, question: str, model_path: str) -> Dict[str, Any]:
    """Text Analysis Agent - focuses on OCR and text extraction."""
    from qwen_vl_utils import process_vision_info
    
    processor, model = load_qwen(model_path, device_id=0)
    
    prompt = f"{question}\nFocus on extracting and analyzing text information from the image. Provide your answer with text evidence and confidence score (0-1)."
    
    img = _resize(Image.open(image_path).convert("RGB"))
    content = [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt}
    ]
    messages = [{"role": "user", "content": content}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        gen_ids = model.generate(**inputs, max_new_tokens=150)
    
    new_tokens = gen_ids[0][len(inputs.input_ids[0]):]
    answer = processor.decode(new_tokens, skip_special_tokens=True).strip()
    
    return {
        "agent": "text",
        "answer": answer,
        "gpu": 0
    }

def chart_agent_inference(image_path: str, question: str, model_path: str) -> Dict[str, Any]:
    """Chart Analysis Agent - focuses on chart patterns and numerical data."""
    from qwen_vl_utils import process_vision_info
    
    processor, model = load_qwen(model_path, device_id=1)
    
    prompt = f"{question}\nFocus on analyzing chart patterns, numerical values, and visual trends. Provide your answer with numerical evidence and confidence score (0-1)."
    
    img = _resize(Image.open(image_path).convert("RGB"))
    content = [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt}
    ]
    messages = [{"role": "user", "content": content}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        gen_ids = model.generate(**inputs, max_new_tokens=150)
    
    new_tokens = gen_ids[0][len(inputs.input_ids[0]):]
    answer = processor.decode(new_tokens, skip_special_tokens=True).strip()
    
    return {
        "agent": "chart",
        "answer": answer,
        "gpu": 1
    }

def aggregator_inference(image_path: str, question: str, text_output: Dict, chart_output: Dict, model_path: str) -> str:
    """Aggregator Agent - synthesizes outputs from both agents."""
    from qwen_vl_utils import process_vision_info
    
    processor, model = load_qwen(model_path, device_id=2)
    
    prompt = f"""Question: {question}

Text Agent Answer:
{text_output['answer']}

Chart Agent Answer:
{chart_output['answer']}

You are a precise chart reasoning expert. 
Use the chart image and the two agents’ analyses to decide the single correct final answer.

Before answering, classify the question type carefully:

1. **COUNT / QUANTITY**  
   - Starts with "How many", "Number of", "How much".  
   → Output one integer only (no list, no brackets).  
   Example: "How many bars are shown?" → 3

2. **RATIO / COMPARISON / DIFFERENCE**  
   - Contains "ratio", "difference", "compare", "proportion".  
   → Output one numeric value only.  
   Example: "What’s the ratio of green and blue bars?" → 1.38

3. **SINGLE SELECTION (one item among several chart elements)**  
   - Starts with "Which", "What", "Who", "When", "Where", or refers to one bar, one line, one color, one label, one group.  
   - Even if the question mentions plural nouns ("bars", "lines", "mins"), if it asks for *the* specific one with a property (e.g., maximum, lowest, color of boys), the answer must be a **single value** (no list).  
   Examples:  
     - "Maximum for how long people waited when they went to vote?" → Over 30 mins  
     - "Which line represents data about boys?" → Teal  
     - "What color shows data for 2020?" → Blue  
     - "Which country leads?" → France

4. **LIST / MULTIPLE-ITEM enumeration**  
   - Clearly asks for *all items* or *more than one item*:  
     Contains "list", "all", "every", "each of", "both", "categories", "countries", "brands", etc.  
   - Output all correct items as a comma-separated list within square brackets.  
   Example: "List all three brands shown" → [Apple, Samsung, Google]

OUTPUT RULES:
A. For single answers → write only the value itself (no brackets, quotes, or units).  
   Examples: 47 / Germany / Over 30 mins / Teal  
B. For multiple answers → use square brackets with comma + single space.  
   Example: [Food, Beverage, Retail]  
C. Never include reasoning, extra text, or punctuation.

Now provide the final answer only, following these rules:
"""
    
    img = _resize(Image.open(image_path).convert("RGB"))
    content = [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt}
    ]
    messages = [{"role": "user", "content": content}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        gen_ids = model.generate(**inputs, max_new_tokens=100)
    
    new_tokens = gen_ids[0][len(inputs.input_ids[0]):]
    answer = processor.decode(new_tokens, skip_special_tokens=True).strip()
    
    return answer

# -------------------------------------------------------------------#
#  Multi-agent VQA pipeline                                          #
# -------------------------------------------------------------------#

def multi_agent_vqa(image_path: str, question: str, model_path: str, output_dir: str, imgname: str) -> Tuple[str, Dict]:
    """Run parallel text/chart agents, then aggregate."""
    
    # Parallel inference
    with ThreadPoolExecutor(max_workers=2) as executor:
        text_future = executor.submit(text_agent_inference, image_path, question, model_path)
        chart_future = executor.submit(chart_agent_inference, image_path, question, model_path)
        
        text_result = text_future.result()
        chart_result = chart_future.result()
    
    # Save intermediate results
    intermediate = {
        "imgname": imgname,
        "question": question,
        "text_agent": text_result,
        "chart_agent": chart_result
    }
    
    inter_path = os.path.join(output_dir, "intermediate", f"{os.path.splitext(imgname)[0]}.json")
    os.makedirs(os.path.dirname(inter_path), exist_ok=True)
    with open(inter_path, "w") as f:
        json.dump(intermediate, f, indent=2)
    
    # Aggregation
    final_answer = aggregator_inference(image_path, question, text_result, chart_result, model_path)
    
    aggregation = {
        "imgname": imgname,
        "question": question,
        "text_agent": text_result,
        "chart_agent": chart_result,
        "final_answer": final_answer,
        "aggregator_gpu": 2
    }
    
    return final_answer, aggregation

# -------------------------------------------------------------------#
#  Relaxed accuracy                                                  #
# -------------------------------------------------------------------#
def relaxed_correctness(target: str, prediction: str, max_relative_change: float = 0.05) -> bool:
    def _to_float(text: str):
        try:
            return float(text.rstrip('%')) / 100.0 if text.endswith('%') else float(text)
        except ValueError:
            return None
    
    prediction, target = str(prediction).strip(), str(target).strip()
    p_float, t_float = _to_float(prediction), _to_float(target)
    
    if p_float is not None and t_float:
        rel_change = abs(p_float - t_float) / abs(t_float)
        return rel_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()

# -------------------------------------------------------------------#
#  Batch runner                                                      #
# -------------------------------------------------------------------#
def run_split(entries, img_root, split_name, model_id, output_dir):
    """Run multi-agent VQA over one split."""
    results = []
    for ex in tqdm(entries, desc=f"Infer {split_name}", ncols=80):
        img_path = os.path.join(img_root, ex["imgname"])
        if not os.path.exists(img_path):
            print(f"[{split_name}] SKIP (missing image): {ex['imgname']}")
            continue
        
        final_pred, aggregation_data = multi_agent_vqa(
            img_path, ex["query"], model_id, output_dir, ex["imgname"]
        )
        
        print(f"[{split_name}] Q: {ex['query']}  →  {final_pred}")
        
        rec = {
            "imgname": ex["imgname"],
            "query": ex["query"],
            "prediction": final_pred,
            "answer": ex["label"],
            "split": split_name,
            "text_agent_output": aggregation_data["text_agent"]["answer"],
            "chart_agent_output": aggregation_data["chart_agent"]["answer"]
        }
        results.append(rec)
    
    return results

def compute_accuracy(recs: List[Dict[str, Any]]) -> float:
    if not recs:
        return 0.0
    hits = sum(relaxed_correctness(r["answer"], r["prediction"]) for r in recs)
    return hits / len(recs)

# -------------------------------------------------------------------#
#  CLI                                                               #
# -------------------------------------------------------------------#
def main():
    ap = argparse.ArgumentParser(description="Multi-Agent VQA for ChartQA")
    ap.add_argument("--test_human", default="/home/khy/Project_CMU/chart_classification/ChartQA_Dataset/test/test_human.json")
    ap.add_argument("--test_augmented", default="/home/khy/Project_CMU/chart_classification/ChartQA_Dataset/test/test_augmented.json")
    ap.add_argument("--img_root", default="/home/khy/Project_CMU/chart_classification/ChartQA_Dataset/test/png")
    ap.add_argument("--out_dir", default="/home/khy/Project_CMU/chart-understanding/multi_agent_output",
                    help="Output directory for all results")
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    
    # Load data
    with open(args.test_human, "r") as f:
        human_entries = json.load(f)
    with open(args.test_augmented, "r") as f:
        aug_entries = json.load(f)
    
    # Inference
    preds_h = run_split(human_entries, args.img_root, "test_human", args.model, args.out_dir)
    preds_a = run_split(aug_entries, args.img_root, "test_augmented", args.model, args.out_dir)
    all_preds = preds_h + preds_a
    
    # Save predictions
    pred_path = os.path.join(args.out_dir, "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(all_preds, f, indent=2)
    
    # Evaluation
    acc_h = compute_accuracy(preds_h)
    acc_a = compute_accuracy(preds_a)
    total = len(preds_h) + len(preds_a)
    acc_o = (acc_h * len(preds_h) + acc_a * len(preds_a)) / total if total else 0.0
    
    eval_json = {
        "test_human": round(acc_h * 100, 2),
        "test_augmented": round(acc_a * 100, 2),
        "overall": round(acc_o * 100, 2)
    }
    
    eval_path = os.path.join(args.out_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(eval_json, f, indent=2)
    
    print("\n────────  Multi-Agent Inference Complete  ────────")
    for k, v in eval_json.items():
        print(f"{k:>15}: {v:.2f}%")
    print(f"Predictions  : {pred_path}")
    print(f"Evaluation   : {eval_path}")
    print(f"Intermediates: {os.path.join(args.out_dir, 'intermediate')}")
    print(f"Elapsed time : {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()