#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────
#  Planning-Perception Multi-Agent VQA System
#  - Max 2 iterations with early stopping
#  - Planning Agent extracts final answer from Image Agent's analysis
#  - Flexible answer matching (target contained in prediction = correct)
# ────────────────────────────────────────────────────────────────────
from __future__ import annotations
import argparse, json, os, time
from typing import Any, Dict, List, Tuple, Optional
import math
from tqdm.auto import tqdm
import multiprocessing
from multiprocessing import Process, Queue
import sys
import re

import torch
from PIL import Image

# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

# -------------------------------------------------------------------#
#  Constants                                                          #
# -------------------------------------------------------------------#
MODEL_ID   = "Qwen/Qwen2.5-VL-7B-Instruct"
MIN_PIXELS = 1280 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_ITERATIONS = 3  # Maximum number of Image Agent iterations

# -------------------------------------------------------------------#
#  Model loader with GPU assignment                                   #
# -------------------------------------------------------------------#
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def load_qwen(model_path: str, device_id: int):
    """Load Qwen-VL model with proper GPU assignment."""
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
#  Persistent Worker Processes                                        #
# -------------------------------------------------------------------#
def persistent_planning_agent(task_queue: Queue, result_queue: Queue, model_path: str):
    """Planning agent - understands questions and structures problem-solving approach."""
    from qwen_vl_utils import process_vision_info
    
    # Load model once on GPU 1
    processor, model = load_qwen(model_path, device_id=1)
    
    while True:
        task = task_queue.get()
        if task is None:  # Poison pill
            break
            
        task_type = task.get('type')
        
        if task_type == 'initial_planning':
            image_path = task['image_path']
            question = task['question']
            task_id = task['task_id']
            
            prompt = f"""You are a QUESTION UNDERSTANDING AND PLANNING SPECIALIST for chart visualizations.

QUESTION: {question}

GOALS:
1. Understand the intent of the question and what constitutes a correct answer.
2. Specify exactly which values, labels, colors, time ranges, or comparisons must be recovered from the chart.
3. Produce a concrete plan the perception agent can execute without guessing.

OUTPUT FORMAT (keep the headings):
Question Type: [classification such as data extraction, comparison, calculation, counting, text identification, boolean]
Answer Expectation: [describe the required format—number + units/precision, specific label/color/category, yes/no, list, etc.]
Critical Evidence Needed:
- [...]
Plan Steps:
1. [...]
2. [...]
Answer Validation:
- [checks to confirm the plan solves the question, e.g., “ensure label not percentage,” “recompute totals”]
Key Instructions for Image Analysis:
- [call out legend usage, axis reading order, arithmetic reminders, etc.]

Be explicit about whether the answer should be a textual label or a numeric value. Mention any units, color names, ratios, or operations that must be performed.
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
                gen_ids = model.generate(**inputs, max_new_tokens=300)
            
            new_tokens = gen_ids[0][len(inputs.input_ids[0]):]
            planning_output = processor.decode(new_tokens, skip_special_tokens=True).strip()
            
            result = {
                "type": "initial_planning",
                "planning": planning_output,
                "task_id": task_id
            }
            result_queue.put(result)
            
        elif task_type == 'refinement_planning':
            image_path = task['image_path']
            question = task['question']
            initial_plan = task['initial_plan']
            perception_result = task['perception_result']
            task_id = task['task_id']
            iteration = task['iteration']
            
            prompt = f"""You are a STRICT EVALUATOR verifying whether the perception agent's analysis
fully satisfies the original plan and contains the exact information needed for the final answer.

QUESTION: {question}

INITIAL PLAN (includes Answer Expectation and Critical Evidence):
{initial_plan}

PERCEPTION AGENT'S ANALYSIS:
{perception_result}

TASK:
Judge objectively whether the perception agent's report already provides all evidence and reasoning necessary to extract the correct final answer.
Focus on **content accuracy and logical completeness**, not writing style.

CHECKLIST:
1. **Answer Expectation Match** – Does the analysis provide the *same answer type* (text label / numeric value / list / yes-no) as required?
2. **Evidence Completeness** – Are *all Critical Evidence Needed* from the plan explicitly found in the perception report?
3. **Computation Verification** – If arithmetic, ratio, or difference is required, was it *explicitly computed and correct*?
4. **Logical Condition Handling** – Were constraints such as "exclude," "only if," "outside," or "difference between" correctly applied?
5. **Answer Relevance & Specificity** – Is the "Answer Candidate" *directly addressing the question* (no unrelated or vague text)? Even if verbose, ensure the correct value or label is clearly contained.

EXTRA STRICTNESS FOR SPECIFIC QUESTION TYPES:
- COUNTING questions ("how many"): Mark Evidence Completeness as FAIL unless the perception agent explicitly lists each counted item with its exact value. A bare number without supporting list is INSUFFICIENT.
- COLOR questions (mentions color names): Mark Answer Relevance as FAIL if the color description is too generic (e.g., "blue" when should be "navy blue" or "dark blue"). Require precise color descriptors.

OUTPUT FORMAT:
Sufficient: [Yes/No]
Checklist Summary:
- Answer Expectation Match: [Pass/Fail]
- Evidence Completeness: [Pass/Fail]
- Computation Verification: [Pass/Fail]
- Logical Condition Handling: [Pass/Fail]
- Answer Relevance & Specificity: [Pass/Fail]
Reasoning: [Brief, concrete justification for each failed item.]
Correction Instructions:
- [List concrete fixes, e.g., "Verify ratio computation between red and blue bars," "Exclude Helsinki from ranking before selecting 3rd largest."]

Final rule:
If **any** item above is marked Fail → Sufficient: No
Only if all are Pass → Sufficient: Yes
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
                gen_ids = model.generate(**inputs, max_new_tokens=200)
            
            new_tokens = gen_ids[0][len(inputs.input_ids[0]):]
            refinement = processor.decode(new_tokens, skip_special_tokens=True).strip()
            
            result = {
                "type": "refinement",
                "refinement": refinement,
                "iteration": iteration,
                "task_id": task_id
            }
            result_queue.put(result)
            
        elif task_type == 'final_answer':
            question = task['question']
            final_perception = task['final_perception']
            initial_plan_text = task.get('initial_plan', '')
            task_id = task['task_id']
            
            # Planning agent extracts final answer based on perception analysis
            prompt = f"""You are the PLANNING AGENT responsible for producing the final answer exactly as requested.

QUESTION: {question}

PLANNING SUMMARY (includes Answer Expectation):
{initial_plan_text}

IMAGE AGENT'S DETAILED ANALYSIS:
{final_perception}

INSTRUCTIONS:
1. Follow the Answer Expectation from the plan (text label, color, number with units/precision, yes/no, etc.).
2. Use the provided evidence to compute or extract the required value; double-check any arithmetic (sums, ratios, medians).
3. If the question asks "which/what" about a label or color, respond with the label itself (not a percentage).
4. Give only the final answer string—no explanation, no extra words.

FORMAT REQUIREMENTS:
- For yes/no questions: Answer "Yes" or "No" (NOT "True" or "False")
- For ratio questions: Provide decimal number (e.g., 1.058) unless question explicitly asks for fraction format
- For counting questions: Provide integer only (e.g., "2" not "2 bars")
- Answer must be in ENGLISH only - no other languages

FINAL ANSWER (answer only):"""
            
            # No image needed for final answer extraction
            content = [{"type": "text", "text": prompt}]
            messages = [{"role": "user", "content": content}]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, images=None, videos=None, padding=True, return_tensors="pt").to(model.device)
            
            with torch.inference_mode():
                gen_ids = model.generate(**inputs, max_new_tokens=50)
            
            new_tokens = gen_ids[0][len(inputs.input_ids[0]):]
            final_answer = processor.decode(new_tokens, skip_special_tokens=True).strip()
            
            result = {
                "type": "final_answer",
                "answer": final_answer,
                "task_id": task_id
            }
            result_queue.put(result)

def persistent_perception_agent(task_queue: Queue, result_queue: Queue, model_path: str):
    """Image perception agent - analyzes visual elements based on planning guidance."""
    from qwen_vl_utils import process_vision_info
    
    # Load model once on GPU 2
    processor, model = load_qwen(model_path, device_id=2)
    
    while True:
        task = task_queue.get()
        if task is None:  # Poison pill
            break
            
        image_path = task['image_path']
        question = task['question']
        planning_guidance = task['planning_guidance']
        task_id = task['task_id']
        iteration = task.get('iteration', 1)
        refinement = task.get('refinement', None)
        
        if refinement:
            prompt = f"""You are an IMAGE PERCEPTION SPECIALIST refining your analysis.

QUESTION: {question}

PLANNING GUIDANCE:
{planning_guidance}

REFINEMENT INSTRUCTIONS:
{refinement}

Follow every refinement point explicitly. Carefully read legends, axes, annotations, and compute any requested values.

REPORT FORMAT (use these exact headings):
Chart Overview: [...]
Key Visual/Text Details:
- ...
Key Numeric Readings:
- ...
Calculations / Comparisons:
- ...
Answer Evidence:
- ...
Answer Candidate (for planner only, do not finalize): [...]

The Answer Candidate should be a short phrase or value that directly addresses the question, matching the expected format (label vs. number, units, etc.).
"""
        else:
            prompt = f"""You are an IMAGE PERCEPTION SPECIALIST analyzing chart visualizations.

QUESTION: {question}

PLANNING GUIDANCE:
{planning_guidance}

INSTRUCTIONS:
- Inspect the entire chart methodically (titles, legends, axes, annotations, colors, numeric markings).
- Extract precise values or labels needed for the plan.
- When arithmetic is required, show the intermediate calculation.

CRITICAL CAREFULNESS FOR SPECIFIC TASKS:
- COUNTING: If counting items, list each item explicitly with its exact value (e.g., "Bar 1: 23%, Bar 2: 23%, Bar 3: 45%"). Only count items that match EXACTLY (not approximately).
- COLORS: Describe colors precisely with shade descriptors (e.g., "dark navy blue" not just "blue", "light brown" not just "brown"). Use the legend to verify color labels.
- READING VALUES: Always read from explicit numeric labels on the chart (not estimated from bar heights). Double-check each value.

REPORT FORMAT (use these exact headings):
Chart Overview: [...]
Key Visual/Text Details:
- ...
Key Numeric Readings:
- ...
Calculations / Comparisons:
- ...
Answer Evidence:
- ...
Answer Candidate (for planner only, do not finalize): [...]

Ensure the Answer Candidate states the exact label, color name, or number (with units/percent) that the question demands. Do not provide extra narrative there.
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
            gen_ids = model.generate(**inputs, max_new_tokens=400)
        
        new_tokens = gen_ids[0][len(inputs.input_ids[0]):]
        perception_output = processor.decode(new_tokens, skip_special_tokens=True).strip()
        
        result = {
            "perception": perception_output,
            "iteration": iteration,
            "task_id": task_id
        }
        result_queue.put(result)

# -------------------------------------------------------------------#
#  Orchestrator function                                              #
# -------------------------------------------------------------------#
def sequential_vqa_with_refinement(image_path: str, question: str, imgname: str,
                                  planning_queue: Queue, planning_result_queue: Queue,
                                  perception_queue: Queue, perception_result_queue: Queue,
                                  task_id: int, output_dir: str):
    """Orchestrate the sequential planning-perception pipeline with max 2 iterations."""
    
    # Step 1: Initial planning
    planning_queue.put({
        'type': 'initial_planning',
        'image_path': image_path,
        'question': question,
        'task_id': task_id
    })
    
    planning_result = planning_result_queue.get()
    initial_plan = planning_result['planning']
    
    # Iterative perception-refinement loop (max 2 iterations)
    perception_history = []
    refinement_history = []
    final_perception = None
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        # Perception step
        perception_task = {
            'image_path': image_path,
            'question': question,
            'planning_guidance': initial_plan,
            'task_id': task_id,
            'iteration': iteration
        }
        
        # Add refinement instructions if this is not the first iteration
        if iteration > 1 and refinement_history:
            perception_task['refinement'] = refinement_history[-1]
        
        perception_queue.put(perception_task)
        perception_result = perception_result_queue.get()
        current_perception = perception_result['perception']
        perception_history.append(current_perception)
        
        # Refinement check by Planning Agent
        planning_queue.put({
            'type': 'refinement_planning',
            'image_path': image_path,
            'question': question,
            'initial_plan': initial_plan,
            'perception_result': current_perception,
            'task_id': task_id,
            'iteration': iteration
        })
        
        refinement_result = planning_result_queue.get()
        current_refinement = refinement_result['refinement']
        refinement_history.append(current_refinement)
        
        # Check if answer is sufficient (early stopping)
        if "Sufficient: Yes" in current_refinement or \
           ("sufficient" in current_refinement.lower() and "yes" in current_refinement.lower()):
            print(f"  [Iteration {iteration}] Planning Agent: Answer is SUFFICIENT - stopping early")
            final_perception = current_perception
            break
        else:
            print(f"  [Iteration {iteration}] Planning Agent: Needs refinement - {'final attempt' if iteration == MAX_ITERATIONS else 'continuing'}")
            
            # If this is the last iteration, use this perception as final
            if iteration == MAX_ITERATIONS:
                final_perception = current_perception
                break
    
    # Extract final answer using Planning Agent
    planning_queue.put({
        'type': 'final_answer',
        'question': question,
        'final_perception': final_perception,
        'initial_plan': initial_plan,
        'task_id': task_id
    })
    
    final_result = planning_result_queue.get()
    final_answer = final_result['answer']
    
    # Build detailed result
    detailed_result = {
        "imgname": imgname,
        "question": question,
        "initial_planning": initial_plan,
        "iterations": len(perception_history),
        "final_answer": final_answer
    }
    
    # Add iteration details
    for i, (perception, refinement) in enumerate(zip(perception_history, refinement_history), 1):
        detailed_result[f"perception_iteration_{i}"] = perception
        detailed_result[f"refinement_{i}"] = refinement
    
    detailed_result["final_perception"] = final_perception
    
    # Save to intermediate directory
    inter_path = os.path.join(output_dir, "intermediate", f"{os.path.splitext(imgname)[0]}.json")
    os.makedirs(os.path.dirname(inter_path), exist_ok=True)
    with open(inter_path, "w") as f:
        json.dump(detailed_result, f, indent=2)
    
    return final_answer, detailed_result

# -------------------------------------------------------------------#
#  Improved accuracy checking                                         #
# -------------------------------------------------------------------#
def relaxed_correctness(target: str, prediction: str, max_relative_change: float = 0.05) -> bool:
    """Simple but robust answer matching with scaling tolerance (e.g., % -> /100)."""
    import re

    if not target or not prediction:
        return False

    prediction = str(prediction).strip()
    target = str(target).strip()

    for prefix in ["Answer:", "Final Answer:", "The answer is", "FINAL ANSWER:", "final answer:"]:
        if prediction.lower().startswith(prefix.lower()):
            prediction = prediction[len(prefix):].strip()

    # 1️⃣ Text normalization (공백, 콤마 제거)
    pred_normalized = prediction.replace(',', '').replace(' ', '')
    target_normalized = target.replace(',', '').replace(' ', '')

    if target_normalized.lower() in pred_normalized.lower():
        return True

    # 2️⃣ Numeric comparison (±5% tolerance)
    def extract_number(text):
        nums = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
        return float(nums[0]) if nums else None

    p_num = extract_number(prediction)
    t_num = extract_number(target)

    if p_num is not None and t_num is not None:
        if t_num == 0:
            return abs(p_num) < 0.01

        rel_error = abs(p_num - t_num) / abs(t_num)
        if rel_error <= max_relative_change:
            return True

        # 3️⃣ Scale correction check (%, 단위 누락 등)
        #   예: 0.22675 vs 22.675 → 22.675 / 100 ≈ 0.22675
        scaled_variants = [p_num / 100, p_num * 100]
        for scaled in scaled_variants:
            rel_error_scaled = abs(scaled - t_num) / abs(t_num)
            if rel_error_scaled <= max_relative_change:
                return True

    return False


# -------------------------------------------------------------------#
#  Batch runner                                                      #
# -------------------------------------------------------------------#
def run_split(entries, img_root, split_name, output_dir,
              planning_queue, planning_result_queue,
              perception_queue, perception_result_queue):
    """Run sequential VQA over one split using persistent workers."""
    results = []
    
    for idx, ex in enumerate(tqdm(entries, desc=f"Infer {split_name}", ncols=80)):
        img_path = os.path.join(img_root, ex["imgname"])
        if not os.path.exists(img_path):
            print(f"[{split_name}] SKIP (missing image): {ex['imgname']}")
            continue
        
        final_pred, detailed_data = sequential_vqa_with_refinement(
            img_path, ex["query"], ex["imgname"],
            planning_queue, planning_result_queue,
            perception_queue, perception_result_queue,
            idx, output_dir
        )
        
        print(f"[{split_name}] Q: {ex['query'][:50]}...  →  {final_pred}")
        
        rec = {
            "imgname": ex["imgname"],
            "query": ex["query"],
            "prediction": final_pred,
            "answer": ex["label"],
            "split": split_name,
            "planning_output": detailed_data["initial_planning"],
            "final_perception": detailed_data["final_perception"]
        }
        results.append(rec)
    
    return results

def compute_accuracy(recs: List[Dict[str, Any]]) -> float:
    """Compute accuracy with relaxed criteria."""
    if not recs:
        return 0.0
    hits = sum(relaxed_correctness(r["answer"], r["prediction"]) for r in recs)
    return hits / len(recs)

# -------------------------------------------------------------------#
#  CLI                                                               #
# -------------------------------------------------------------------#
def main():
    ap = argparse.ArgumentParser(description="Planning-Perception Multi-Agent VQA for ChartQA")
    ap.add_argument("--test_human", default="/home/khy/Project_CMU/chart_classification/ChartQA_Dataset/test_simple/test_human.json")
    ap.add_argument("--test_augmented", default="/home/khy/Project_CMU/chart_classification/ChartQA_Dataset/test_simple/test_augmented.json")
    ap.add_argument("--img_root", default="/home/khy/Project_CMU/chart_classification/ChartQA_Dataset/test_simple/png")
    ap.add_argument("--out_dir", default="/home/khy/Project_CMU/chart-understanding/planning_perception_output_small_v7",
                    help="Output directory for all results")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--sample_size", type=int, default=None, help="Number of samples to process (for testing)")
    args = ap.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    
    # Create queues for persistent workers
    planning_task_queue = Queue()
    planning_result_queue = Queue()
    perception_task_queue = Queue()
    perception_result_queue = Queue()
    
    # Start persistent worker processes
    print("Starting persistent worker processes...")
    print("Planning Agent -> GPU 1")
    print("Perception Agent -> GPU 2")
    
    planning_worker = Process(target=persistent_planning_agent, 
                            args=(planning_task_queue, planning_result_queue, args.model))
    perception_worker = Process(target=persistent_perception_agent,
                              args=(perception_task_queue, perception_result_queue, args.model))
    
    planning_worker.start()
    perception_worker.start()
    
    # Wait for models to load
    print("Waiting for models to load...")
    time.sleep(20)
    
    try:
        # Load data
        with open(args.test_human, "r") as f:
            human_entries = json.load(f)
        with open(args.test_augmented, "r") as f:
            aug_entries = json.load(f)
        
        # Apply sample size if specified
        if args.sample_size:
            human_entries = human_entries[:args.sample_size]
            aug_entries = aug_entries[:args.sample_size]
        
        # Inference
        print("\n" + "="*50)
        print("Starting Sequential Planning-Perception VQA")
        print("="*50 + "\n")
        
        preds_h = run_split(human_entries, args.img_root, "test_human", args.out_dir,
                          planning_task_queue, planning_result_queue,
                          perception_task_queue, perception_result_queue)
        
        preds_a = run_split(aug_entries, args.img_root, "test_augmented", args.out_dir,
                          planning_task_queue, planning_result_queue,
                          perception_task_queue, perception_result_queue)
        
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
            "overall": round(acc_o * 100, 2),
            "samples_processed": {
                "human": len(preds_h),
                "augmented": len(preds_a),
                "total": total
            }
        }
        
        eval_path = os.path.join(args.out_dir, "evaluation.json")
        with open(eval_path, "w") as f:
            json.dump(eval_json, f, indent=2)
        
        print("\n" + "="*60)
        print("  Planning-Perception Multi-Agent Inference Complete")
        print("="*60)
        for k, v in eval_json.items():
            if k != "samples_processed":
                print(f"{k:>15}: {v:.2f}%")
        print(f"\nSamples processed: {eval_json['samples_processed']['total']}")
        print(f"Predictions      : {pred_path}")
        print(f"Evaluation       : {eval_path}")
        print(f"Intermediates    : {os.path.join(args.out_dir, 'intermediate')}")
        print(f"Elapsed time     : {time.time() - t0:.1f}s")
        
    finally:
        # Stop workers
        print("\nStopping workers...")
        planning_task_queue.put(None)
        perception_task_queue.put(None)
        
        planning_worker.join(timeout=5)
        perception_worker.join(timeout=5)
        
        if planning_worker.is_alive():
            planning_worker.terminate()
        if perception_worker.is_alive():
            perception_worker.terminate()

if __name__ == "__main__":
    main()
