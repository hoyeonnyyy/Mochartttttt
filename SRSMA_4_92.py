#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────
#  Self-Refining Chart-Type Specialized Multi-Agent VQA System
#  - Chart-type classification → specialized perception agents
#  - Instruction refinement through iterative feedback
#  - Cross-problem learning with instruction memory
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
MAX_ITERATIONS = 3  # Maximum number of perception iterations per problem

# Chart type categories (ChartQA only has these 3 types)
CHART_TYPES = ["bar", "line", "pie"]

# Initial instructions for each chart type (simple baseline)
INITIAL_INSTRUCTIONS = {
    "bar": "Analyze the bar chart carefully. Follow the planning agent's guidance and provide a simple, clear answer.",
    "line": "Analyze the line chart carefully. Follow the planning agent's guidance and provide a simple, clear answer.",
    "pie": "Analyze the pie chart carefully. Follow the planning agent's guidance and provide a simple, clear answer."
}

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
#  Instruction Memory Manager                                         #
# -------------------------------------------------------------------#
class InstructionMemory:
    """Manages chart-type specific instructions that evolve over time."""
    
    def __init__(self, output_dir: str):
        self.instructions = INITIAL_INSTRUCTIONS.copy()
        self.history = {chart_type: [] for chart_type in CHART_TYPES}
        self.output_dir = output_dir
        
        # Create instruction tracking directory
        self.instruction_dir = os.path.join(output_dir, "instruction_evolution")
        os.makedirs(self.instruction_dir, exist_ok=True)
    
    def get_instruction(self, chart_type: str) -> str:
        """Get current instruction for a chart type."""
        return self.instructions.get(chart_type, INITIAL_INSTRUCTIONS["bar"])
    
    def update_instruction(self, chart_type: str, new_instruction: str, problem_id: int):
        """Update instruction for a chart type and log the change."""
        old_instruction = self.instructions[chart_type]
        self.instructions[chart_type] = new_instruction
        
        # Log the change
        self.history[chart_type].append({
            "problem_id": problem_id,
            "old_instruction": old_instruction,
            "new_instruction": new_instruction,
            "timestamp": time.time()
        })
        
        # Save to file
        self._save_instruction_history(chart_type)
    
    def _save_instruction_history(self, chart_type: str):
        """Save instruction evolution history for a chart type."""
        history_path = os.path.join(self.instruction_dir, f"{chart_type}_history.json")
        with open(history_path, "w") as f:
            json.dump({
                "current_instruction": self.instructions[chart_type],
                "history": self.history[chart_type]
            }, f, indent=2)
    
    def save_final_state(self):
        """Save final instruction state for all chart types."""
        final_state_path = os.path.join(self.instruction_dir, "final_instructions.json")
        with open(final_state_path, "w") as f:
            json.dump(self.instructions, f, indent=2)
        
        print(f"\n{'='*60}")
        print("FINAL INSTRUCTION STATE")
        print('='*60)
        for chart_type, instruction in self.instructions.items():
            print(f"\n[{chart_type.upper()}]")
            print(f"{instruction}")
        print('='*60)

# -------------------------------------------------------------------#
#  Persistent Worker Processes                                        #
# -------------------------------------------------------------------#
def persistent_planning_agent(task_queue: Queue, result_queue: Queue, model_path: str):
    """Planning agent - classifies chart type, refines instructions, extracts answers."""
    from qwen_vl_utils import process_vision_info
    
    # Load model once on GPU 1
    processor, model = load_qwen(model_path, device_id=1)
    
    while True:
        task = task_queue.get()
        if task is None:  # Poison pill
            break
            
        task_type = task.get('type')
        
        if task_type == 'initial_planning':
            # Chart classification + Initial planning in one call
            image_path = task['image_path']
            question = task['question']
            task_id = task['task_id']
            
            prompt = f"""You are a chart question answering PLANNING SPECIALIST.

QUESTION: {question}

YOUR TASK:
Identify the chart type and generate a solving plan specifying Solution Steps, Critical Data, and the required Answer Format.

IMPORTANT: Do NOT extract actual values from the chart. Only specify WHAT needs to be extracted and HOW to solve.

OUTPUT:
Chart Type: [bar/line/pie]
Solution Steps: [numbered steps explaining how to solve this problem]
Critical Data: [WHAT specific items need to be extracted from the chart - describe them but do NOT provide actual values]
Answer Format: [describe the type of answer required, such as a single label, a number, yes/no, or a list]

Begin your plan:"""
            
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
            planning_output = processor.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Extract chart type from planning output
            chart_type = "bar"  # Default
            planning_lower = planning_output.lower()
            for ct in CHART_TYPES:
                if f"chart type: {ct}" in planning_lower or f"{ct} chart" in planning_lower:
                    chart_type = ct
                    break
            
            result = {
                "type": "initial_planning",
                "chart_type": chart_type,
                "planning_output": planning_output,
                "task_id": task_id
            }
            result_queue.put(result)
        
        elif task_type == 'instruction_refinement':
            # Evaluate perception against original plan and refine instruction
            image_path = task['image_path']
            question = task['question']
            original_planning = task['original_planning']
            current_instruction = task['current_instruction']
            perception_result = task['perception_result']
            chart_type = task['chart_type']
            task_id = task['task_id']
            iteration = task['iteration']
            
            prompt = f"""You are an EVALUATION AGENT verifying the perception agent's work.

    QUESTION: {question}
    CHART TYPE: {chart_type}

    YOUR ORIGINAL PLAN:
    {original_planning}

    PERCEPTION RESULT:
    {perception_result}

    YOUR TASK:
    Act as a FACT-CHECKER. Verify each step of the perception's work.

    DETAILED CHECKLIST:

    1. Value Extraction Accuracy
    - Did they extract the ACTUAL NUMERICAL VALUES from the chart?
    - Are the values PRECISE (not approximated)?
    - Verify by looking at the chart YOURSELF

    2. Reasoning Correctness
    - Is the logical reasoning VALID?
    - Are comparisons and interpretations correct?
    - Any logical errors or false assumptions?

    3. Calculation Verification
    - If arithmetic was needed, PERFORM THE CALCULATION YOURSELF
    - Show your work: [value1] [operator] [value2] = [result]
    - Recalculate yourself: Does it match?
    - Check for arithmetic errors, unit mistakes

    4. Answer Correctness
    - Is the final answer CORRECT based on data and calculations?
    - Does it logically follow from the reasoning?
    - Does it answer the question asked?

    Sufficiency RULES:
    - If ANY item is Fail → Sufficient: No.
    - Only if ALL items are Pass → Sufficient: Yes.

    OUTPUT FORMAT (produce EXACTLY this format):

    Value Extraction: Pass/Fail
    Reasoning Logic: Pass/Fail
    Calculation: Pass/Fail
    Answer Correctness: Pass/Fail
    Sufficient: Yes/No

    Prompt Refinement Guideline:
    [If Sufficient: No, write ONE short general guideline (1 sentence) addressing the error TYPE.
    Format as: "- When [situation], [action to take]."
    If Sufficient=Yes, write: "none"]

    Begin evaluation:"""
            
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
            refinement_output = processor.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Parse the output
            is_sufficient = "sufficient: yes" in refinement_output.lower()
            
            # Extract additional guideline if provided
            additional_guideline = None
            if not is_sufficient:
                # Look for "Additional Guideline:" section
                match = re.search(r'Prompt Refinement Guideline:\s*(.+?)(?:\n\n|\Z)', refinement_output, re.DOTALL | re.IGNORECASE)
                if match:
                    guideline_text = match.group(1).strip()
                    if guideline_text.lower() != "none":
                        additional_guideline = guideline_text
            
            result = {
                "type": "instruction_refinement",
                "is_sufficient": is_sufficient,
                "refinement_output": refinement_output,
                "additional_guideline": additional_guideline,
                "iteration": iteration,
                "task_id": task_id
            }
            result_queue.put(result)

def persistent_perception_agent(task_queue: Queue, result_queue: Queue, model_path: str):
    """Specialized perception agent - analyzes charts based on type-specific instructions."""
    from qwen_vl_utils import process_vision_info
    
    # Load model once on GPU 2
    processor, model = load_qwen(model_path, device_id=2)
    
    while True:
        task = task_queue.get()
        if task is None:  # Poison pill
            break
            
        image_path = task['image_path']
        question = task['question']
        chart_type = task['chart_type']
        planning_guidance = task['planning_guidance']
        instruction = task['instruction']
        task_id = task['task_id']
        iteration = task.get('iteration', 1)
        
        prompt = f"""You are a SPECIALIZED {chart_type.upper()} CHART ANALYST.

QUESTION: {question}

PLANNING GUIDANCE:
{planning_guidance}

YOUR TASK:
Follow the Solution Steps from the plan, extract the items listed in Critical Data,
and produce an answer that matches the required Answer Format.

SPECIALIZED INSTRUCTION:
{instruction}

OUTPUT FORMAT (produce EXACTLY these three sections):

Data Extraction:
- List the extracted Critical Data items exactly as required in the plan.
- ALWAYS include the actual numerical values from the chart, not just category names.

Reasoning:
- Explain your approach AND perform any necessary calculations.

Answer Candidate:
- Only the final answer.
- Must follow the exact Answer Format.

Begin your analysis:
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
            gen_ids = model.generate(**inputs, max_new_tokens=500)
        
        new_tokens = gen_ids[0][len(inputs.input_ids[0]):]
        perception_output = processor.decode(new_tokens, skip_special_tokens=True).strip()
        
        result = {
            "perception": perception_output,
            "iteration": iteration,
            "chart_type": chart_type,
            "task_id": task_id
        }
        result_queue.put(result)

# -------------------------------------------------------------------#
#  Helper function to extract answer from perception                 #
# -------------------------------------------------------------------#
def extract_answer_from_perception(perception_output: str) -> str:
    """Extract the Answer Candidate from perception output."""
    # Look for "Answer Candidate:" section
    match = re.search(r'Answer Candidate:\s*(.+?)(?:\n|$)', perception_output, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        # Clean up common prefixes
        for prefix in ["Answer:", "Final Answer:", "The answer is"]:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        return answer
    
    # Fallback: return last line if no "Answer Candidate" found
    lines = [l.strip() for l in perception_output.strip().split('\n') if l.strip()]
    return lines[-1] if lines else perception_output.strip()

# -------------------------------------------------------------------#
#  Orchestrator function                                              #
# -------------------------------------------------------------------#
def self_refining_vqa(image_path: str, question: str, imgname: str,
                     planning_queue: Queue, planning_result_queue: Queue,
                     perception_queue: Queue, perception_result_queue: Queue,
                     task_id: int, output_dir: str, instruction_memory: InstructionMemory):
    """Orchestrate the self-refining specialized agent pipeline."""
    
    # Step 1: Chart type classification + Initial planning (combined)
    planning_queue.put({
        'type': 'initial_planning',
        'image_path': image_path,
        'question': question,
        'task_id': task_id
    })
    
    planning_result = planning_result_queue.get()
    chart_type = planning_result['chart_type']
    original_planning = planning_result['planning_output']
    
    print(f"  [Chart Type: {chart_type}]", end=" ")
    
    # Step 2: Get global accumulated instruction
    global_instruction = instruction_memory.get_instruction(chart_type)
    
    # Step 3: Iterative perception-refinement loop
    perception_history = []
    refinement_history = []
    final_perception = None
    problem_guideline = None  # This problem's guideline (replace within problem)
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        # Build current instruction: global + problem_guideline
        if problem_guideline:
            current_instruction = global_instruction + "\n" + problem_guideline
        else:
            current_instruction = global_instruction
        
        # Perception step with current instruction and planning guidance
        perception_queue.put({
            'image_path': image_path,
            'question': question,
            'chart_type': chart_type,
            'planning_guidance': original_planning,
            'instruction': current_instruction,
            'task_id': task_id,
            'iteration': iteration
        })
        
        perception_result = perception_result_queue.get()
        current_perception = perception_result['perception']
        perception_history.append(current_perception)
        
        # Instruction refinement check
        planning_queue.put({
            'type': 'instruction_refinement',
            'image_path': image_path,
            'question': question,
            'original_planning': original_planning,
            'current_instruction': current_instruction,
            'perception_result': current_perception,
            'chart_type': chart_type,
            'task_id': task_id,
            'iteration': iteration
        })
        
        refinement_result = planning_result_queue.get()
        is_sufficient = refinement_result['is_sufficient']
        refinement_history.append(refinement_result['refinement_output'])
        
        if is_sufficient:
            print(f"[Iter {iteration}: Sufficient ✓]", end=" ")
            final_perception = current_perception
            break
        else:
            print(f"[Iter {iteration}: Refining]", end=" ")
            
            # REPLACE problem_guideline (not append within problem)
            if refinement_result['additional_guideline']:
                problem_guideline = refinement_result['additional_guideline']
            
            # If last iteration, use current perception
            if iteration == MAX_ITERATIONS:
                final_perception = current_perception
                break
    
    # Step 4: After problem ends - APPEND problem_guideline to global memory
    if problem_guideline:
        new_global_instruction = global_instruction + "\n" + problem_guideline
        instruction_memory.update_instruction(chart_type, new_global_instruction, task_id)
    
    # Step 5: Extract final answer directly from perception
    final_answer = extract_answer_from_perception(final_perception)
    
    # Build detailed result
    detailed_result = {
        "imgname": imgname,
        "question": question,
        "chart_type": chart_type,
        "original_planning": original_planning,
        "global_instruction_used": global_instruction,
        "problem_guideline_added": problem_guideline,
        "final_instruction": new_global_instruction if problem_guideline else global_instruction,
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
    """Simple but robust answer matching with scaling tolerance, including ratio a:b normalization."""
    import re

    if not target or not prediction:
        return False

    prediction = str(prediction).strip()
    target = str(target).strip()

    # Remove prefix patterns
    for prefix in ["Answer:", "Final Answer:", "The answer is", "FINAL ANSWER:", "final answer:"]:
        if prediction.lower().startswith(prefix.lower()):
            prediction = prediction[len(prefix):].strip()

    # 1️⃣ Text normalization
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

    # If both numbers exist, compare directly
    if p_num is not None and t_num is not None:
        if t_num == 0:
            return abs(p_num) < 0.01

        rel_error = abs(p_num - t_num) / abs(t_num)
        if rel_error <= max_relative_change:
            return True

        # 3️⃣ Scale correction check
        scaled_variants = [p_num / 100, p_num * 100]
        for scaled in scaled_variants:
            rel_error_scaled = abs(scaled - t_num) / abs(t_num)
            if rel_error_scaled <= max_relative_change:
                return True

    # 4️⃣ Ratio “a:b” → decimal normalization
    # Example: prediction="6:5", target="1.2"
    ratio_match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*$", prediction)
    if ratio_match:
        a = float(ratio_match.group(1))
        b = float(ratio_match.group(2))
        if b != 0 and t_num is not None:
            ratio_val = a / b
            rel_error_ratio = abs(ratio_val - t_num) / abs(t_num)
            if rel_error_ratio <= max_relative_change:
                return True

    return False


# -------------------------------------------------------------------#
#  Batch runner                                                      #
# -------------------------------------------------------------------#
def run_split(entries, img_root, split_name, output_dir,
              planning_queue, planning_result_queue,
              perception_queue, perception_result_queue,
              instruction_memory):
    """Run self-refining VQA over one split using persistent workers."""
    results = []
    
    for idx, ex in enumerate(tqdm(entries, desc=f"Infer {split_name}", ncols=80)):
        img_path = os.path.join(img_root, ex["imgname"])
        if not os.path.exists(img_path):
            print(f"[{split_name}] SKIP (missing image): {ex['imgname']}")
            continue
        
        final_pred, detailed_data = self_refining_vqa(
            img_path, ex["query"], ex["imgname"],
            planning_queue, planning_result_queue,
            perception_queue, perception_result_queue,
            idx, output_dir, instruction_memory
        )
        
        print(f"→ {final_pred[:50]}")
        
        rec = {
            "imgname": ex["imgname"],
            "query": ex["query"],
            "prediction": final_pred,
            "answer": ex["label"],
            "split": split_name,
            "chart_type": detailed_data["chart_type"]
        }
        results.append(rec)
    
    return results

def compute_accuracy(recs: List[Dict[str, Any]]) -> float:
    """Compute accuracy with relaxed criteria."""
    if not recs:
        return 0.0
    hits = sum(relaxed_correctness(r["answer"], r["prediction"]) for r in recs)
    return hits / len(recs)

def compute_accuracy_by_chart_type(recs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute accuracy breakdown by chart type."""
    by_type = {}
    for chart_type in CHART_TYPES:
        type_recs = [r for r in recs if r.get("chart_type") == chart_type]
        if type_recs:
            by_type[chart_type] = compute_accuracy(type_recs)
    return by_type

# -------------------------------------------------------------------#
#  CLI                                                               #
# -------------------------------------------------------------------#
def main():
    ap = argparse.ArgumentParser(description="Self-Refining Chart-Type Specialized Multi-Agent VQA")
    ap.add_argument("--test_human", default="/home/khy/Project_CMU/chart_classification/ChartQA_Dataset/test_simple/test_human.json")
    ap.add_argument("--test_augmented", default="/home/khy/Project_CMU/chart_classification/ChartQA_Dataset/test_simple/test_augmented.json")
    ap.add_argument("--img_root", default="/home/khy/Project_CMU/chart_classification/ChartQA_Dataset/test_simple/png")
    ap.add_argument("--out_dir", default="/home/khy/Project_CMU/chart-understanding/self_refining_output_v4",
                    help="Output directory for all results")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--sample_size", type=int, default=None, help="Number of samples to process (for testing)")
    args = ap.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    
    # Initialize instruction memory
    instruction_memory = InstructionMemory(args.out_dir)
    
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
        print("Starting Self-Refining Specialized Agent VQA")
        print("="*50 + "\n")
        
        preds_h = run_split(human_entries, args.img_root, "test_human", args.out_dir,
                          planning_task_queue, planning_result_queue,
                          perception_task_queue, perception_result_queue,
                          instruction_memory)
        
        preds_a = run_split(aug_entries, args.img_root, "test_augmented", args.out_dir,
                          planning_task_queue, planning_result_queue,
                          perception_task_queue, perception_result_queue,
                          instruction_memory)
        
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
        
        # Compute accuracy by chart type
        acc_by_type_h = compute_accuracy_by_chart_type(preds_h)
        acc_by_type_a = compute_accuracy_by_chart_type(preds_a)
        acc_by_type_all = compute_accuracy_by_chart_type(all_preds)
        
        eval_json = {
            "test_human": round(acc_h * 100, 2),
            "test_augmented": round(acc_a * 100, 2),
            "overall": round(acc_o * 100, 2),
            "by_chart_type": {
                chart_type: round(acc * 100, 2)
                for chart_type, acc in acc_by_type_all.items()
            },
            "samples_processed": {
                "human": len(preds_h),
                "augmented": len(preds_a),
                "total": total
            },
            "instruction_updates": {
                chart_type: len(instruction_memory.history[chart_type])
                for chart_type in CHART_TYPES
            }
        }
        
        eval_path = os.path.join(args.out_dir, "evaluation.json")
        with open(eval_path, "w") as f:
            json.dump(eval_json, f, indent=2)
        
        # Save final instruction state
        instruction_memory.save_final_state()
        
        print("\n" + "="*60)
        print("  Self-Refining Specialized Agent Inference Complete")
        print("="*60)
        for k, v in eval_json.items():
            if k not in ["samples_processed", "instruction_updates", "by_chart_type"]:
                print(f"{k:>15}: {v:.2f}%")
        
        print("\nAccuracy by Chart Type:")
        for chart_type, acc in eval_json["by_chart_type"].items():
            print(f"  {chart_type:>10}: {acc:.2f}%")
        
        print("\nInstruction Updates per Chart Type:")
        for chart_type, count in eval_json["instruction_updates"].items():
            print(f"  {chart_type:>10}: {count} updates")
        
        print(f"\nSamples processed: {eval_json['samples_processed']['total']}")
        print(f"Predictions      : {pred_path}")
        print(f"Evaluation       : {eval_path}")
        print(f"Instructions     : {instruction_memory.instruction_dir}")
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
