import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from huggingface_hub.errors import HFValidationError
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# --- Configuration ---
OUTPUT_DIR = "stats_outputs"

def parse_instruction_to_target(instruction_text):
    """
    Parses instruction text to extract target counts for beans.
    Logic imported/adapted from vse.py
    """
    targets = {"soybeans": 0, "red_beans": 0}
    if not isinstance(instruction_text, str):
        return targets

    text = instruction_text.lower()
    
    # Patterns to match "N scoops of bean"
    # Handling variations like "beans", "bean", "red beans", "redbeans"
    patterns = [
        (r"(\d+)\s+scoops?\s+of\s+soybeans?", "soybeans"),
        (r"(\d+)\s+scoops?\s+of\s+red\s?beans?", "red_beans"),
    ]

    for pattern, key in patterns:
        matches = re.findall(pattern, text)
        for count in matches:
            targets[key] += int(count)
            
    return targets

def plot_distributions(df, output_dir):
    """
    Generates visualizations for the dataset statistics.
    """
    sns.set_theme(style="whitegrid", font_scale=1.2)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Total Scoops (N) Distribution
    plt.figure(figsize=(10, 6))
    ax1 = sns.countplot(data=df, x="total_scoops", palette="viridis")
    plt.title("Distribution of Total Scoops (N)")
    plt.xlabel("Total Scoops (N)")
    plt.ylabel("Number of Episodes")
    for i in ax1.containers:
        ax1.bar_label(i)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dataset_stats_total_n.png"), dpi=300)
    plt.close()

    # 2. Task Type Distribution
    plt.figure(figsize=(8, 6))
    task_type_counts = df["task_type"].value_counts()
    plt.pie(task_type_counts, labels=task_type_counts.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
    plt.title("Task Type Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dataset_stats_task_type.png"), dpi=300)
    plt.close()

    # 3. Detailed Distribution (Soy vs Red) - Heatmap
    # Create a pivot table for the heatmap
    heatmap_data = df.pivot_table(index="red_beans", columns="soybeans", aggfunc="size", fill_value=0)
    # Sort index descending to have (0,0) at bottom-left if desired, or standard matrix view
    # Usually heatmaps have (0,0) at top-left. Let's make it intuitive: Y=Red, X=Soy. 
    # Invert Y axis for plot? sns.heatmap plots matrix as is.
    # Let's sort index ascending (0 at top) is default matrix.
    # To have 0 at bottom, we can invert index order or use backend.
    # We'll just stick to standard matrix for now but label axes clearly.
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label': 'Number of Episodes'})
    plt.title("Episode Distribution by Bean Counts")
    plt.xlabel("Soybean Scoops")
    plt.ylabel("Red Bean Scoops")
    plt.gca().invert_yaxis() # Put 0 at the bottom
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dataset_stats_detailed.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze LeRobot dataset statistics with detailed task parsing")
    parser.add_argument("dataset_name", type=str, help="Dataset name (e.g., 0xNOY/so101-with-narration)")
    parser.add_argument(
        "revision", type=str, nargs="?", default="main", help="Dataset revision (default: main)"
    )
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_name} (revision: {args.revision})...")
    try:
        dataset = LeRobotDataset(args.dataset_name, revision=args.revision)
    except HFValidationError:
        dataset_root = Path(args.dataset_name)
        args.dataset_name = f"{dataset_root.parent.name}/{dataset_root.name}"
        dataset = LeRobotDataset(args.dataset_name, revision=args.revision, root=dataset_root)

    print("\n=== Dataset Overview ===")
    print(f"Total Episodes: {dataset.num_episodes}")

    # Analyze Tasks
    print("\nAnalyzing Tasks & Narrations...")
    
    records = []
    
    # Pre-fetch tasks/instructions efficiently
    all_instructions = []
    
    # Try metadata first
    if "instruction" in dataset.meta.episodes:
        all_instructions = dataset.meta.episodes["instruction"]
    elif "task" in dataset.meta.episodes:
        all_instructions = dataset.meta.episodes["task"]
    
    # If not in metadata, scan frames (slow fallback)
    if len(all_instructions) != dataset.num_episodes:
        print("Instruction/Task not in metadata, scanning episodes (this might be slow)...")
        all_instructions = []
        for ep_idx in tqdm(range(dataset.num_episodes)):
            from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
            # Try efficient access
            val = "unknown"
            try:
                frame_data = None
                if hasattr(dataset, "hf_dataset"):
                    # Use index to get item
                    frame_data = dataset.hf_dataset[from_idx]
                else:
                    frame_data = dataset[from_idx]

                # 1. Try string columns directly
                if "instruction" in frame_data:
                    val = frame_data["instruction"]
                elif "task" in frame_data:
                    val = frame_data["task"]
                elif "language_instruction" in frame_data:
                    val = frame_data["language_instruction"]
                
                # 2. Try task_index mapping
                elif "task_index" in frame_data:
                    tid = frame_data["task_index"]
                    # If it's a tensor/list, get scalar
                    if hasattr(tid, "item"):
                        tid = tid.item()
                    elif isinstance(tid, list):
                        tid = tid[0]
                    
                    # Map via dataset.meta.tasks
                    if hasattr(dataset.meta, "tasks"):
                        tasks_df = dataset.meta.tasks
                        # tasks_df usually has index as task string and a column 'task_index'
                        # We want to find index where col 'task_index' == tid
                        matched = tasks_df[tasks_df["task_index"] == tid]
                        if not matched.empty:
                            val = matched.index[0]
                        else:
                            val = f"unknown_task_index_{tid}"
                    else:
                        val = f"task_index_{tid}"
                        
            except Exception as e:
                # print(f"Error extracting task: {e}")
                pass
            
            all_instructions.append(str(val))

    # Process each episode
    print("\n[DEBUG] Sample Instructions found:")
    unique_instructions = sorted(list(set(all_instructions)))
    for instr in unique_instructions[:20]:
        print(f"  - {repr(instr)}")
    print(f"  ... (Total unique: {len(unique_instructions)})")

    for i, instruction in enumerate(all_instructions):
        targets = parse_instruction_to_target(instruction)
        n_soy = targets["soybeans"]
        n_red = targets["red_beans"]
        total = n_soy + n_red
        
        # Determine Task Type
        if n_soy > 0 and n_red > 0:
            t_type = "Mixed"
        elif n_soy > 0:
            t_type = "Single (Soy)"
        elif n_red > 0:
            t_type = "Single (Red)"
        else:
            t_type = "Other/Zero"
            
        records.append({
            "episode_index": i,
            "instruction": instruction,
            "soybeans": n_soy,
            "red_beans": n_red,
            "total_scoops": total,
            "task_type": t_type
        })
        
    df = pd.DataFrame(records)
    
    # --- Statistics Output ---
    print(f"\n=== Analysis Results ({len(df)} episodes) ===")
    
    print("\n[Total Scoops Distribution]")
    print(df["total_scoops"].value_counts().sort_index().to_string())
    
    print("\n[Task Type Distribution]")
    print(df["task_type"].value_counts().to_string())
    
    print("\n[Detailed Breakdown (Soy, Red)]")
    breakdown = df.groupby(["soybeans", "red_beans"]).size().reset_index(name="count")
    print(breakdown.to_string(index=False))

    # --- Visualization ---
    if len(df) > 0:
        print(f"\nGenerating plots in '{OUTPUT_DIR}'...")
        try:
            plot_distributions(df, OUTPUT_DIR)
            print("Plots generated successfully:")
            print(f"- {OUTPUT_DIR}/dataset_stats_total_n.png")
            print(f"- {OUTPUT_DIR}/dataset_stats_task_type.png")
            print(f"- {OUTPUT_DIR}/dataset_stats_detailed.png")
        except Exception as e:
            print(f"Failed to generate plots: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
