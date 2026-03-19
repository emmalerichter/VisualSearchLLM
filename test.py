import os
import numpy as np
import pandas as pd
import time
from constructMessage import constructImage


conditions = ["2Among5Colour", "2Among5ConjRed", "2Among5NoColour", "NoDistractors"]
base_dir = "results/Images"
samples_per_bin = 6
active_operations = []
generation_log = []
RANDOM_SEED = 36
LOG_PATH = "veo_results/generation_log.csv"
PLAN_PATH = "veo_results/submission_plan.csv"
DAILY_LIMIT = 10

def build_or_load_plan(conditions, base_dir, samples_per_bin):
    if os.path.exists(PLAN_PATH):
        return pd.read_csv(PLAN_PATH)
    
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for condition in conditions:
        folder_path = os.path.join(base_dir, condition)
        ann = pd.read_csv(os.path.join(folder_path, "annotations.csv"))
        
        bins_to_sample = [0] if "NoDistractor" in condition else range(1, 7)
        
        for bin_id in bins_to_sample:
            bin_data = ann[ann['bin_group'] == bin_id]
            unique_images = bin_data['filename'].unique()
            sampled = rng.choice(unique_images, size=samples_per_bin, replace=False)
            for idx, img in enumerate(sampled):
                rows.append({
                    "video_filename": f"{condition}_Bin{bin_id}_Sample{idx+1}.mp4",
                    "condition": condition,
                    "bin": bin_id,
                    "source_image": img,
                    "status": "pending"
                })
    plan = pd.DataFrame(rows)
    plan.to_csv(PLAN_PATH, index=False)
    return plan
 
def save_log():
    """Save the current generation log to CSV, appending or creating."""
    os.makedirs("veo_results", exist_ok=True)
    log_df = pd.DataFrame(generation_log)
    log_df.to_csv(LOG_PATH, index=False)
 
 
def sample_img_bin():
    plan = build_or_load_plan(conditions, base_dir, samples_per_bin)    
    pending = plan[plan["status"] == "pending"].head(DAILY_LIMIT)
    submission_counter = 0
    rng = np.random.default_rng(RANDOM_SEED)
    if pending.empty:
        print("Nothing pending — all done or daily limit already reached.")
        return

    for _, row in pending.iterrows():
        folder_path = os.path.join(base_dir, row['condition'])

        if not os.path.exists(os.path.join(folder_path, "annotations.csv")):
            raise FileNotFoundError(f"Missing critical metadata: {os.path.join(folder_path, 'annotations.csv')}")

        df = pd.read_csv(os.path.join(folder_path, "annotations.csv"))
        full_image_path = os.path.join(folder_path, row["source_image"])
        ann_row = df[(df['filename'] == row["source_image"]) & (df['target'] == True)]

        if ann_row.empty:
            raise RuntimeError(f"No target annotation found for {row['source_image']}")
        if not os.path.exists(full_image_path):
            raise FileNotFoundError(f"Image not found: {full_image_path}")
       
        print(f"starting Bin {row['bin']}...")

        prompt_key = "2Among5-prompt-Conj" if "2Among5Conjunctive" in row['condition'] else \
                             ("2Among5-prompt-Col" if "2Among5Colour" in row['condition'] else "5Among2-prompt-NoCol")
 
        input_image = constructImage(full_image_path)
        if input_image is None:
            raise RuntimeError(f"Failed to process image: {full_image_path}")
 
        output_name = row['video_filename']
        submission_counter += 1
        print(f"Submitting ({submission_counter}): {output_name}")
 
        operation = print(f"image submitted {full_image_path}") 
                
 
        active_operations.append({
            "op": operation,
            "filename": output_name,
            "condition": row["condition"],
            "bin": row["bin"],
            "done": False
        })
 
        generation_log.append({
            "video_filename": output_name,
            "condition": row["condition"],
            "bin": row["bin"],
            "sample_index": row.name,
            "source_image": row["source_image"],
            "prompt_key": prompt_key,
            "color": ann_row['color'].values[0],
            "distractor_color": row.get('distractor_color'),
            "status": "submitted"
            })
        save_log()
        plan.loc[plan["video_filename"] == output_name, "status"] = "submitted"
        plan.to_csv(PLAN_PATH, index=False)     

        time.sleep(5)
 
    print(f"Total submitted: {submission_counter}") 

    os.makedirs("veo_results", exist_ok=True)
    log_df = pd.DataFrame(generation_log)
    log_df.to_csv("veo_results/generation_log.csv", index=False)
    print(f"Total submitted: {submission_counter}")

 
 
if __name__ == "__main__":
    sample_img_bin()
