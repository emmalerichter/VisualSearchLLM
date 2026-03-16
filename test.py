import os
import numpy as np
import pandas as pd
import time
from constructMessage import constructImage


conditions = ["2Among5Colour"]
base_dir = "results/Images"
samples_per_bin = 6
active_operations = []
generation_log = []
RANDOM_SEED = 36
LOG_PATH = "veo_results/generation_log.csv"

 
def save_log():
    """Save the current generation log to CSV, appending or creating."""
    os.makedirs("veo_results", exist_ok=True)
    log_df = pd.DataFrame(generation_log)
    log_df.to_csv(LOG_PATH, index=False)
 
 
def sample_img_bin():
    submission_counter = 0
    rng = np.random.default_rng(RANDOM_SEED)
    for condition in conditions:
        folder_path = os.path.join(base_dir, condition)
        csv_path = os.path.join(folder_path, "annotations.csv")
 
        if not os.path.exists(csv_path):
            print(f"Looking for: {os.path.abspath(csv_path)}")
            raise FileNotFoundError(f"Missing critical metadata: {csv_path}.")
 
        df = pd.read_csv(csv_path)
 
        for bin_id in range(1, 2):
            print(f"starting Bin {bin_id}...")
            bin_data = df[df['bin_group'] == bin_id]
            if bin_data.empty:
                raise RuntimeError(f"Bin {bin_id} in {condition} is empty.")
            
            unique_images = bin_data['filename'].unique()
            if len(unique_images) < samples_per_bin:
                raise ValueError(f"Insufficient data in {condition} Bin {bin_id}.")
 
            sampled_images = rng.choice(unique_images, size=samples_per_bin, replace=False)
 
            for idx, image_filename in enumerate(sampled_images):
               
                full_image_path = os.path.join(folder_path, image_filename)
                row = bin_data[(bin_data['filename'] == image_filename) & (bin_data['target'] == True)]
                if not os.path.exists(full_image_path):
                    raise FileNotFoundError(f"Image not found: {full_image_path}")
 
                prompt_key = "2Among5-prompt-Conj" if "2Among5Conjunctive" in condition else \
                             ("2Among5-prompt-Col" if "2Among5Colour" in condition else "5Among2-prompt-NoCol")
 
                input_image = constructImage(full_image_path)
                if input_image is None:
                    raise RuntimeError(f"Failed to process image: {full_image_path}")
 
                output_name = f"{condition}_Bin{bin_id}_Sample{idx+1}.mp4"
                submission_counter += 1
                print(f"Submitting ({submission_counter}): {output_name}")
 
                operation = print(f"image submitted {full_image_path}") 
                
 
                active_operations.append({
                    "op": operation,
                    "filename": output_name,
                    "condition": condition,
                    "bin": bin_id,
                    "done": False
                })
 
                generation_log.append({
                    "video_filename": output_name,
                    "condition": condition,
                    "bin": bin_id,
                    "sample_index": idx + 1,
                    "source_image": image_filename,
                    "prompt_key": prompt_key,
                    "color": row['color'],
                    "distractor_color": row.get('distractor_color'),
                    "status": "submitted"
                })
                save_log()
 
                time.sleep(15)
 
    print(f"Total submitted: {submission_counter}") 

    os.makedirs("veo_results", exist_ok=True)
    log_df = pd.DataFrame(generation_log)
    log_df.to_csv("veo_results/generation_log.csv", index=False)
    print(f"Total submitted: {submission_counter}")

 
 
if __name__ == "__main__":
    sample_img_bin()