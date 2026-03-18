import os
import numpy as np
import pandas as pd
import time
from google import genai
from google.genai import types
from constructMessage import constructMessage
from constructMessage import constructImage
from APIAccess import api_key_hide

client = genai.Client(api_key=api_key_hide, http_options={'api_version': 'v1beta'})

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
            raise FileNotFoundError(f"Missing critical metadata: {csv_path}.")
 
        df = pd.read_csv(csv_path)
 
        for bin_id in range(1, 2):
            bin_data = df[df['bin_group'] == bin_id]
            if bin_data.empty:
                raise RuntimeError(f"Bin {bin_id} in {condition} is empty.")
            
            unique_images = bin_data['filename'].nunique()
            if len(unique_images) < samples_per_bin:
                raise ValueError(f"Insufficient data in {condition} Bin {bin_id}.")
 
            sampled_images = rng.choice(unique_images, size=samples_per_bin, replace=False)
 
            for idx, image_filename in enumerate(sampled_images):
               
                full_image_path = os.path.join(folder_path, image_filename)
                row = bin_data[(bin_data['filename'] == image_filename) & (bin_data['target'] == True)]
                if not os.path.exists(full_image_path):
                    raise FileNotFoundError(f"Image not found: {full_image_path}")
 
                prompt_key = "2Among5-prompt-Conj" if "2Among5Conjunctive" in condition else \
                             ("2Among5-prompt-Col" if "2Among5Colour" in condition else \
                              ("NoDistractors-prompt" if "NoDistractors" in condition else "5Among2-prompt-NoCol"))
 
                prompt_text = constructMessage(
                    writing=prompt_key,
                    colour=row['color'],
                    distractor_color=row.get('distractor_color'))
 
                input_image = constructImage(full_image_path)
                if input_image is None:
                    raise RuntimeError(f"Failed to process image: {full_image_path}")
 
                output_name = f"{condition}_Bin{bin_id}_Sample{idx+1}.mp4"
                submission_counter += 1
                print(f"Submitting ({submission_counter}): {output_name}")
 
                operation = client.models.generate_videos(
                    model="veo-3.1-fast-generate-preview",
                    prompt=prompt_text,
                    image=input_image,
                    config={'numberOfVideos': 1, 'durationSeconds': 4}
                )
 
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
 
                time.sleep(35)
 
    print(f"Total submitted: {submission_counter}") 

    os.makedirs("veo_results", exist_ok=True)
    log_df = pd.DataFrame(generation_log)
    log_df.to_csv("veo_results/generation_log.csv", index=False)
    print(f"Total submitted: {submission_counter}")


def monitor_and_save():
    print(f"\nMonitoring {len(active_operations)} total video tasks...")
    completed_count = 0
    total = len(active_operations)
 
    while completed_count < total:
        for task in active_operations:
            if task.get("done"):
                continue
 
            # Refresh operation status
            task["op"] = client.operations.get(task["op"])
 
            if task["op"].done:
                # --- Handle error from API ---
                if task["op"].error and task["op"].error.code != 0:
                    print(f"ERROR for {task['filename']}: {task['op'].error.message}")
                    task["done"] = True
                    completed_count += 1
                    for entry in generation_log:
                        if entry["video_filename"] == task["filename"]:
                            entry["status"] = f"failed: {task['op'].error.message}"
                    save_log()
                    continue
 
                if not task["op"].result or not task["op"].result.generated_videos:
                    raise ValueError(f"No video returned for {task['filename']}.")
 
                video = task["op"].result.generated_videos[0]
 
                save_dir = os.path.join("veo_results", task["condition"], f"Bin_{task['bin']}")
                try:
                    os.makedirs(save_dir, exist_ok=True)
                except Exception as e:
                    raise OSError(f"Failed to create directory {save_dir}: {e}")
 
                save_path = os.path.join(save_dir, task["filename"])
                try:
                    client.files.download(file=video.video)
                    video.video.save(save_path)
                    print(f"✓ Saved ({completed_count+1}/{total}): {save_path}")
                except Exception as e:
                    raise IOError(f"Failed to save {save_path}: {e}")
 
                task["done"] = True
                completed_count += 1

                for entry in generation_log:
                    if entry["video_filename"] == task["filename"]:
                        entry["status"] = "saved"
                save_log()
 
        if completed_count < total:
            print(f"Status: {completed_count}/{total} done. Checking again in 30s...")
            time.sleep(35)
 
    print(f"\nAll {total} videos completed successfully.")
 
 
if __name__ == "__main__":
    sample_img_bin()
    monitor_and_save()
