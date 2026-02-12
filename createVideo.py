import os 
import pandas as pd
import time
from google import genai
from constructMessage import constructMessage
from constructMessage import constructImage

# Insert API Access!!
client = genai.Client(api_key = "DEFINE")

conditions = ["2Among5Colour", "2Among5NoColour", "2Among5Conjunctive"]
base_dir = "Images"
samples_per_bin = 6
active_operations = []

def sample_img_bin ():
    for condition in conditions: 
        folder_path = os.path.join(base_dir, condition)
        csv_path = os.path.join(folder_path, "annotations.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing critical metadata: {csv_path}. Data collection cannot proceed without annotations.")
        df = pd.read_csv(csv_path)

        for bin_id in range(1,7):
            bin_data = df[df['bin_group']== bin_id]
            if bin_data.empty:
                raise RuntimeError(f"Data imbalance detected: Bin {bin_id} in {condition} is empty. Expected {samples_per_bin} images.")
            if len(bin_data) < samples_per_bin:
                raise ValueError(f"Insufficient data in {condition} Bin {bin_id}: Found {len(bin_data)}, need {samples_per_bin}.")
            samples = bin_data.sample(n=min(len(bin_data), samples_per_bin))
            
            for idx,(_,row) in enumerate(samples.iterrows()):
                image_filename = row['filename']
                full_image_path = os.path.join(folder_path, image_filename)
                if not os.path.exists(full_image_path):
                    raise FileNotFoundError(f"Image file referenced in CSV does not exist: {full_image_path}")
                #prompt selection
                prompt_key = "2Among5-prompt-Conj" if "2Among5Conjunctive" in condition else \
                ("2Among5-prompt-Col" if "2Among5Colour" in condition else "5Among2-prompt-NoCol")
                
                prompt_text = constructMessage(
                        writing=prompt_key,
                        colour=row['color'],
                        distractor_color=row.get('distractor_color'))
                
                input_image = constructImage(full_image_path)
                if input_image is None:
                    raise RuntimeError(f"Failed to process image object for: {full_image_path}")
                
                output_name = f"{condition}_Bin{bin_id}_Sample{idx+1}.mp4"
                print(f"Submitting: {output_name}")
                
                operation = client.models.generate_videos(
                    model="veo-3.1-generate-preview",
                    prompt=prompt_text,
                    image=input_image,  # in batches put in above stratified images with respective prompt
                    config=types.GenerateVideosConfig(
                        number_of_videos=1,
                        durationSeconds=4
                    )
                )
                active_operations.append({
                    "op": operation,
                    "filename": output_name,
                    "condition": condition,
                    "bin": bin_id,
                    "done": False
                })

# saved along structure in the directory that specefies under veo_results the condition and the bin_X; herein the filename is defined as condtion_Bin_'_SampleID.mp4
def monitor_and_safe():
    print(f"\nMonitoring {len(active_operations)} total video tasks...")
    completed_count = 0
    total = len(active_operations)

    while completed_count < total:
        for task in active_operations:
            if task.get("done"):
                continue
            
            op = client.models.get_operation(task["op"].name)
            
            if op.done:
                if op.error:
                    raise RuntimeError(
                        f"API Error during generation of {task['filename']}: "
                        f"Code {op.error.code} - {op.error.message}. "
                        "This may be due to safety filters or quota limits."
                    )

                if not op.response or not op.response.generated_videos:
                    raise ValueError(f"API returned a successful 'done' status for {task['filename']} but no video data was found.")

                video = op.response.generated_videos[0]
                
                save_dir = os.path.join("veo_results", task["condition"], f"Bin_{task['bin']}") 
                try:
                    os.makedirs(save_dir, exist_ok=True)
                except Exception as e:
                    raise OSError(f"Failed to create directory {save_dir}: {e}")

                save_path = os.path.join(save_dir, task["filename"])
                
                try:
                    video.video.save(save_path)
                    print(f"Successfully saved: {save_path}")
                except Exception as e:
                    raise IOError(f"Failed to save video to {save_path}: {e}")
                
                task["done"] = True
                completed_count += 1
        
        if completed_count < total:
            print(f"Status: {completed_count}/{total} done. Checking again in 30s...")
            time.sleep(30)

if __name__== "__main__":
    sample_img_bin()
    monitor_and_safe()