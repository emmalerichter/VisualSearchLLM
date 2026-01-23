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
                print(f"Skipping {condition}: CSV not found.")
                continue
        df = pd.read_csv(csv_path)

        for bin_id in range(1,7):
            bin_data = df[df['bin_group']== bin_id]
            if bin_data.empty:
                continue
            samples = bin_data.sample(n=min(len(bin_data), samples_per_bin))
            
            for idx,(_,row) in enumerate(samples.iterrows()):
                image_filename = row['filename']
                full_image_path = os.path.join(folder_path, image_filename)
                #prompt selection
                prompt_key = "2Among5-prompt-Conj" if "2Among5Conjunctive" in condition else \
                ("2Among5-prompt-Col" if "2Among5Colour" in condition else "5Among2-prompt-NoCol")
                
                prompt_text = constructMessage(
                        writing=prompt_key,
                        colour=row['color'],
                        distractor_color=row.get('distractor_color'))
                
                input_image = constructImage(full_image_path) # ifx in constructMessage.py
                
                output_name = f"{condition}_Bin{bin_id}_Sample{idx+1}.mp4"
                print(f"Submitting: {output_name}")
                
                operation = client.models.generate_videos(
                    model="veo-3.1-generate-preview",
                    prompt=prompt_text,
                    image = input_image # in batches put in above stratified images with respective prompt
                    )
                active_operations.append({
                    "op": operation,
                    "filename": output_name,
                    "condition": condition,
                    "bin": bin_id
                })

def monitor_and_safe():
    completed_count = 0
    total = len(active_operations)

    while completed_count < total:
        for task in active_operations:
            if task.get("done"):
                continue
            
            op = client.models.get_operation(task["op"].name)
            
            if op.done:
                video = op.response.generated_videos[0]
                
                # Save logic
                save_dir = os.path.join("veo_results", task["condition"], f"Bin_{task['bin']}")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, task["filename"])
                
                video.video.save(save_path)
                print(f"Successfully saved: {save_path}")
                
                task["done"] = True
                completed_count += 1
        
        if completed_count < total:
            print(f"Status: {completed_count}/{total} done. Checking again in 30s...")
            time.sleep(30)

if __name__== "__main__":
    sample_img_bin()
    monitor_and_safe()