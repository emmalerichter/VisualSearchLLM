from google import genai
import time
import os

# 1. Initialize with v1beta
client = genai.Client(api_key="AIzaSyAggBZ1yix22DaHAc-sf4GXrEMjXtzfHWA", http_options={'api_version': 'v1beta'})

def main():
    model_id = "veo-3.1-generate-preview"
    
    print(f":rocket: Launching 4s generation on {model_id}...")
    
    try:
        # 2. Request the video
        operation = client.models.generate_videos(
            model=model_id,
            prompt="A macro shot of a calming forest scene, 4k.",
            config={'duration_seconds': 4}
        )

        print(f"Operation started! (ID: {operation.name})")
        
        # 3. Correct Polling Logic
        while not operation.done:
            print("Rendering... (waiting 15s)")
            time.sleep(15)
            # Pass the ENTIRE operation object to refresh it
            operation = client.operations.get(operation)

        # 4. Extract and Download results
        if operation.result:
            video_info = operation.result.generated_videos[0]
            print(f"\n:sparkles: Generation Complete! Video URI: {video_info.video.uri}")
            print(":inbox_tray: Downloading to your local folder...")
            
            # This fetches the raw video bytes from Google's servers
            client.files.download(file=video_info.video)
            
            # Save it to your current working directory
            output_filename = "output.mp4"
            
            # The SDK handles the file write for you
            if hasattr(video_info.video, 'save'):
                video_info.video.save(output_filename)
            else:
                # Fallback just in case you are on a slightly older minor version
                with open(output_filename, "wb") as f:
                    f.write(video_info.video.video_bytes)
            
            filepath = os.path.abspath(output_filename)
            print(f":white_check_mark: Success! Your video is waiting for you at:\n{filepath}")

        else:
            print("\nGeneration finished but no video found. Check AI Studio for safety blocks.")

    except Exception as e:
        print(f"\n:x: Error: {e}")

if __name__ == "__main__":
    main()