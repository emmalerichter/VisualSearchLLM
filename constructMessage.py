import PIL.Image

colourMap = {
    "#FF0000": "red", 
    "#00FF00": "green", 
    "#0000FF": "blue", 
    "#000000": "black",
    "#FFFFFF": "white"
}

def constructMessage(writing, colour, distractor_color=None):
    # Map Hex colors to English names
    target_color_name = colourMap.get(colour, "colored")
    dist_color_name = colourMap.get(distractor_color, "colored") if distractor_color else "various colored"

    # define prompt
    prompts = { 
        "2Among5-prompt-Col": (
            f"The camera is fixed and completely stationary. The {dist_color_name} digits 5 remain static and unchanged. The single {target_color_name} digit 2 becomes the sole focus; starting at one second, a circle appears around it.No camera movement, no zooming, no panning. The digits do not move or change position."
        ),         
        "5Among2-prompt-NoCol": (
            f"The camera is fixed and completely stationary. The {dist_color_name} digits 2 remain static and unchanged. The single {target_color_name} digit 5 becomes the sole focus; starting at one second, a circle appears around it.No camera movement, no zooming, no panning. The digits do not move or change position."
        ),         
        "2Among5-prompt-Conj": (
            f"The camera is fixed and completely stationary. The mixed {target_color_name} and {dist_color_name} digits remain static. The single {target_color_name} digit 2 becomes the sole focus; starting at one second, a circle appears around it. No camera movement, no zooming, no panning."
        ),         
    }
    return prompts[writing]


def constructImage(full_image_path):
    # filepath to first frame
    try:
        img = PIL.Image.open(full_image_path)
        img = img.convert("RGB")
        img.load()
        return img
    except FileNotFoundError:
        print(f"Error: The file at {full_image_path} was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the image: {e}")
        return None