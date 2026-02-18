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
            f"The {target_color_name} digit 2 becomes the sole focus; starting at one second, a black circle appears around it. Static camera perspective, no zoom no pan no movement no dolly no rotation."
        ),         
        "5Among2-prompt-NoCol": (
            f"The {target_color_name} digit 2 becomes the sole focus; starting at one second, a black circle appears around it. Static camera perspective, no zoom no pan no movement no dolly no rotation."
        ),         
        "2Among5-prompt-Conj": (
            f"The {target_color_name} digit 2 becomes the sole focus; starting at one second, a black circle appears around it. Static camera perspective, no zoom no pan no movement no dolly no rotation."
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