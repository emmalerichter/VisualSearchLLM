
# VisualSearchLLM

This code is for investigating the pop-out effect in Visual AI systems. And was adapted to specifically probe generative video models as a dissertation project that focussed on spatial configuration search. 

To run this code from scratch to replicate our results the following should be done:

## Install requirements

- Create a python environment `python -m venv visualSearch`
- Activate the environment (shown for Windows) `.\visualSearch\Scripts\Activate.ps1`
- Install the requirements `pip install -r requirements.txt`

## Generate Images

The dataset of images with varying visual processing demands. Image ratio is fitted as a 9:16 ratio 
The datasets can be created using the generateImages.py file. 
The intended use is `python generateImages.py -n x -d x -p x`
where `-n` is the flag for the number of images to create, `-d` is the directory to store the images, and `-p` is a selection from preset options configuring the images created. More control over the images generated is possible using other flags.

The presented results only consider the 2Among5ColourRand, 2Among5NoColourRand and 2Among5ConjRand conditions. 


| Experiment Type     | Presets to Use                          |
|---------------------|------------------------------------------|
| 2Among5       | `2Among5ColourRand`<br>`2Among5NoColourRand`<br>`2Among5ConjRand`<br> `5Among2ColourRand`<br>`5Among2NoColourRand`<br>`5Among2ConjRand` <br> `NoDisctractors` |
| Light Priors  | `LitSpheresTop`<br>`LitSpheresBottom`<br>`LitSpheresLeft`<br>`LitSpheresRight` |
| Circle Sizes  | `CircleSizesSmall`<br>`CircleSizesMedium`<br>`CircleSizesLarge` |
| Ctrl Condition | `2TargetOnly`


## API requests

API requests can be handeled in the `constructVideo.py`. The present experimetn specified use of Gemini's Veo 3.1 fast, as well as a duration of videos of 4 seconds. 
Our results were created with the prompt preset defined in `constructMessage.py`

# Video Processing
Videos are marked by hand, using the following website's tool: https://emmalerichter.github.io/dissertation/video-coder.html
To turn the results into a more readable format first processing was performed in `ProcessingModel.py` this includes merging of the image, video and marker data.

## Restults Analysis 
Results are analyised in R, along the three hypothesis. Files associated with data analysis are 
| Datatype   | file to use                          |
|---------------------|------------------------------------------|
| Ambiguity coded as failure     | `HypothesisTesting.ipynb` |
| Ambiguity coded as success  | `HypothesisTestingAS.ipynb` |
