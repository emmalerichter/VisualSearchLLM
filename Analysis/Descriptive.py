import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

humans_general = pd.read_csv("humanResults/e1_numbers_processed.csv")
humans = humans_general[humans_general['target_color'] == 'red']
humans = humans[humans['shape_type'] == 2.0]
# HUMAN VALUES 
print(humans.describe())
print(humans['accuracy'].mean())
print(humans.groupby('colour_type')['accuracy'].mean())
print(humans.groupby(['colour_type', 'distractor_bin'])['accuracy'].mean())
