import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


humans_general = pd.read_csv("humanResults/e1_numbers_processed.csv")
# model = pd.read_csv("XYZ.csv")
condition_map = {
    'Conjunctive': '2Among5ConjRed',
    'Inefficient disjunctive': '2Among5NoColour',
    'Efficient disjunctive': '2Among5Colour'
}
bin_map = {
    '1–4': 1,
    '5–8': 2,
    '9–16': 3,
    '17–32': 4,
    '33–64': 5,
    '65–99': 6,
}
colours = {
    '2Among5Colour': "#9B69FF",
    '2Among5ConjRed': "#FFED61",
    '2Among5NoColour': "#F84AC1",
    'NoDistractors': "#51D772"
}

humans_general['distractor_bin'] = humans_general['distractor_bin'].map(bin_map)
humans_general['colour_type'] = humans_general['colour_type'].replace(condition_map)

humans = humans_general[humans_general['target_color'] == 'red']
humans = humans[humans['shape_type'] == 2.0]
mean_humans = humans.groupby(['colour_type', 'distractor_bin'])['accuracy'].mean().reset_index()
print(humans['colour_type'].unique())

for condition, color in colours.items():
    data = mean_humans[mean_humans['colour_type'] == condition]
    plt.plot(data['distractor_bin'], data['accuracy'], color=color, marker='o', ms=3, label=condition)# mean_model = model.groupby('distractor_bin')['accuracy'].mean()
# plt.plot(model['distractor_bin'], model['accuracy'], color = "#FF6961", marker='o', ms=3)
plt.xlabel('Number of Distractors')
plt.ylabel('Accuracy')
plt.title('Performance by Distractor Count & Condition')
plt.legend()
plt.show()
