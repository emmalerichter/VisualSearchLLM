import pandas as pd

# import all 
imagedata = pd.read_csv("veo_results/submission_plan.csv")
Marker1data = pd.read_csv("veo_results/coding_ER_01_2026-03-31.csv")
# Marter2data = pd.read_csv("veo_results/2Among5ConjRed_coded/xyz.csv")
annotationsColour = pd.read_csv("results/Images/2Among5Colour/annotations.csv")
annotationsNocolour = pd.read_csv("results/Images/2Among5NoColour/annotations.csv")
annotationsConjRed = pd.read_csv("results/Images/2Among5ConjRed/annotations.csv")
annotationsNoDistractor = pd.read_csv("results/Images/NoDistractors/annotations.csv")
# merging in single csv 
annotations = pd.concat([
    annotationsColour.assign(condition='2Among5Colour'),
    annotationsNocolour.assign(condition='2Among5NoColour'),
    annotationsConjRed.assign(condition='2Among5ConjRed'),
    annotationsNoDistractor.assign(condition='NoDistractors')
], ignore_index=True)

MergedAnnotations = annotations[annotations['target'] == True]
# find shared collumn 
imagedata['image_condition'] = imagedata['source_image'] + '_' + imagedata['condition']
MergedAnnotations['image_condition'] = MergedAnnotations['filename'] + '_' + MergedAnnotations['condition']
MergedAnnotations.to_csv("results/mergedAnnotations.csv", index=False)

AnnotatedSubmissions = imagedata.merge(
    MergedAnnotations,
    on='image_condition',
    how='right'
)
AnnotatedSubmissions = AnnotatedSubmissions.dropna(subset=['video_filename'])
AnnotatedSubmissions.to_csv("veo_results/annotatedSubmissions.csv", index=False)

mergedVeo3  = AnnotatedSubmissions.merge(
        Marker1data,
        left_on='video_filename',
        right_on='video_name',
        how='left'
    )

mergedVeo3 = mergedVeo3.drop(columns=['video_name', 'filename', 'condition_y'])
mergedVeo3 = mergedVeo3.rename(columns={'condition_x': 'condition'})
mergedVeo3 = mergedVeo3[mergedVeo3['target'] == True]
mergedVeo3.to_csv("veo_results/mergedVeo3.csv", index=False)

print(mergedVeo3.shape)
print(mergedVeo3.head())
print(mergedVeo3['target'].value_counts(dropna=False))