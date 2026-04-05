import pandas as pd

# import all 
imagedata = pd.read_csv("veo_results/submission_plan.csv")
Marker1data = pd.read_csv("veo_results/coding_ER_01_2026-03-31.csv", sep=';')
Marker1data['err_movement'] = Marker1data[['err_movement', 'err_other']].sum(axis=1)
Marker1data = Marker1data.drop(columns=['err_other'])
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
# edit fidelity scores as of true grouping 
## drop old
mergedVeo3 = mergedVeo3.drop(columns=[
    'fidelity_score_total', 'fidelity_omission', 'fidelity_numeric', 
    'fidelity_temporal', 'fidelity_physical_vanishing'
], errors='ignore')
# --- Category mappings ---

missing_elements = {
'omission_error': ['err_no_distractors', 'err_no_target', 'err_wrong_manip'],  # No Distractors, No Targets, No Identification
'temporal_incongruence': ['err_id_change_before', 'err_id_change_after', 'err_vanishing']  # Before, After, Intermittent Disappearance
}

incorrect_elements = {
'manipulation_error': {
'size_style': ['err_enlarge_target', 'err_merging', 'err_enlarge_distractors', 'err_font_change'],  # Target Enlarges, Merging, Distractors Enlarge, Font Changes
'movement':   ['err_rotation', 'err_target_centre', 'err_movement']  # Numbers Rotate Upright, Target Moves To Centre/Corner, Other Movement
    },

'addition_error': {
'new_target':      ['err_new_target_id', 'err_new_target_unid'],  # Identified, Not Identified
'new_distractors': ['err_extra_distractors']
    }
}

# --- Missing Elements ---
mergedVeo3['fidelity_omission']  = mergedVeo3[missing_elements['omission_error']].sum(axis=1)
mergedVeo3['fidelity_temporal']  = mergedVeo3[missing_elements['temporal_incongruence']].sum(axis=1)
mergedVeo3['fidelity_missing']   = mergedVeo3[missing_elements['omission_error'] + missing_elements['temporal_incongruence']].sum(axis=1)

# --- Incorrect Elements ---
mergedVeo3['fidelity_size_style']   = mergedVeo3[incorrect_elements['manipulation_error']['size_style']].sum(axis=1)
mergedVeo3['fidelity_movement']     = mergedVeo3[incorrect_elements['manipulation_error']['movement']].sum(axis=1)
mergedVeo3['fidelity_manipulation'] = mergedVeo3['fidelity_size_style'] + mergedVeo3['fidelity_movement']
mergedVeo3['fidelity_addition']     = mergedVeo3[
incorrect_elements['addition_error']['new_target'] +
incorrect_elements['addition_error']['new_distractors']
].sum(axis=1)
mergedVeo3['fidelity_incorrect']    = mergedVeo3['fidelity_manipulation'] + mergedVeo3['fidelity_addition']

# --- Total ---
mergedVeo3['fidelity_score_total'] = mergedVeo3['fidelity_missing'] + mergedVeo3['fidelity_incorrect']
print(mergedVeo3[['fidelity_omission', 'fidelity_temporal', 'fidelity_missing',
'fidelity_size_style', 'fidelity_movement', 'fidelity_manipulation',
'fidelity_addition', 'fidelity_incorrect', 'fidelity_score_total']].describe())

mergedVeo3.to_csv("veo_results/mergedVeo3.csv", index=False)

print(mergedVeo3.shape)
print(mergedVeo3.head())
print(mergedVeo3.columns.tolist())
print(mergedVeo3['target'].value_counts(dropna=False))

Marker2data = pd.read_csv("veo_results/coding_DK_01_2026-04-02.csv", sep=",")
Marker2data = Marker2data.add_suffix('_DK')
Marker2data['err_movement_DK'] = Marker2data[['err_movement_DK', 'err_other_DK']].sum(axis=1)
Marker2data = Marker2data.drop(columns=[ 'err_other_DK'])

# look into differences
print(Marker2data['notes_DK'].value_counts(dropna=False))
print(Marker2data[Marker2data['notes_DK'].notna()][['video_name_DK', 'notes_DK']])
print(Marker1data[Marker1data['video_name'].isin([
    '2Among5Colour_Bin5_Sample4.mp4',
    '2Among5NoColour_Bin1_Sample2.mp4'
])][['video_name', 'fidelity_score_total', 'success']])
print(Marker2data[Marker2data['video_name_DK'].isin([
    '2Among5Colour_Bin5_Sample4.mp4',
    '2Among5NoColour_Bin1_Sample2.mp4'
])][['video_name_DK', 'fidelity_score_total_DK', 'success_DK']])


MarkerMerged = Marker1data.merge(
    Marker2data,
    left_on='video_name',
    right_on='video_name_DK',
    how='left'
)
MarkerMerged.to_csv("analysis/MarkerMerged.csv")

## Human data
HumanData = pd.read_csv("humanResults/e1_numbers_processed.csv")
bin_map = {
    "1–4":   1,
    "5–8":   2,
    "9–16":  3,
    "17–32": 4,
    "33–64": 5,
    "65–99": 6
}
condition_map = {
    'Conjunctive': '2Among5ConjRed',
    'Inefficient disjunctive': '2Among5NoColour',
    'Efficient disjunctive': '2Among5Colour'
}
HumanData['colour_type'] = HumanData['colour_type'].replace(condition_map)
HumanData["bin_group"] = HumanData["distractor_bin"].map(bin_map)
HumanData.to_csv("analysis/HumanData.csv", index=False)