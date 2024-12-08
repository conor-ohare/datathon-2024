import pandas as pd

# Load intra_op.drug_fluids.csv
intra_op_df = pd.read_csv("intra_op.drug_fluids.csv")

# Calculate total_transfusion and additional intra-op features
intra_op_summary = (
    intra_op_df.groupby("anon_case_no", as_index=False)
    .agg(
        total_transfusion=("fluid_volume_actual", "sum"),
        num_fluid_events=("fluid_name", "count"),
        fluid_types=("fluid_name", lambda x: ", ".join(x.dropna().astype(str).unique()))
    )
)

# Load pre_op.char.csv
pre_op_df = pd.read_csv("pre_op.char.csv")

# Select relevant pre-op features
pre_op_relevant = pre_op_df[
    [
        "anon_case_no", "age_time_of_surgery", "gender", "race", 
        "bmi", "systolic_bp", "diastolic_bp", "heart_rate", 
        "o2_saturation", "temperature", "pain_score",
        "allergy_information", "presence_of_malignancy", "smoking_history", 
        "any_steroids_past_6_months", "any_aspirin_warfarin_anti_platelet_past_2_weeks"
    ]
]

# Merge the intra-op and pre-op dataframes on anon_case_no
final_table = pd.merge(intra_op_summary, pre_op_relevant, on="anon_case_no", how="left")

# Save the final table to a CSV file
final_table.to_csv("final_transfusion_comorbidities_extended.csv", index=False)

# Display the first few rows of the final table
print(final_table.head())