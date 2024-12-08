import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import random
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def remove_outliers_iqr(df):
    for column in df.select_dtypes(include=['float64', 'int64']):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

time event =
ratio = pd.read_csv("E:/Pycharm_Project/pythonProject/Datathon/1st-demo-blood-volume.csv")
char = pd.read_csv('E:/Pycharm_Project/pythonProject/Datathon/preop_char.csv')
char = char.groupby('anon_case_no', group_keys=False).apply(lambda x: x.sample(1))
char = pd.merge(ratio, char, on="anon_case_no", how="inner")
# preprocessing
features_keep = ['anon_case_no', 'gender','race','sum_fluid_vol']
categorical_features  = ['gender','race']
numerical_features = ['sum_fluid_vol']
char = char[features_keep]

ratio_cleaned = ratio.dropna()
ratio_cleaned = ratio_cleaned[['anon_case_no','sum_ffp_volume']]
## missing data
missing_ratio = char.isnull().mean()
columns_to_keep = missing_ratio[missing_ratio == 0].index
char_cleaned = char[columns_to_keep]

## outliers
char_cleaned = remove_outliers_iqr(char_cleaned)

## one-hot encoded
char_encoded = pd.get_dummies(char_cleaned, columns=categorical_features, drop_first=True)
bool_columns = char_encoded.select_dtypes(include=["bool"]).columns
char_encoded[bool_columns] = char_encoded[bool_columns].astype(int)

## standardization
scaler = StandardScaler()
scaled_features = scaler.fit_transform(char_encoded[numerical_features])
char_standardized = char_encoded.copy()
char_standardized[numerical_features] = scaled_features

## feature selection
final  = pd.merge(ratio_cleaned, char_standardized, on="anon_case_no", how="inner")

print(final.info())
print(final.describe())
# distribution visualization
columns_to_plot = final.columns[1:]
for col in columns_to_plot:
    plt.figure(figsize=(6, 4))
    plt.hist(final[col], bins=10, edgecolor="black", alpha=0.7)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

# Training
X = final.drop(columns=["anon_case_no", "time", "event"])
survival_data = final[["time", "event"]].join(X)

cox_model = CoxPHFitter()
cox_model.fit(survival_data, duration_col="time", event_col="event")


print(cox_model.summary)

cox_model.plot()