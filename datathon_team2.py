import numpy as np
import pandas as pd
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
        if column == 'death_in_30_days':
            continue
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

def plasma_trans(x):
    if 0 < x <= 0.5:
        return 1
    elif 0.5 < x <= 1:
        return 2
    elif 1 < x :
        return 2
    else:
        return 0

def platelet_trans(x):
    if 0 < x <= 1:
        return 1
    elif 1 < x <= 2:
        return 2
    elif 2 < x :
        return 2
    else:
        return 0

'''
blood_loss = pd.read_csv('E:/Pycharm_Project/pythonProject/Datathon/blood_loss.csv')
blood_loss = blood_loss.groupby('session_id', as_index=False)['blood_loss'].sum()
ratio = pd.read_csv("E:/Pycharm_Project/pythonProject/Datathon/1st-demo-blood-volume.csv")
ratio = ratio.fillna(0)
ratio['plasma/rbc'] =ratio['sum_ffp_volume']/ratio['sum_prc_volumes']
ratio['platelet/rbc'] =ratio['sum_platelet_volume']/ratio['sum_prc_volumes']
ratio['plasma/rbc'] = ratio['plasma/rbc'].apply(plasma_trans)
ratio['platelet/rbc'] = ratio['platelet/rbc'].apply(platelet_trans)

char = pd.read_csv('E:/Pycharm_Project/pythonProject/Datathon/pre_op.char.csv')
char = char.groupby('session_id', as_index=False).apply(lambda x: x.sample(1)).reset_index(drop=True)
intra = pd.read_csv('E:/Pycharm_Project/pythonProject/Datathon/intra_op_cleaned.csv')
transxenamic = pd.read_csv('E:/Pycharm_Project/pythonProject/Datathon/transxenamic.csv')
transxenamic = transxenamic.loc[transxenamic.groupby('session_id')['transxenamic'].idxmax()].reset_index(drop=True)
score = pd.read_csv('E:/Pycharm_Project/pythonProject/Datathon/pre_op.risk_index.csv')
score = score[['session_id','asa_class','cardiac_risk_index','osa_risk_index']]
lab = pd.read_csv('E:/Pycharm_Project/pythonProject/Datathon/demographic_with_lab_test.csv').drop(columns=['anon_case_no'])

data = pd.merge(ratio, blood_loss, on="session_id", how="left")
data = pd.merge(data, char, on="session_id", how="left")
data = pd.merge(data, intra, on="session_id", how="left")
data = pd.merge(data, transxenamic, on="session_id", how="left")
data = pd.merge(data, score, on="session_id", how="left")
data = pd.merge(data, lab, on="session_id", how="left")
'''

data = pd.read_csv('E:/Pycharm_Project/pythonProject/Datathon/final4.csv')
data_codes = pd.read_csv('E:/Pycharm_Project/pythonProject/Datathon/TOSP_final_210428.csv')
df_mapped = pd.merge(data, data_codes, left_on="procedure_code", right_on="New_TOSP_codes", how="left")
data = df_mapped

# preprocessing
features_keep = ['session_id','death_in_30_days',
                 'plasma/rbc',
                 'platelet/rbc',
                 'blood_loss',
                 'age_time_of_surgery',
                 'gender_x',
                 'race_x',
                 'smoking_history_x',
                 'any_steroids_past_6_months_x',
                 'any_aspirin_warfarin_anti_platelet_past_2_weeks',
                 'alcohol_consumption_x',
                 'bmi_x',
                 'Op_risk_rank',
                 'session_duration',
                 'urgency',
                 'surgeon_grade',
                 'transxenamic',
                 'asa_class',
                 'h_o_ihd',
                 'h_o_chf',
                 'h_o_cva',
                 'dm_on_insulin',
                 'hemoglobin_test_value',
                 'platelet_test_value',
                 'aptt_test_value',
                 'pt_test_value']

data = data[features_keep]
categorical_features  = ['plasma/rbc', 'platelet/rbc','gender_x','race_x','smoking_history_x',
                         'any_steroids_past_6_months_x','any_aspirin_warfarin_anti_platelet_past_2_weeks',
                         'alcohol_consumption_x','Op_risk_rank','urgency','surgeon_grade','transxenamic',
                         'asa_class', 'h_o_ihd', 'h_o_chf',  'h_o_cva',  'dm_on_insulin']
numerical_features = list(set(features_keep) - set(categorical_features))
numerical_features.remove('session_id')
numerical_features.remove('death_in_30_days')
# preprocessing blood loss

## missing data

columns_to_fill = ['h_o_ihd', 'h_o_chf', 'h_o_cva', 'dm_on_insulin']
data[columns_to_fill] = data[columns_to_fill].fillna("No")


mode_value = data['Op_risk_rank'].mode()[0]
data['Op_risk_rank'] = data['Op_risk_rank'].fillna(mode_value)

mode_value = data['asa_class'].mode()[0]
data['asa_class'] = data['asa_class'].fillna(mode_value)

columns_to_fill = ['hemoglobin_test_value', 'platelet_test_value', 'aptt_test_value', 'pt_test_value']
for column in columns_to_fill:
    median_value = data[column].median()
    data[column] = data[column].fillna(median_value)
'''
missing_ratio = data.isnull().mean()
columns_to_keep = missing_ratio[missing_ratio == 0].index
char_cleaned = data[columns_to_keep]
'''
char_cleaned = data.dropna()
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
final = char_standardized

print(final.info())
print(final.describe())
# distribution visualization
'''
columns_to_plot = final.columns[1:]
for col in columns_to_plot:
    plt.figure(figsize=(6, 4))
    plt.hist(final[col], bins=10, edgecolor="black", alpha=0.7)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
'''
# Training
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

X = final.drop(columns=["session_id","death_in_30_days"])
y = final["death_in_30_days"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "eta": 0.1,
    "seed": 42,
    'scale_pos_weight': scale_pos_weight
}

model = xgb.train(params, dtrain, num_boost_round=100)

y_pred_prob = model.predict(dtest)

y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

feature_importance = model.get_score(importance_type='weight')  # 'weight', 'gain', 'cover'
print(feature_importance)

xgb.plot_importance(model, importance_type='weight')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
