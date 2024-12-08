import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import shap
from econml.dml import CausalForestDML
import matplotlib.pyplot as plt
import seaborn as sns

class TransfusionCausalAnalysis:
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)
        self.cox_model = None
        self.causal_forest = None
        self.feature_importance = None
        
    def check_missing_values(self, df: pd.DataFrame, stage: str = ""):
        """Print missing value information for debugging."""
        missing = df.isnull().sum()
        if missing.any():
            print(f"\nMissing values {stage}:")
            print(missing[missing > 0])
            
    def preprocess_data(self):
        """Prepare data with robust missing value handling."""
        print("\nStarting data preprocessing...")
        df = self.data.copy()
        
        # Initial missing value check
        self.check_missing_values(df, "before preprocessing")
        
        # Handle categorical variables first
        print("\nProcessing categorical variables...")
        
        # 1. Gender
        df['gender'] = df['gender'].fillna('Unknown')
        df['is_male'] = (df['gender'] == 'MALE').astype(int)
        
        # 2. Race
        df['race'] = df['race'].fillna('Unknown')
        race_dummies = pd.get_dummies(df['race'], prefix='race')
        
        # 3. Binary variables - convert to numeric indicators
        binary_vars = {
            'smoking_history': 'is_smoker',
            'alcohol_consumption': 'drinks_alcohol',
            'any_steroids_past_6_months': 'uses_steroids',
            'curr_tcm_herbal_treatment': 'uses_tcm',
            'hypertension': 'has_hypertension'
        }
        
        for original, new in binary_vars.items():
            df[original] = df[original].fillna('Unknown')
            df[new] = (df[original] == 'Yes').astype(int)
        
        # 4. Handle presence_of_malignancy
        df['has_malignancy'] = df['presence_of_malignancy'].notna().astype(int)
        
        # Process numeric variables
        print("\nProcessing numeric variables...")
        numeric_cols = [
            'age_time_of_surgery', 'bmi', 'cardiac_risk_index', 
            'total_transfusion', 'APTT', 'HAEMOGLOBIN', 
            'PLATELET COUNT', 'PROTHROMBIN TIME'
        ]
        
        # Convert to numeric and handle missing values
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled {df[col].isnull().sum()} missing values in {col}")
        
        # Handle time difference and create event indicator
        print("\nProcessing time-to-event data...")
        
        # Ensure time_difference_hours is numeric
        df['time_difference_hours'] = pd.to_numeric(df['time_difference_hours'], errors='coerce')
        
        # Create event indicator (1 if we observe a next transfusion, 0 if censored)
        df['had_next_transfusion'] = df['time_difference_hours'].notna().astype(int)
        
        # For censored observations, set time to maximum observed time
        max_observed_time = df['time_difference_hours'].max()
        df['time_difference_hours'] = df['time_difference_hours'].fillna(max_observed_time)
        
        # Combine all features
        print("\nCreating final feature set...")
        feature_cols = [
            'is_male', 'is_smoker', 'drinks_alcohol', 'uses_steroids',
            'uses_tcm', 'has_malignancy', 'has_hypertension',
            'age_time_of_surgery', 'bmi', 'cardiac_risk_index', 'total_transfusion',
            'APTT', 'HAEMOGLOBIN', 'PLATELET COUNT', 'PROTHROMBIN TIME',
            'had_next_transfusion', 'time_difference_hours'
        ]
        
        # Create final processed dataset
        self.processed_data = pd.concat([
            df[['anon_case_no'] + feature_cols],
            race_dummies
        ], axis=1)
        
        # Standardize numeric features
        numeric_features = [
            'age_time_of_surgery', 'bmi', 'cardiac_risk_index', 'total_transfusion',
            'APTT', 'HAEMOGLOBIN', 'PLATELET COUNT', 'PROTHROMBIN TIME'
        ]
        
        # Standardize only if the column exists and has no missing values
        valid_numeric_features = [col for col in numeric_features 
                                if col in self.processed_data.columns 
                                and not self.processed_data[col].isnull().any()]
        
        if valid_numeric_features:
            scaler = StandardScaler()
            self.processed_data[valid_numeric_features] = scaler.fit_transform(
                self.processed_data[valid_numeric_features]
            )
        
        # Final missing value check
        self.check_missing_values(self.processed_data, "after preprocessing")
        
        # Remove any remaining rows with NaN values
        initial_rows = len(self.processed_data)
        self.processed_data = self.processed_data.dropna()
        final_rows = len(self.processed_data)
        
        print(f"\nPreprocessing complete:")
        print(f"Initial rows: {initial_rows}")
        print(f"Final rows: {final_rows}")
        print(f"Rows removed: {initial_rows - final_rows}")
        print("\nFeatures in final dataset:")
        print(self.processed_data.columns.tolist())

    def fit_cox_model(self):
        """Fit Cox Proportional Hazards model with error checking."""
        try:
            self.cox_model = CoxPHFitter()
            
            # Select features for Cox model (exclude case number)
            cox_features = [col for col in self.processed_data.columns 
                          if col not in ['anon_case_no']]
            
            # Double check for any remaining NaN values
            data_for_cox = self.processed_data[cox_features].copy()
            if data_for_cox.isnull().any().any():
                print("\nWarning: NaN values detected before Cox model fitting:")
                print(data_for_cox.isnull().sum()[data_for_cox.isnull().sum() > 0])
                print("\nDropping rows with NaN values...")
                data_for_cox = data_for_cox.dropna()
            
            print(f"\nFitting Cox model with {len(data_for_cox)} observations...")
            self.cox_model.fit(
                data_for_cox,
                duration_col='time_difference_hours',
                event_col='had_next_transfusion',
                show_progress=True
            )
            print("Cox model fitting complete.")
            
        except Exception as e:
            print(f"Error fitting Cox model: {str(e)}")
            raise

    # [Previous methods remain the same: analyze_feature_importance, fit_causal_forest, plot_results, save_results]

def main():
    """Run the causal analysis pipeline with error handling."""
    try:
        analysis = TransfusionCausalAnalysis('final_transfusion_analysis_with_time_to_next.csv')
        
        # Prepare data and fit models
        analysis.preprocess_data()
        analysis.fit_cox_model()
        analysis.analyze_feature_importance()
        analysis.fit_causal_forest()
        
        # Generate and save results
        analysis.plot_results()
        analysis.save_results()
        
    except Exception as e:
        print(f"Error in analysis pipeline: {str(e)}")
        raise

        
    def analyze_feature_importance(self):
        """Analyze feature importance using Random Forests and SHAP values."""
        # Prepare features and target
        X = self.processed_data.drop(['time_difference_hours', 'had_next_transfusion', 'anon_case_no'], axis=1)
        y = self.processed_data['time_difference_hours']
        
        # Fit Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(shap_values).mean(0)
        }).sort_values('importance', ascending=False)
        
    def fit_causal_forest(self):
        """Fit Causal Forest for heterogeneous treatment effect estimation."""
        # Prepare data for causal forest
        X = self.processed_data.drop(['time_difference_hours', 'had_next_transfusion', 'anon_case_no'], axis=1)
        T = self.processed_data['total_transfusion']  # Treatment (initial transfusion volume)
        Y = self.processed_data['time_difference_hours']  # Outcome
        
        # Fit causal forest
        self.causal_forest = CausalForestDML(
            model_t=RandomForestRegressor(n_estimators=100),
            model_y=RandomForestRegressor(n_estimators=100),
            n_estimators=1000,
            random_state=42
        )
        self.causal_forest.fit(Y, T, X=X)
        
    def plot_results(self):
        """Generate visualization of analysis results."""
        # Plot 1: Cox model hazard ratios
        self.cox_model.plot()
        plt.title('Hazard Ratios from Cox Proportional Hazards Model')
        plt.tight_layout()
        plt.savefig('cox_hazard_ratios.png')
        plt.close()
        
        # Plot 2: Feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=self.feature_importance.head(15),
            x='importance',
            y='feature'
        )
        plt.title('Top 15 Most Important Features (SHAP values)')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        # Plot 3: Treatment effect heterogeneity
        treatment_effects = self.causal_forest.effect(
            X=self.processed_data.drop(['time_difference_hours', 'had_next_transfusion', 'anon_case_no'], axis=1)
        )
        plt.figure(figsize=(10, 6))
        sns.histplot(treatment_effects, bins=50)
        plt.title('Distribution of Estimated Treatment Effects')
        plt.xlabel('Effect on Time to Next Transfusion (hours)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('treatment_effects.png')
        plt.close()
        
    def save_results(self):
        """Save analysis results to files."""
        # Save Cox model summary
        with open('cox_model_summary.txt', 'w') as f:
            f.write(self.cox_model.print_summary())
        
        # Save feature importance
        self.feature_importance.to_csv('feature_importance.csv', index=False)
        
        # Save causal forest effects
        effects_df = pd.DataFrame({
            'case_no': self.processed_data['anon_case_no'],
            'treatment_effect': self.causal_forest.effect(
                X=self.processed_data.drop(['time_difference_hours', 'had_next_transfusion', 'anon_case_no'], axis=1)
            )
        })
        effects_df.to_csv('treatment_effects.csv', index=False)

def main():
    """Run the causal analysis pipeline."""
    analysis = TransfusionCausalAnalysis('final_transfusion_analysis_with_time_to_next.csv')
    
    # Prepare data and fit models
    analysis.preprocess_data()
    analysis.fit_cox_model()
    analysis.analyze_feature_importance()
    analysis.fit_causal_forest()
    
    # Generate and save results
    analysis.plot_results()
    analysis.save_results()

if __name__ == '__main__':
    main()