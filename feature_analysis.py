import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from typing import Set, Dict
from pathlib import Path

class TransfusionAnalysis:
    """
    A class to handle the analysis of transfusion data across multiple datasets.
    Includes data loading, preprocessing, analysis, and visualization.
    """
    
    def __init__(self, data_dir: str = "data", min_transfusion: float = 800.0):
        """
        Initialize with data directory path and minimum transfusion threshold.
        
        Args:
            data_dir: Directory containing the data files
            min_transfusion: Minimum total transfusion volume to include in analysis
        """
        self.data_dir = Path(data_dir)
        self.datasets = {}
        self.final_table = None
        self.min_transfusion = min_transfusion
        
    def load_datasets(self) -> None:
        """Load all required datasets from CSV files."""
        dataset_files = {
            'pre_op_char': 'pre_op.char.csv',
            'pre_op_risk': 'pre_op.risk_index.csv',
            'intra_op': 'intra_op.drug_fluids.csv',
            'pre_op_lab': 'pre_op.lab.csv',
            'post_op_blood': 'post_op.blood_product.csv'
        }
        
        for key, filename in dataset_files.items():
            self.datasets[key] = pd.read_csv(self.data_dir / filename)
            
    def preprocess_data(self) -> None:
        """Preprocess all datasets and prepare them for merging."""
        # Convert fluid volume to numeric
        self.datasets['intra_op']['fluid_volume_actual'] = pd.to_numeric(
            self.datasets['intra_op']['fluid_volume_actual'], 
            errors='coerce'
        )
        
        # Select and deduplicate pre-op characteristics
        pre_op_columns = [
            'anon_case_no', 'age_time_of_surgery', 'gender', 'race',
            'smoking_history', 'alcohol_consumption', 'any_steroids_past_6_months',
            'curr_tcm_herbal_treatment', 'presence_of_malignancy', 'bmi'
        ]
        self.datasets['pre_op_char'] = (
            self.datasets['pre_op_char'][pre_op_columns]
            .drop_duplicates(subset='anon_case_no')
        )
        
        # Select and deduplicate pre-op risk factors
        risk_columns = ['anon_case_no', 'hypertension', 'cardiac_risk_index', 
                       'osa_risk_index', 'asa_class']
        self.datasets['pre_op_risk'] = (
            self.datasets['pre_op_risk'][risk_columns]
            .drop_duplicates(subset='anon_case_no')
        )
        
        # Aggregate transfusion volumes
        self.datasets['transfusion_summary'] = (
            self.datasets['intra_op']
            .groupby('anon_case_no', as_index=False)
            .agg(total_transfusion=('fluid_volume_actual', 'sum'))
        )
        
        # Filter for minimum transfusion volume
        self.datasets['transfusion_summary'] = (
            self.datasets['transfusion_summary']
            [self.datasets['transfusion_summary']['total_transfusion'] >= self.min_transfusion]
        )
        
        # Pivot lab results
        self.datasets['pre_op_lab_pivot'] = (
            self.datasets['pre_op_lab']
            .pivot_table(
                index='anon_case_no',
                columns='preop_lab_test_description',
                values='preop_lab_result_value',
                aggfunc='first'
            )
            .reset_index()
        )
    
    def find_common_cases(self) -> Set[str]:
        """Find cases common to all datasets."""
        return (
            set(self.datasets['pre_op_char']['anon_case_no']) &
            set(self.datasets['pre_op_risk']['anon_case_no']) &
            set(self.datasets['transfusion_summary']['anon_case_no']) &
            set(self.datasets['pre_op_lab_pivot']['anon_case_no']) &
            set(self.datasets['post_op_blood']['anon_case_no'])
        )
    
    def merge_datasets(self, common_cases: Set[str]) -> None:
        """Merge all datasets using common cases."""
        # Filter datasets to common cases
        filtered_datasets = {
            key: df[df['anon_case_no'].isin(common_cases)]
            for key, df in self.datasets.items()
        }
        
        # Merge all datasets
        self.final_table = (
            filtered_datasets['pre_op_char']
            .merge(filtered_datasets['pre_op_risk'], on='anon_case_no', how='inner')
            .merge(filtered_datasets['transfusion_summary'], on='anon_case_no', how='inner')
            .merge(filtered_datasets['pre_op_lab_pivot'], on='anon_case_no', how='inner')
            .merge(
                filtered_datasets['post_op_blood'][['anon_case_no', 'time_difference']], 
                on='anon_case_no', 
                how='inner'
            )
        ).drop_duplicates()
        
        # Convert time difference to hours
        self.final_table['time_difference_hours'] = (
            self.final_table['time_difference'].apply(self._convert_to_hours)
        )
        
        # Print summary of filtered data
        print(f"\nAnalysis Summary:")
        print(f"Total cases with transfusion ≥ {self.min_transfusion} ml: {len(self.final_table)}")
        print(f"Average transfusion volume: {self.final_table['total_transfusion'].mean():.2f} ml")
    
    @staticmethod
    def _convert_to_hours(time_str: str) -> float:
        """Convert time string to total hours."""
        try:
            if ' days ' in time_str:
                days, time = time_str.split(' days ')
                hours, minutes, seconds = map(int, time.split(':'))
                return int(days) * 24 + hours + minutes / 60 + seconds / 3600
            else:
                hours, minutes, seconds = map(int, time_str.split(':'))
                return hours + minutes / 60 + seconds / 3600
        except (ValueError, AttributeError):
            return np.nan
    
    def generate_plots(self) -> None:
        """Generate all analysis plots."""
        plot_configs = [
            {
                'type': 'boxplot',
                'params': {
                    'x': 'gender',
                    'y': 'total_transfusion',
                    'data': self.final_table,
                    'filename': 'total_transfusion_by_gender.png',
                    'title': f'Total Transfusion by Gender (≥{self.min_transfusion} ml)'
                }
            },
            {
                'type': 'scatter',
                'params': {
                    'x': 'bmi',
                    'y': 'total_transfusion',
                    'hue': 'presence_of_malignancy',
                    'data': self.final_table,
                    'filename': 'bmi_vs_total_transfusion.png',
                    'title': f'BMI vs Total Transfusion (≥{self.min_transfusion} ml)'
                }
            },
            {
                'type': 'count',
                'params': {
                    'x': 'hypertension',
                    'data': self.final_table,
                    'filename': 'hypertension_distribution.png',
                    'title': f'Hypertension Distribution (Transfusion ≥{self.min_transfusion} ml)'
                }
            },
            {
                'type': 'histogram',
                'params': {
                    'data': self.final_table['age_time_of_surgery'],
                    'bins': 15,
                    'kde': True,
                    'filename': 'age_distribution.png',
                    'title': f'Age Distribution (Transfusion ≥{self.min_transfusion} ml)'
                }
            }
        ]
        
        for config in plot_configs:
            self._create_plot(config)
        
        # Generate time difference distribution plot
        self._create_time_difference_plot()
    
    def _create_plot(self, config: Dict) -> None:
        """Create and save a single plot based on configuration."""
        plt.figure(figsize=(10, 6))
        
        plot_params = config['params'].copy()
        title = plot_params.pop('title')
        filename = plot_params.pop('filename')
        
        if config['type'] == 'boxplot':
            sns.boxplot(**plot_params)
        elif config['type'] == 'scatter':
            sns.scatterplot(**plot_params)
        elif config['type'] == 'count':
            sns.countplot(**plot_params)
        elif config['type'] == 'histogram':
            sns.histplot(**plot_params)
        
        plt.title(title)
        plt.savefig(filename)
        plt.close()
    
    def _create_time_difference_plot(self) -> None:
        """Create and save time difference distribution plot."""
        first_25_hours = self.final_table[
            self.final_table['time_difference_hours'] <= 25
        ]
        
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=first_25_hours['time_difference_hours'],
            bins=np.arange(0, 26, 1),
            kde=False,
            color='royalblue',
            edgecolor='black',
            alpha=0.9
        )
        
        plt.title(f'Distribution of Time Difference (First 25 Hours)\nTransfusion ≥{self.min_transfusion} ml',
                 fontsize=18, fontweight='bold', pad=15)
        plt.xlabel('Time Difference (in hours)', fontsize=14, labelpad=10)
        plt.ylabel('Frequency', fontsize=14, labelpad=10)
        
        # Customize plot appearance
        ax = plt.gca()
        ax.set_xlim(0, 25)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('time_difference_distribution_first_25_hours.png', dpi=300)
        plt.close()
    
    def save_results(self) -> None:
        """Save analysis results to files."""
        # Save final table
        self.final_table.to_csv('final_transfusion_analysis_with_time_to_next.csv', 
                              index=False)
        
        # Save analysis results
        total_blood_transfused = self.final_table['total_transfusion'].sum()
        with open('analysis_results.txt', 'w') as f:
            f.write('Feature Analysis Results\n')
            f.write('========================\n\n')
            f.write(f'Analysis for cases with transfusion ≥ {self.min_transfusion} ml\n\n')
            f.write('Descriptive Statistics\n')
            f.write(self.final_table.describe(include='all').to_string() + '\n\n')
            f.write(f'Total Blood Transfused: {total_blood_transfused:.2f} ml\n')
            f.write(f'Number of Cases: {len(self.final_table)}\n')
            f.write(f'Average Transfusion Volume: {self.final_table["total_transfusion"].mean():.2f} ml\n\n')
        
        print(f'Final table and analysis results saved to files')

def main():
    """Main execution function."""
    # Initialize and run analysis with minimum transfusion threshold
    analysis = TransfusionAnalysis(min_transfusion=800.0)
    analysis.load_datasets()
    analysis.preprocess_data()
    common_cases = analysis.find_common_cases()
    analysis.merge_datasets(common_cases)
    analysis.generate_plots()
    analysis.save_results()

if __name__ == '__main__':
    main()