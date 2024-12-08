# Transfusion and Data Analysis Project

## Overview

This repository contains Python scripts for analyzing transfusion data and related clinical information. It includes preprocessing, visualization, statistical analysis, survival modeling, and missing data handling. The goal is to provide insights into patient outcomes and transfusion requirements using structured tabular datasets.

## Features

1. **Data Preprocessing**: Cleaning and merging datasets, handling missing values, removing outliers, and standardizing features.
2. **Visualization**: Generating insightful plots for missing data, distributions, and relationships between variables.
3. **Survival Analysis**: Modeling time-to-event data using Cox Proportional Hazards.
4. **Causal Inference**: Estimating treatment effects using Causal Forests.
5. **Missing Data Analysis**: Detecting and visualizing missing data patterns.
6. **Feature Importance**: Analyzing feature importance with SHAP values for Random Forest models.

## Prerequisites

- Python 3.8 or higher
- The following Python packages:
  - `pandas`
  - `numpy`
  - `lifelines`
  - `matplotlib`
  - `seaborn`
  - `torch`
  - `sklearn`
  - `shap`
  - `econml`
  - `missingno`

Install dependencies using pip:

```bash
pip install pandas numpy lifelines matplotlib seaborn torch scikit-learn shap econml missingno

---

### Part 2: Data Requirements and How to Run

```markdown
## Data Requirements

The scripts require the following datasets:
- **pre_op.char.csv**: Preoperative characteristics.
- **pre_op.risk_index.csv**: Risk indices for patients.
- **intra_op.drug_fluids.csv**: Intraoperative fluid data.
- **pre_op.lab.csv**: Preoperative lab test results.
- **post_op.blood_product.csv**: Postoperative blood transfusion data.
- **final3.csv**: Merged and processed dataset for missing data analysis.

## How to Run

### 1. Preprocessing and Merging Data

**Script**: `preprocessing_and_merging.py`

This script preprocesses and merges datasets to create a unified table with features for analysis.

```bash
python preprocessing_and_merging.py

---

### Part 3: Survival and Causal Analysis

```markdown
### 2. Survival and Causal Analysis

**Script**: `survival_causal_analysis.py`

Performs survival analysis using Cox Proportional Hazards and causal inference with Causal Forests.

```bash
python survival_causal_analysis.py

---

### Part 4: Missing Data Visualization

```markdown
### 3. Missing Data Visualization

**Script**: `missing_data_visualization.py`

Visualizes missing data patterns using Missingno.

```bash
python missing_data_visualization.py

---

### Part 5: Time-to-Next Transfusion Analysis

```markdown
### 4. Time-to-Next Transfusion Analysis

**Script**: `time_to_next_transfusion.py`

Analyzes time between transfusion events and generates distribution plots.

```bash
python time_to_next_transfusion.py

---

### Part 6: Example Visualizations

```markdown
## Example Visualizations

1. **Missing Data Bar Chart**  
   ![Bar Chart](missing_data_bar_chart.png)

2. **Hazard Ratios from Cox Model**  
   ![Hazard Ratios](cox_hazard_ratios.png)

3. **Time Difference Distribution**  
   ![Time Difference](time_difference_distribution_first_25_hours.png)

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

This project is licensed under the MIT License.
