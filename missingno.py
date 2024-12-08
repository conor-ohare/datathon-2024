import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

# Step 1: Load the dataset
print("Loading dataset...")
df = pd.read_csv("final3.csv")
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.\n")

# Step 2: Drop columns that end with '_y'
print("Dropping columns that end with '_y'...")
df = df[[col for col in df.columns if not col.endswith('_y')]]
print(f"Remaining columns after dropping: {len(df.columns)}\n")

# Step 3: Set a date column as the index (if applicable)
if 'date_column' in df.columns:
    print("Processing date column...")
    df['date_column'] = pd.to_datetime(df['date_column'])  # Ensure datetime format
    df = df.set_index('date_column')
    print("Date column set as index.\n")
else:
    print("No date column found. Skipping date indexing.\n")

# Step 4: Define a utility function to save visualizations
def save_plot(plot_func, filename, title=None, figsize=(10, 6), dpi=300, **kwargs):
    """
    Utility function to save missingno plots as PNG files.
    
    Parameters:
    - plot_func: The missingno function to create the plot (e.g., msno.matrix).
    - filename: Name of the file to save the plot.
    - title: Optional title for the plot.
    - figsize: Size of the plot (default: (10, 6)).
    - dpi: Resolution of the plot (default: 300).
    - kwargs: Additional keyword arguments passed to the plot function.
    """
    plt.figure(figsize=figsize)
    plot_func(**kwargs)
    if title:
        plt.title(title, fontsize=16, pad=20)
    plt.savefig(filename, dpi=dpi)
    plt.close()
    print(f"Saved: {filename}")

# Step 5: Generate and save missing data visualizations
print("Creating missing data visualizations...")

# 1. Missing data matrix
save_plot(
    msno.matrix,
    "missing_data_matrix.png",
    title="Missing Data Matrix",
    figsize=(12, 8),
    df=df,
    sparkline=True
)

# 2. Bar chart of missing data
save_plot(
    msno.bar,
    "missing_data_bar_chart.png",
    title="Missing Data Bar Chart",
    figsize=(12, 8),
    df=df,
    color=(0.3, 0.6, 0.9)
)

# 3. Heatmap showing missing data correlations
save_plot(
    msno.heatmap,
    "missing_data_heatmap.png",
    title="Missing Data Correlation Heatmap",
    figsize=(12, 8),
    df=df,
    cmap="viridis"
)

# 4. Dendrogram of missing data clustering
save_plot(
    msno.dendrogram,
    "missing_data_dendrogram.png",
    title="Missing Data Dendrogram",
    figsize=(12, 8),
    df=df,
    orientation="top"
)

# 5. Filtered matrix for columns with missing data only
print("Filtering columns with missing data...")
missing_cols = df.columns[df.isnull().any()]
filtered_df = df[missing_cols]
if not filtered_df.empty:
    save_plot(
        msno.matrix,
        "filtered_missing_data_matrix.png",
        title="Filtered Missing Data Matrix (Columns with Missing Data Only)",
        figsize=(12, 8),
        df=filtered_df
    )
else:
    print("No columns with missing data found. Skipping filtered matrix.\n")

print("\nEnhanced missing data visualizations saved as PNG files.")