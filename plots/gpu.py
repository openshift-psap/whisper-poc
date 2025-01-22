import pandas as pd
import plotly.express as px

# Read the CSV file
df = pd.read_csv('../whisper_bench-output/gpu_metrics.csv', parse_dates=['timestamp'])

# Clean column names by stripping any leading/trailing spaces
df.columns = df.columns.str.strip()

# Define a dictionary to map each column to its respective filename
plots = {
    'utilization.gpu [%]': 'gpu_utilization_plot.png',
    'utilization.memory [%]': 'memory_utilization_plot.png',
    'power.draw [W]': 'power_draw_plot.png'
}

# Create and save plots
for column, filename in plots.items():
    if column in df.columns:  # Ensure the column exists
        fig = px.line(
            df,
            x='timestamp',
            y=column,
            title=f"{column} Over Time",
            labels={'timestamp': 'Time', column: column},
            markers=True
        )
        fig.write_image(filename)  # Save the plot as a PNG image
