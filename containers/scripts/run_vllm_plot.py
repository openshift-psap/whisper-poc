import json
import os
import plotly.graph_objects as go

# Directory where the output files are saved
output_dir = "/tmp/output"
images_dir = f"{output_dir}/images"

# Ensure the images directory exists
os.makedirs(images_dir, exist_ok=True)

# Metrics to plot
metrics = ["total_time", "latency", "seconds_transcribed_per_sec"]

# Data storage
throughput_values = []
batch_sizes = []
data = {metric: [] for metric in metrics}
labels = []
summary_text = ""

# Load the performance metrics from JSON files
for batch_size in [1, 2, 4, 8, 16, 32]:
    json_filename = f"{output_dir}/output-{batch_size:03d}.json"
    
    if os.path.exists(json_filename):
        with open(json_filename, "r") as f:
            metrics_data = json.load(f)
            
            throughput = metrics_data.get("throughput", 0)
            throughput_values.append(throughput)
            batch_sizes.append(batch_size)
            labels.append(f"BS: {batch_size}")
            
            summary_text += (
                f"Batch Size: {batch_size}, num_tokens: {metrics_data.get('num_tokens', 'N/A')}, "
                f"requests_processed: {metrics_data.get('requests_processed', 'N/A')}<br>"
            )
            
            for metric in metrics:
                data[metric].append(metrics_data.get(metric, 0))

# Generate and save plots
for metric in metrics:
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=throughput_values,
        y=data[metric],
        mode="markers+lines+text",  # Include "text" in the mode
        name=f"Throughput vs {metric}",
        text=labels,  # Display batch size labels next to markers
        textposition="top center",  # Position the text above the markers
        textfont=dict(size=16, color="black"),  # Increase font size for text on markers
        marker=dict(size=12)
    ))
    
    fig.update_layout(
        title=f"Throughput vs {metric}",
        title_font=dict(size=24),  # Increase the font size for the title
        xaxis_title="Throughput (tokens/sec)",
        xaxis_title_font=dict(size=18),  # Increase font size for x-axis label
        yaxis_title=metric,
        yaxis_title_font=dict(size=18),  # Increase font size for y-axis label
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=16)  # Increase font size for legend
        ),
        annotations=[
            dict(
                text=summary_text,
                xref="paper", yref="paper",
                x=0, y=-0.2,  # Adjusted y value to position the summary text below the plot
                showarrow=False,
                align="left",
                bgcolor="white",
                bordercolor="white",
                borderwidth=1,
                font=dict(size=16, color="black"),  # Increase font size for annotation text

            )
        ],
        width=1200,  # Adjust the width of the entire figure (canvas size)
        height=1200,  # Adjust the height of the entire figure (canvas size)
        margin=dict(l=50, r=50, t=50, b=200)  # Increased bottom margin to ensure space for annotation
    )
    
    # Save the figure with higher resolution
    fig.write_image(f"{images_dir}/{metric}.png", scale=3)
