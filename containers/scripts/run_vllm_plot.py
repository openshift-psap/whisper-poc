import json
import os
import plotly.graph_objects as go

# Load the performance metrics from JSON files in /tmp/output/
latency_values = []
throughput_values = []
batch_sizes = []

# Directory where the output files are saved
output_dir = "/tmp/output"
images_dir = "/tmp/output/images"

# Make sure the images directory exists
os.makedirs(images_dir, exist_ok=True)

# Loop through the JSON files
for batch_size in [1, 2, 4, 8, 16, 32]:
    json_filename = f"{output_dir}/output-{batch_size:03d}.json"
    
    # Check if the JSON file exists
    if os.path.exists(json_filename):
        with open(json_filename, "r") as f:
            metrics = json.load(f)
            latency_values.append(metrics['latency'])
            throughput_values.append(metrics['throughput'])
            batch_sizes.append(batch_size)

# Create the Plotly graph for Latency vs Throughput
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=throughput_values,
    y=latency_values,
    mode='markers+lines',
    name='Latency vs Throughput',
    text=[f"Batch Size {batch_size}" for batch_size in batch_sizes],
    marker=dict(size=10)
))

fig.update_layout(
    title="Throughput vs Latency for Different Batch Sizes",
    xaxis_title="Throughput (tokens/sec)",
    yaxis_title="Latency (seconds/request)",
    showlegend=True
)

# Save the figure as a static image (e.g., latency.png)
fig.write_image(f"{images_dir}/latency.png")

# Optional: Save more graphs if needed (e.g., throughput vs batch size, etc.)
# For example, you can plot batch_size vs throughput or other metrics if required.
