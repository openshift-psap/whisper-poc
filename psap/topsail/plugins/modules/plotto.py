#!/usr/bin/python

from ansible.module_utils.basic import AnsibleModule
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

def generate_plots(csv_file_path, output_dir, module):
    """Generate plots from the CSV file and save them to the output directory."""

    # Ensure the CSV file exists
    if not os.path.exists(csv_file_path):
        module.fail_json(msg=f"CSV file '{csv_file_path}' does not exist.")

    base_dir = output_dir
    output_dir = os.path.join(output_dir, "images")
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            module.fail_json(msg=f"Failed to create output directory '{output_dir}': {e}")

    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_path, parse_dates=['timestamp'])
    except Exception as e:
        module.fail_json(msg=f"Failed to read CSV file '{csv_file_path}': {e}")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Filter out rows where any value is 0
    columns_to_check = ['utilization.gpu [%]', 'utilization.memory [%]', 'power.draw [W]']
    df = df[(df[columns_to_check] != 0).all(axis=1)]


    # Define columns to plot and their filenames
    plots = {
        'utilization.gpu [%]': 'gpu_utilization_plot.png',
        'utilization.memory [%]': 'memory_utilization_plot.png',
        'power.draw [W]': 'power_draw_plot.png'
    }

    generated_files = []

    # Read JSON data for vertical lines
    json_files = [f for f in os.listdir(base_dir) if f.endswith(".json")]
    vertical_lines = []
    for json_file in json_files:
        try:
            with open(os.path.join(base_dir, json_file), 'r') as f:
                data = json.load(f)
                start_time = pd.to_datetime(data["start_time"], unit='s')
                # end_time = pd.to_datetime(data["end_time"], unit='s')
                concurrency = data.get("concurrency", "N/A")
                vertical_lines.append((start_time, concurrency))
                # vertical_lines.append((end_time, concurrency))
        except Exception as e:
            module.fail_json(msg=f"Failed to read JSON file '{json_file}': {e}")

    # Generate and save plots
    for column, filename in plots.items():
        if column in df.columns:
            try:

                fig = make_subplots(
                    rows=1, cols=2,
                    column_widths=[2, 0.3],
                    subplot_titles=(f"{column} Over Time", f"{column} Distribution")
                )

                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df[column],
                        mode='markers',  # Only markers, no lines
                        name=f"{column} Over Time"
                    ),
                    row=1, col=1
                )

                for vline, concurrency in vertical_lines:
                    fig.add_vline(x=vline, line=dict(color='red', width=1, dash='dash'))
                    fig.add_annotation(
                        x=vline, y=df[column].max(),
                        text=f"{concurrency}",
                        showarrow=True,
                        arrowhead=2,
                        yshift=10
                    )

                fig.add_trace(
                    go.Box(
                        y=df[column],
                        name=f"{column} Distribution",
                        boxmean=True
                    ),
                    row=1, col=2
                )

                fig.update_layout(
                    title=f"{column} Analysis",
                    xaxis_title="Time",
                    yaxis_title=column,
                    showlegend=False,
                    width=1800
                )

                output_path = os.path.join(output_dir, filename)
                fig.write_image(output_path)
                generated_files.append(output_path)
            except Exception as e:
                module.fail_json(msg=f"Failed to generate plot for column '{column}': {e}")

    return generated_files

def main():
    """Main entry point for the Ansible module."""

    module = AnsibleModule(
        argument_spec=dict(
            csv_file_path=dict(type='str', required=True)
        ),
        supports_check_mode=True
    )

    csv_file_path = module.params['csv_file_path']

    # Determine the output directory based on the CSV file's location
    output_dir = os.path.dirname(csv_file_path)

    # If running in check mode, exit early
    if module.check_mode:
        module.exit_json(changed=False, msg="Check mode: No changes made.")

    try:
        generated_files = generate_plots(csv_file_path, output_dir, module)
        module.exit_json(changed=True, msg="Plots generated successfully.", files=generated_files)
    except Exception as e:
        module.fail_json(msg=f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
