#!/usr/bin/python

from ansible.module_utils.basic import AnsibleModule
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_plots(csv_file_path, output_dir, module):
    """Generate plots from the CSV file and save them to the output directory."""

    # Ensure the CSV file exists
    if not os.path.exists(csv_file_path):
        module.fail_json(msg=f"CSV file '{csv_file_path}' does not exist.")

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

    # Generate and save plots
    for column, filename in plots.items():
        if column in df.columns:
            try:
                # Create a subplot layout with two panels: line plot and boxplot
                fig = make_subplots(
                    rows=1, cols=2,
                    column_widths=[0.7, 0.3],
                    subplot_titles=(f"{column} Over Time", f"{column} Distribution")
                )

                # Add line plot to the first panel
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df[column],
                        mode='lines+markers',
                        name=f"{column} Over Time"
                    ),
                    row=1, col=1
                )

                # Add boxplot to the second panel
                fig.add_trace(
                    go.Box(
                        y=df[column],
                        name=f"{column} Distribution",
                        boxmean=True
                    ),
                    row=1, col=2
                )

                # Update layout for aesthetics
                fig.update_layout(
                    title=f"{column} Analysis",
                    xaxis_title="Time",
                    yaxis_title=column,
                    showlegend=False
                )

                # Save the plot as an image
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
