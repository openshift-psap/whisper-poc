import os
import json
import csv
import argparse

def collect_json_files(directory):
    """Collect all JSON files ending with '-summary.json' in the given directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith("-summary.json")]

def read_json(file_path):
    """Read and return the JSON content from a file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def write_csv(data, output_file):
    """Write data to a CSV file."""
    if not data:
        print("No data to write.")
        return
    
    keys = data[0].keys()  # Assuming all JSON files have the same structure
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

def main(directory, output_file):
    json_files = collect_json_files(directory)
    if not json_files:
        print("No matching JSON files found.")
        return
    
    data = [read_json(file) for file in json_files]
    write_csv(data, output_file)
    print(f"CSV file created: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON summary files to CSV.")
    parser.add_argument("directory", help="Directory containing JSON files")
    parser.add_argument("output", help="Output CSV file")
    args = parser.parse_args()
    
    main(args.directory, args.output)