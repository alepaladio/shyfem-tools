#!/usr/bin/env python3
"""
Convert text file with headers to simple format with no headers.
Output format: yyyy-mm-dd::HH:MM:SS variable
Usage: python convert_data.py input.txt output.txt
"""

import sys
from datetime import datetime

def read_input_file(filename, header_lines):
    """Read the input file and return the data lines."""
    try:
        with open(filename, 'r') as file:
            # Skip the N (header_lines) header lines
            for _ in range(header_lines):
                file.readline()
            # Read the remaining data lines
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def parse_line(line):
    """Parse a line and extract the first datetime and value."""
    parts = line.split(',')
    if len(parts) >= 3:
        datetime_str = parts[0]  # Format: yyyy-mm-dd HH:MM:SS+00:00
        value = parts[2]
        
        # Parse the datetime and reformat it
        try:
            # Remove the +00:00 timezone part and parse
            clean_datetime = datetime_str.split('+')[0]
            dt = datetime.strptime(clean_datetime, '%Y-%m-%d %H:%M:%S')
            # Reformat to yyyy-mm-dd::HH:MM:SS
            formatted_datetime = dt.strftime('%Y-%m-%d::%H:%M:%S')
            return f"{formatted_datetime} {value}"
        except ValueError as e:
            print(f"Warning: Could not parse date '{datetime_str}': {e}")
            return None
    return None

def process_data(data_lines):
    """Process all data lines and return structured data."""
    processed_lines = []
    for line in data_lines:
        parsed = parse_line(line)
        if parsed:
            processed_lines.append(parsed)
    return processed_lines

def write_output_file(data, output_filename):
    """Write the processed data to output file."""
    try:
        with open(output_filename, 'w') as file:
            for line in data:
                file.write(line + '\n')
        print(f"Successfully converted data to '{output_filename}'")
    except Exception as e:
        print(f"Error writing file: {e}")
        sys.exit(1)

def main():
    # Check command line arguments
    if len(sys.argv) != 4:
        print("Usage: python convert_data.py <input_file> <output_file> <header_lines>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    header_lines = sys.argv[3]
    
    # Process the data
    print(f"Reading data from '{input_file}'...")
    data_lines = read_input_file(input_file, int(header_lines))
    
    if not data_lines:
        print("Warning: No data lines found after headers.")
        sys.exit(0)
    
    print(f"Processing {len(data_lines)} lines of data...")
    processed_data = process_data(data_lines)
    
    print(f"Writing data to '{output_file}'...")
    write_output_file(processed_data, output_file)
    
    print("Conversion completed successfully!")
    print(f"Total records converted: {len(processed_data)}")

if __name__ == "__main__":
    main()