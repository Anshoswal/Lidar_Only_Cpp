# import sys

# # Define the input and output filenames
# input_filename = 'point_level_data.csv'
# output_filename = 'z_binned_data.csv'

# # --- Binning Configuration ---
# Z_MIN = -0.110
# Z_MAX = 0.180
# NUM_BINS = 20
# BIN_WIDTH = (Z_MAX - Z_MIN) / NUM_BINS

# def process_cluster_and_bin(cluster_buffer, col_indices):
#     """
#     Normalizes intensity for a cluster, bins by z-coordinate, and returns a single row of binned data.
#     """
#     if not cluster_buffer:
#         return None

#     # --- 1. Extract data and perform intensity normalization ---
#     parsed_points = []
#     intensities = []
#     for line in cluster_buffer:
#         parts = line.strip().split(',')
#         try:
#             intensity = float(parts[col_indices['intensity']])
#             z_coord = float(parts[col_indices['z_coordinate']])
#             cluster_id = parts[col_indices['cluster_id']]
#             frame_id = parts[col_indices['frame_id']]
            
#             intensities.append(intensity)
#             parsed_points.append({'z': z_coord, 'intensity': intensity, 'cid': cluster_id, 'fid': frame_id})
#         except (ValueError, IndexError):
#             continue # Skip malformed lines

#     if not intensities:
#         return None

#     min_intensity = min(intensities)
#     max_intensity = max(intensities)
#     intensity_range = max_intensity - min_intensity

#     # Add normalized intensity to each point
#     for point in parsed_points:
#         if intensity_range == 0:
#             point['norm_intensity'] = 0.0
#         else:
#             point['norm_intensity'] = (point['intensity'] - min_intensity) / intensity_range

#     # --- 2. Place normalized intensities into z-bins ---
#     bins = [[] for _ in range(NUM_BINS)]
#     for point in parsed_points:
#         z = point['z']
#         if Z_MIN <= z < Z_MAX:
#             bin_index = int((z - Z_MIN) / BIN_WIDTH)
#             bins[bin_index].append(point['norm_intensity'])

#     # --- 3. Calculate the final value for each bin (average or -1) ---
#     final_bin_values = []
#     for bin_contents in bins:
#         if bin_contents:
#             average_intensity = sum(bin_contents) / len(bin_contents)
#             final_bin_values.append(str(average_intensity))
#         else:
#             final_bin_values.append('-1')
            
#     # Get cluster and frame ID from the first valid point
#     cluster_id = parsed_points[0]['cid']
#     frame_id = parsed_points[0]['fid']
    
#     return [cluster_id, frame_id] + final_bin_values

# try:
#     with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
#         # --- Write the new header for the binned data ---
#         header_line = infile.readline()
#         original_cols = header_line.strip().split(',')
        
#         # Create a map of column names to their index for easy access
#         col_indices = {name: i for i, name in enumerate(original_cols)}
        
#         bin_headers = [f'bin_{i+1}' for i in range(NUM_BINS)]
#         new_header = ['cluster_id', 'frame_id'] + bin_headers
#         outfile.write(','.join(new_header) + '\n')

#         cluster_buffer = []
#         for line in infile:
#             if not line.strip():
#                 # End of a cluster, process it
#                 binned_row = process_cluster_and_bin(cluster_buffer, col_indices)
#                 if binned_row:
#                     outfile.write(','.join(binned_row) + '\n')
#                 cluster_buffer = [] # Reset for next cluster
#             else:
#                 cluster_buffer.append(line)
        
#         # Process the very last cluster in the file
#         if cluster_buffer:
#             binned_row = process_cluster_and_bin(cluster_buffer, col_indices)
#             if binned_row:
#                 outfile.write(','.join(binned_row) + '\n')

#     print("Processing complete!")
#     print(f"Original file: '{input_filename}'")
#     print(f"Binned data saved to: '{output_filename}'")

# except FileNotFoundError:
#     print(f"Error: The file '{input_filename}' was not found.")
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")

import sys

# Define the input and output filenames
input_filename = 'point_level_data.csv'
output_filename = 'z_binned_data.csv'

# --- Binning Configuration ---
Z_MIN = -0.110
Z_MAX = 0.180
NUM_BINS = 20
BIN_WIDTH = (Z_MAX - Z_MIN) / NUM_BINS

def process_cluster_and_bin(cluster_buffer, col_indices):
    """
    Normalizes intensity for a cluster, bins by z-coordinate, and returns a single row of binned data.
    """
    if not cluster_buffer:
        return None

    # --- 1. Extract data and perform intensity normalization ---
    parsed_points = []
    intensities = []
    for line in cluster_buffer:
        parts = line.strip().split(',')
        try:
            intensity = float(parts[col_indices['intensity']])
            z_coord = float(parts[col_indices['z_coordinate']])
            cluster_id = parts[col_indices['cluster_id']]
            frame_id = parts[col_indices['frame_id']]
            color_marker = parts[col_indices['color_marker']]
            
            intensities.append(intensity)
            parsed_points.append({'z': z_coord, 'intensity': intensity, 'cid': cluster_id, 'fid': frame_id, 'marker': color_marker})
        except (ValueError, IndexError):
            continue # Skip malformed lines

    if not intensities:
        return None

    min_intensity = min(intensities)
    max_intensity = max(intensities)
    intensity_range = max_intensity - min_intensity

    # Add normalized intensity to each point
    for point in parsed_points:
        if intensity_range == 0:
            point['norm_intensity'] = 0.0
        else:
            point['norm_intensity'] = (point['intensity'] - min_intensity) / intensity_range

    # --- 2. Place normalized intensities into z-bins ---
    bins = [[] for _ in range(NUM_BINS)]
    for point in parsed_points:
        z = point['z']
        if Z_MIN <= z < Z_MAX:
            bin_index = int((z - Z_MIN) / BIN_WIDTH)
            bins[bin_index].append(point['norm_intensity'])

    # --- 3. Calculate the final value for each bin (average or -1) ---
    final_bin_values = []
    for bin_contents in bins:
        if bin_contents:
            average_intensity = sum(bin_contents) / len(bin_contents)
            final_bin_values.append(str(average_intensity))
        else:
            final_bin_values.append('-1')
            
    # Get cluster, frame ID, and color marker from the first valid point
    cluster_id = parsed_points[0]['cid']
    frame_id = parsed_points[0]['fid']
    color_marker = parsed_points[0]['marker']
    
    return [cluster_id, frame_id, color_marker] + final_bin_values

try:
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        # --- Write the new header for the binned data ---
        header_line = infile.readline()
        original_cols = header_line.strip().split(',')
        
        # Create a map of column names to their index for easy access
        col_indices = {name: i for i, name in enumerate(original_cols)}
        
        bin_headers = [f'bin_{i+1}' for i in range(NUM_BINS)]
        new_header = ['cluster_id', 'frame_id', 'color_marker'] + bin_headers
        outfile.write(','.join(new_header) + '\n')

        cluster_buffer = []
        for line in infile:
            if not line.strip():
                # End of a cluster, process it
                binned_row = process_cluster_and_bin(cluster_buffer, col_indices)
                if binned_row:
                    outfile.write(','.join(binned_row) + '\n')
                cluster_buffer = [] # Reset for next cluster
            else:
                cluster_buffer.append(line)
        
        # Process the very last cluster in the file
        if cluster_buffer:
            binned_row = process_cluster_and_bin(cluster_buffer, col_indices)
            if binned_row:
                outfile.write(','.join(binned_row) + '\n')

    print("Processing complete!")
    print(f"Original file: '{input_filename}'")
    print(f"Binned data saved to: '{output_filename}'")

except FileNotFoundError:
    print(f"Error: The file '{input_filename}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

