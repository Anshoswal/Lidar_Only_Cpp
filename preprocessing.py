# # import sys

# # # Define the input and output filenames
# # input_filename = 'point_level_data.csv'
# # output_filename = 'z_binned_data.csv'

# # # --- Binning Configuration ---
# # Z_MIN = -0.110
# # Z_MAX = 0.180
# # NUM_BINS = 20
# # BIN_WIDTH = (Z_MAX - Z_MIN) / NUM_BINS

# # def process_cluster_and_bin(cluster_buffer, col_indices):
# #     """
# #     Normalizes intensity for a cluster, bins by z-coordinate, and returns a single row of binned data.
# #     """
# #     if not cluster_buffer:
# #         return None

# #     # --- 1. Extract data and perform intensity normalization ---
# #     parsed_points = []
# #     intensities = []
# #     for line in cluster_buffer:
# #         parts = line.strip().split(',')
# #         try:
# #             intensity = float(parts[col_indices['intensity']])
# #             z_coord = float(parts[col_indices['z_coordinate']])
# #             cluster_id = parts[col_indices['cluster_id']]
# #             frame_id = parts[col_indices['frame_id']]
# #             # --- MODIFICATION: Corrected the column name from 'color_marker_value' to 'color_marker' ---
# #             color_marker_value = parts[col_indices['color_marker']]
            
# #             intensities.append(intensity)
# #             parsed_points.append({'z': z_coord, 'intensity': intensity, 'cid': cluster_id, 'fid': frame_id, 'marker': color_marker_value})
# #         except (ValueError, IndexError, KeyError) as e:
# #             # print(f"Skipping malformed line or header: {line.strip()} | Error: {e}")
# #             continue # Skip malformed lines

# #     if not intensities:
# #         return None

# #     min_intensity = min(intensities)
# #     max_intensity = max(intensities)
# #     intensity_range = max_intensity - min_intensity

# #     # Add normalized intensity to each point
# #     for point in parsed_points:
# #         if intensity_range == 0:
# #             point['norm_intensity'] = 0.0
# #         else:
# #             point['norm_intensity'] = (point['intensity'] - min_intensity) / intensity_range

# #     # --- 2. Place normalized intensities into z-bins ---
# #     bins = [[] for _ in range(NUM_BINS)]
# #     for point in parsed_points:
# #         z = point['z']
# #         if Z_MIN <= z < Z_MAX:
# #             bin_index = int((z - Z_MIN) / BIN_WIDTH)
# #             bins[bin_index].append(point['norm_intensity'])

# #     # --- 3. Calculate the final value for each bin (average or -1) ---
# #     final_bin_values = []
# #     for bin_contents in bins:
# #         if bin_contents:
# #             average_intensity = sum(bin_contents) / len(bin_contents)
# #             final_bin_values.append(str(average_intensity))
# #         else:
# #             final_bin_values.append('-1')
            
# #     # Get cluster, frame ID, and color marker from the first valid point
# #     cluster_id = parsed_points[0]['cid']
# #     frame_id = parsed_points[0]['fid']
# #     color_marker = parsed_points[0]['marker']
    
# #     return [cluster_id, frame_id, color_marker] + final_bin_values

# # try:
# #     with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
# #         # --- Write the new header for the binned data ---
# #         header_line = infile.readline()
# #         original_cols = header_line.strip().split(',')
        
# #         # Create a map of column names to their index for easy access
# #         col_indices = {name: i for i, name in enumerate(original_cols)}
        
# #         bin_headers = [f'bin_{i+1}' for i in range(NUM_BINS)]
# #         new_header = ['cluster_id', 'frame_id', 'color_marker'] + bin_headers
# #         outfile.write(','.join(new_header) + '\n')

# #         cluster_buffer = []
# #         for line in infile:
# #             if not line.strip():
# #                 # End of a cluster, process it
# #                 binned_row = process_cluster_and_bin(cluster_buffer, col_indices)
# #                 if binned_row:
# #                     outfile.write(','.join(binned_row) + '\n')
# #                 cluster_buffer = [] # Reset for next cluster
# #             else:
# #                 cluster_buffer.append(line)
        
# #         # Process the very last cluster in the file
# #         if cluster_buffer:
# #             binned_row = process_cluster_and_bin(cluster_buffer, col_indices)
# #             if binned_row:
# #                 outfile.write(','.join(binned_row) + '\n')

# #     print("Processing complete!")
# #     print(f"Original file: '{input_filename}'")
# #     print(f"Binned data saved to: '{output_filename}'")

# # except FileNotFoundError:
# #     print(f"Error: The file '{input_filename}' was not found.")
# # except Exception as e:
# #     print(f"An unexpected error occurred: {e}")

# import sys

# # --- Configuration ---
# INPUT_FILENAME = 'point_level_data.csv'
# OUTPUT_FILENAME = 'z_binned_data.csv'
# Z_MIN = -0.110
# Z_MAX = 0.180
# NUM_BINS = 20
# BIN_WIDTH = (Z_MAX - Z_MIN) / NUM_BINS

# def process_cluster_and_bin(cluster_buffer, col_indices):
#     """
#     Normalizes intensity for a cluster, bins by z-coordinate, and returns a 
#     single row of binned data. This function remains largely the same as your original.
#     """
#     if not cluster_buffer:
#         return None

#     # --- 1. Extract data and perform intensity normalization ---
#     parsed_points = []
#     intensities = []
    
#     # Store key identifiers from the first point
#     first_point = cluster_buffer[0]
#     cluster_id = first_point[col_indices['cluster_id']]
#     frame_id = first_point[col_indices['frame_id']]
#     color_marker = first_point[col_indices['color_marker']]

#     for parts in cluster_buffer:
#         try:
#             intensity = float(parts[col_indices['intensity']])
#             z_coord = float(parts[col_indices['z_coordinate']])
#             intensities.append(intensity)
#             # We already have the main identifiers, just need z and intensity here
#             parsed_points.append({'z': z_coord, 'intensity': intensity})
#         except (ValueError, IndexError) as e:
#             # This handles cases where a row might be malformed
#             print(f"Skipping malformed data row: {','.join(parts)} | Error: {e}")
#             continue

#     if not intensities:
#         return None

#     min_intensity = min(intensities)
#     max_intensity = max(intensities)
#     intensity_range = max_intensity - min_intensity

#     # Add normalized intensity to each point
#     for i, point in enumerate(parsed_points):
#         # The original intensity is retrieved from the initial parsed_points list
#         original_intensity = point['intensity']
#         if intensity_range == 0:
#             point['norm_intensity'] = 0.0
#         else:
#             point['norm_intensity'] = (original_intensity - min_intensity) / intensity_range

#     # --- 2. Place normalized intensities into z-bins ---
#     bins = [[] for _ in range(NUM_BINS)]
#     for point in parsed_points:
#         z = point['z']
#         if Z_MIN <= z < Z_MAX:
#             try:
#                 bin_index = int((z - Z_MIN) / BIN_WIDTH)
#                 # Ensure bin_index is within the valid range
#                 if 0 <= bin_index < NUM_BINS:
#                     bins[bin_index].append(point['norm_intensity'])
#             except (ValueError, IndexError):
#                 continue # Skip if z-coordinate is out of expected range

#     # --- 3. Calculate the final value for each bin (average or -1) ---
#     final_bin_values = []
#     for bin_contents in bins:
#         if bin_contents:
#             average_intensity = sum(bin_contents) / len(bin_contents)
#             final_bin_values.append(f"{average_intensity:.6f}") # Format for consistency
#         else:
#             final_bin_values.append('-1')
            
#     return [cluster_id, frame_id, color_marker] + final_bin_values

# def main():
#     """
#     Main function to read the input file, process it line by line,
#     and write the binned data to the output file.
#     """
#     try:
#         with open(INPUT_FILENAME, 'r') as infile, open(OUTPUT_FILENAME, 'w') as outfile:
#             header_line = infile.readline().strip()
#             original_cols = header_line.split(',')
            
#             # Create a map of column names to their index for robust access
#             try:
#                 col_indices = {name: i for i, name in enumerate(original_cols)}
#                 # Verify necessary columns exist
#                 required_cols = ['cluster_id', 'frame_id', 'z_coordinate', 'intensity', 'color_marker']
#                 for col in required_cols:
#                     if col not in col_indices:
#                         raise ValueError(f"Missing required column in header: '{col}'")
#             except ValueError as e:
#                 print(f"Error processing header: {e}")
#                 sys.exit(1)

#             # --- Write the new header for the binned data ---
#             bin_headers = [f'bin_{i+1}' for i in range(NUM_BINS)]
#             new_header = ['cluster_id', 'frame_id', 'color_marker'] + bin_headers
#             outfile.write(','.join(new_header) + '\n')

#             # --- State tracking variables ---
#             current_cluster_id = None
#             current_frame_id = None
#             cluster_buffer = []

#             for line in infile:
#                 line = line.strip()
#                 if not line:
#                     continue  # Skip empty lines

#                 parts = line.split(',')
#                 if len(parts) != len(original_cols):
#                     continue # Skip rows with incorrect number of columns

#                 cluster_id = parts[col_indices['cluster_id']]
#                 frame_id = parts[col_indices['frame_id']]

#                 # Initialize on the first line of data
#                 if current_cluster_id is None:
#                     current_cluster_id = cluster_id
#                     current_frame_id = frame_id

#                 # Check for a change in cluster_id or frame_id, which signals the end of the current cluster
#                 if cluster_id != current_cluster_id or frame_id != current_frame_id:
#                     # Process the buffered cluster
#                     if cluster_buffer:
#                         binned_row = process_cluster_and_bin(cluster_buffer, col_indices)
#                         if binned_row:
#                             outfile.write(','.join(binned_row) + '\n')
                    
#                     # Reset the buffer and update the state for the new cluster
#                     cluster_buffer = [parts]
#                     current_cluster_id = cluster_id
#                     current_frame_id = frame_id
#                 else:
#                     # If it's the same cluster, just add the data to the buffer
#                     cluster_buffer.append(parts)
            
#             # --- Process the very last cluster in the file ---
#             if cluster_buffer:
#                 binned_row = process_cluster_and_bin(cluster_buffer, col_indices)
#                 if binned_row:
#                     outfile.write(','.join(binned_row) + '\n')

#         print("Processing complete!")
#         print(f"Original file: '{INPUT_FILENAME}'")
#         print(f"Binned data saved to: '{OUTPUT_FILENAME}'")

#     except FileNotFoundError:
#         print(f"Error: The file '{INPUT_FILENAME}' was not found.")
#         sys.exit(1)
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         sys.exit(1)

# if __name__ == '__main__':
#     main()

import sys

# --- Configuration ---
# List of input CSV files to be merged and processed.
INPUT_FILENAMES = ['point_level_data.csv', 'point_level_data_2.csv'] 
OUTPUT_FILENAME = 'z_binned_data_merged.csv'
Z_MIN = -0.110
Z_MAX = 0.180
NUM_BINS = 20
BIN_WIDTH = (Z_MAX - Z_MIN) / NUM_BINS

def process_cluster_and_bin(cluster_buffer, col_indices):
    """
    Normalizes intensity for a cluster, bins by z-coordinate, and returns a 
    single row of binned data. This function remains largely the same.
    """
    if not cluster_buffer:
        return None

    # --- 1. Extract data and perform intensity normalization ---
    parsed_points = []
    intensities = []
    
    # Store key identifiers from the first point
    first_point = cluster_buffer[0]
    cluster_id = first_point[col_indices['cluster_id']]
    frame_id = first_point[col_indices['frame_id']]
    color_marker = first_point[col_indices['color_marker']]

    for parts in cluster_buffer:
        try:
            intensity = float(parts[col_indices['intensity']])
            z_coord = float(parts[col_indices['z_coordinate']])
            intensities.append(intensity)
            # We already have the main identifiers, just need z and intensity here
            parsed_points.append({'z': z_coord, 'intensity': intensity})
        except (ValueError, IndexError) as e:
            # This handles cases where a row might be malformed
            print(f"Skipping malformed data row: {','.join(parts)} | Error: {e}")
            continue

    if not intensities:
        return None

    min_intensity = min(intensities)
    max_intensity = max(intensities)
    intensity_range = max_intensity - min_intensity

    # Add normalized intensity to each point
    for i, point in enumerate(parsed_points):
        # The original intensity is retrieved from the initial parsed_points list
        original_intensity = point['intensity']
        if intensity_range == 0:
            point['norm_intensity'] = 0.0
        else:
            point['norm_intensity'] = (original_intensity - min_intensity) / intensity_range

    # --- 2. Place normalized intensities into z-bins ---
    bins = [[] for _ in range(NUM_BINS)]
    for point in parsed_points:
        z = point['z']
        if Z_MIN <= z < Z_MAX:
            try:
                bin_index = int((z - Z_MIN) / BIN_WIDTH)
                # Ensure bin_index is within the valid range
                if 0 <= bin_index < NUM_BINS:
                    bins[bin_index].append(point['norm_intensity'])
            except (ValueError, IndexError):
                continue # Skip if z-coordinate is out of expected range

    # --- 3. Calculate the final value for each bin (average or -1) ---
    final_bin_values = []
    for bin_contents in bins:
        if bin_contents:
            average_intensity = sum(bin_contents) / len(bin_contents)
            final_bin_values.append(f"{average_intensity:.6f}") # Format for consistency
        else:
            final_bin_values.append('-1')
            
    return [cluster_id, frame_id, color_marker] + final_bin_values

def main():
    """
    Main function to read multiple input files, merge them with continuous
    frame IDs, and write the binned data to the output file.
    """
    try:
        with open(OUTPUT_FILENAME, 'w') as outfile:
            # --- State tracking variables ---
            current_cluster_id = None
            current_frame_id = None
            cluster_buffer = []
            frame_id_offset = 0
            col_indices = {}
            original_cols = []

            # Loop through each input file
            for file_index, filename in enumerate(INPUT_FILENAMES):
                print(f"Processing file: '{filename}'...")
                try:
                    with open(filename, 'r') as infile:
                        # Read header and set up column mapping from the first file
                        if file_index == 0:
                            header_line = infile.readline().strip()
                            original_cols = header_line.split(',')
                            try:
                                col_indices = {name: i for i, name in enumerate(original_cols)}
                                required_cols = ['cluster_id', 'frame_id', 'z_coordinate', 'intensity', 'color_marker']
                                for col in required_cols:
                                    if col not in col_indices:
                                        raise ValueError(f"Missing required column in header: '{col}'")
                            except ValueError as e:
                                print(f"Error processing header in {filename}: {e}")
                                return
                            
                            # Write the new header to the output file
                            bin_headers = [f'bin_{i+1}' for i in range(NUM_BINS)]
                            new_header = ['cluster_id', 'frame_id', 'color_marker'] + bin_headers
                            outfile.write(','.join(new_header) + '\n')
                        else:
                            # For subsequent files, just skip the header
                            infile.readline()
                        
                        max_frame_id_in_file = -1

                        # Process each line in the current file
                        for line in infile:
                            line = line.strip()
                            if not line:
                                continue

                            parts = line.split(',')
                            if len(parts) != len(original_cols):
                                continue

                            # --- Frame ID continuity logic ---
                            original_frame_id = int(parts[col_indices['frame_id']])
                            max_frame_id_in_file = max(max_frame_id_in_file, original_frame_id)
                            
                            # Apply offset to create a continuous frame ID
                            adjusted_frame_id_str = str(original_frame_id + frame_id_offset)
                            
                            # Overwrite the original frame_id in our working data
                            parts[col_indices['frame_id']] = adjusted_frame_id_str
                            
                            cluster_id = parts[col_indices['cluster_id']]
                            frame_id = adjusted_frame_id_str

                            # Initialize on the first line of data
                            if current_cluster_id is None:
                                current_cluster_id = cluster_id
                                current_frame_id = frame_id

                            # Check for a change in cluster or (adjusted) frame ID
                            if cluster_id != current_cluster_id or frame_id != current_frame_id:
                                if cluster_buffer:
                                    binned_row = process_cluster_and_bin(cluster_buffer, col_indices)
                                    if binned_row:
                                        outfile.write(','.join(binned_row) + '\n')
                                
                                cluster_buffer = [parts]
                                current_cluster_id = cluster_id
                                current_frame_id = frame_id
                            else:
                                cluster_buffer.append(parts)
                        
                        # After processing a file, update the offset for the next one
                        if max_frame_id_in_file > -1:
                            frame_id_offset += max_frame_id_in_file + 1

                except FileNotFoundError:
                    print(f"Warning: The file '{filename}' was not found. Skipping.")
                    continue
            
            # Process the very last cluster from the last file
            if cluster_buffer:
                binned_row = process_cluster_and_bin(cluster_buffer, col_indices)
                if binned_row:
                    outfile.write(','.join(binned_row) + '\n')

        print("\nProcessing complete!")
        print(f"Merged and binned data saved to: '{OUTPUT_FILENAME}'")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

