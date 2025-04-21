import pandas as pd

# --- Configuration ---
input_csv_filepath = 'legal_text_classification.csv'
output_txt_filepath = 'case_texts.txt'
# The column name containing the text you want to extract
column_to_extract = 'case_text'
# --- ---

# Read the entire CSV file into a pandas DataFrame
# pandas handles quoting and commas within fields automatically
# Specify encoding for robustness
df = pd.read_csv(input_csv_filepath, encoding='utf-8')

# Check if the required column exists in the DataFrame
# If not, accessing it below will raise a KeyError (as requested instead of try/except)
if column_to_extract not in df.columns:
     # You could raise an error here, but letting the next line fail provides traceback
     print(f"Warning: Column '{column_to_extract}' not found in the CSV header.")
     # Let it proceed to fail on the next line to get the standard traceback

# Select the specified column as a pandas Series
case_texts_series = df[column_to_extract]
print(f"Found '{column_to_extract}' column.")

# Open the output text file for writing
with open(output_txt_filepath, mode='w', encoding='utf-8') as outfile:
    # Iterate through the pandas Series and write each text block
    for text_block in case_texts_series:
        # Convert potential non-string types just in case, though usually not needed here
        outfile.write(str(text_block) + '\n') # Add a newline character

print(f"\nSuccessfully extracted {len(case_texts_series)} text blocks from '{column_to_extract}' column.")
print(f"Output saved to '{output_txt_filepath}'")