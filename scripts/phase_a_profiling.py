import pandas as pd

# Define paths (Updated to match your actual file name)
input_file = 'working/working_dataset_small.csv'
output_report = 'reports/phase_a_dataset_profile.txt'

# Load dataset
df = pd.read_csv(input_file)

# Open the report file for writing
with open(output_report, 'w') as f:
    f.write("=== PHASE A: DATASET PROFILING ===\n\n")
    
    f.write(f"File analyzed: {input_file}\n")
    f.write(f"Total Rows: {df.shape[0]}\n")
    f.write(f"Total Columns: {df.shape[1]}\n\n")
    
    f.write("--- COLUMNS & DATA TYPES ---\n")
    f.write(f"{df.dtypes.to_string()}\n\n")
    
    f.write("--- CLASS DISTRIBUTION ---\n")
    f.write(f"{df['class'].value_counts().to_string()}\n\n")
    
    f.write("--- MISSING VALUES ---\n")
    f.write(f"{df.isnull().sum().to_string()}\n\n")
    
    f.write("--- SAMPLE OF RAW TEXT ---\n")
    for index, row in df.head(5).iterrows():
        f.write(f"Row {index}: IP='{row['ip']}' | Class='{row['class']}' | Text='{row['text']}'\n")

print(f"Profiling complete! Report saved to {output_report}")
print("\nHere is the class distribution we found:")
print(df['class'].value_counts())
