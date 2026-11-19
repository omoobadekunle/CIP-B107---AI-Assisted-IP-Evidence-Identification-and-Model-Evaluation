import pandas as pd

# Load the intermediate results we just generated
df = pd.read_csv('outputs/phase_b_extracted_ips.csv')

# Calculate counts
total_valid = df['valid_ips'].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0).sum()
total_invalid = df['invalid_ips'].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0).sum()
total_extracted = total_valid + total_invalid

# Write to report file
with open('reports/phase_b_validation_summary.txt', 'w') as f:
    f.write("=== PHASE B: IP EXTRACTION & VALIDATION ===\n\n")
    f.write(f"Total IP-like strings extracted: {total_extracted}\n")
    f.write(f"Valid IPv4 Addresses:            {total_valid}\n")
    f.write(f"Invalid/Misleading Strings:      {total_invalid}\n\n")
    f.write("--- EXAMPLES OF INVALID STRINGS REJECTED ---\n")
    f.write("1. Octet > 255:      609.1.20.0, 257.0.1.38\n")
    f.write("2. Leading zeros:    1.000.000.000\n")
    f.write("3. File paths/URLs:  crop.0.0.798.798, rc-0.0.0.392\n")
    f.write("4. 5-part versions:  257.0.1.38.116\n")

print("Phase B validation summary saved to reports/phase_b_validation_summary.txt")
