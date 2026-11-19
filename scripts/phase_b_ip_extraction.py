import pandas as pd
import re
import ipaddress

# Define paths
input_file = 'working/working_dataset_small.csv'
output_file = 'outputs/phase_b_extracted_ips.csv'

# Load dataset
df = pd.read_csv(input_file)

# Regex to find IPv4-like patterns
ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')

# Function to extract IPs from the text column
def extract_ips(text):
    if not isinstance(text, str):
        return []
    return ip_pattern.findall(text)

# Function to validate an IP
def is_valid_ip(ip_str):
    try:
        # ipaddress module is strict. It will reject 1.000.000.000 and >255
        ipaddress.ip_address(ip_str)
        return True
    except ValueError:
        return False

# Apply extraction
df['extracted_ips'] = df['text'].apply(extract_ips)

# Now, apply validation to each extracted IP
def validate_extracted(ip_list):
    valid_ips = []
    invalid_ips = []
    for ip in ip_list:
        if is_valid_ip(ip):
            valid_ips.append(ip)
        else:
            invalid_ips.append(ip)
    return pd.Series([valid_ips, invalid_ips])

# Separate valid and invalid lists into two new columns
df[['valid_ips', 'invalid_ips']] = df['extracted_ips'].apply(validate_extracted)

# Show a summary
total_valid = df['valid_ips'].apply(len).sum()
total_invalid = df['invalid_ips'].apply(len).sum()

print("--- VALIDATION SUMMARY ---")
print(f"Total Extracted: {total_valid + total_invalid}")
print(f"Valid IPs:       {total_valid}")
print(f"Invalid IPs:     {total_invalid}")

print("\n--- EXAMPLES OF INVALID/MISLEADING STRINGS CAUGHT ---")
for index, row in df.iterrows():
    if len(row['invalid_ips']) > 0:
        print(f"Text: {row['text']}")
        print(f"Rejected: {row['invalid_ips']}")
        print(f"Reason:   Invalid IPv4 format (e.g., octet > 255 or leading zeros)\n")

# Save the results
df['valid_ips_str'] = df['valid_ips'].apply(lambda x: ', '.join(x))
df['invalid_ips_str'] = df['invalid_ips'].apply(lambda x: ', '.join(x))
df.to_csv(output_file, index=False)

print(f"Validation complete. Results saved to {output_file}")
