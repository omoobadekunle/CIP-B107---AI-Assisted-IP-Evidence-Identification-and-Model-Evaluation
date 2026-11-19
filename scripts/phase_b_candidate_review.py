import json
import re
import ipaddress
import pandas as pd

train_file = 'working/working_dataset_train.jsonl'
test_file = 'working/working_dataset_test.jsonl'

def load_jsonl(path):
    records = []
    with open(path, 'r') as f:
        for i, line in enumerate(f, 1):
            records.append(json.loads(line))
    return records

all_records = load_jsonl(train_file) + load_jsonl(test_file)
ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')

def is_valid_ipv4(s):
    try:
        ipaddress.IPv4Address(s)
        return True
    except ValueError:
        return False

rows = []
for idx, rec in enumerate(all_records):
    prompt = rec['prompt']
    label = rec['completion'].strip()
    
    # Prompt format: "context;candidate ->"
    # Split on the semicolon to get context and candidate
    if ';' in prompt:
        context, rest = prompt.split(';', 1)
        candidate = rest.replace('->', '').strip()
    else:
        context = prompt
        candidate = ''
    
    # Regex check
    regex_match = bool(ip_pattern.search(candidate))
    
    # Strict IPv4 check
    ipv4_valid = is_valid_ipv4(candidate) if candidate else False
    
    # Analyst note
    if not regex_match:
        note = "No IP-like pattern in candidate"
    elif not ipv4_valid and regex_match:
        note = "Regex matched but IPv4 validation failed (leading zeros, octet > 255, or extra groups)"
    elif ipv4_valid and label == "positive":
        note = "Valid IPv4 and source-labelled positive"
    elif ipv4_valid and label == "negative":
        note = "Valid IPv4 but source-labelled negative (ambiguous)"
    else:
        note = ""
    
    rows.append({
        'source_row': idx,
        'context': context.strip(),
        'candidate': candidate,
        'regex_match': regex_match,
        'ipv4_valid': ipv4_valid,
        'source_label': label,
        'analyst_note': note
    })

df = pd.DataFrame(rows)
df.to_csv('outputs/candidate_review.csv', index=False)

print(f"candidate_review.csv created with {len(df)} rows.")
print("\n--- SUMMARY ---")
print(f"Regex matches:  {df['regex_match'].sum()}")
print(f"Valid IPv4:     {df['ipv4_valid'].sum()}")
print(f"Source positive: {(df['source_label']=='positive').sum()}")
print(f"Source negative: {(df['source_label']=='negative').sum()}")
print("\n--- MISMATCHES BETWEEN REGEX AND VALID IPV4 ---")
mismatches = df[df['regex_match'] & ~df['ipv4_valid']]
print(f"Regex matched but failed IPv4 validation: {len(mismatches)}")
print(mismatches[['candidate', 'analyst_note']].head(10).to_string())
