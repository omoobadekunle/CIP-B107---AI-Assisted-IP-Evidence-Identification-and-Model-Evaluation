import json
import pandas as pd

train_file = 'working/working_dataset_train.jsonl'
test_file = 'working/working_dataset_test.jsonl'

def load_jsonl(path):
    records = []
    with open(path, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {i} is not valid JSON in {path}: {e}")
    return records

train = load_jsonl(train_file)
test = load_jsonl(test_file)

print("=== PHASE A: JSONL PROFILING ===")
print(f"Training records: {len(train)}")
print(f"Testing records:  {len(test)}")

print("\n--- LABEL BALANCE (TRAIN) ---")
train_df = pd.DataFrame(train)
train_df['completion'] = train_df['completion'].str.strip()
print(train_df['completion'].value_counts().to_string())

print("\n--- LABEL BALANCE (TEST) ---")
test_df = pd.DataFrame(test)
test_df['completion'] = test_df['completion'].str.strip()
print(test_df['completion'].value_counts().to_string())

print("\n--- FIRST TRAIN RECORD ---")
print(train[0])

print("\n--- CSV vs JSONL COMPARISON (FIRST 5 CSV ROWS) ---")
csv_df = pd.read_csv('working/working_dataset_small.csv')
for i in range(5):
    csv_row = csv_df.iloc[i]
    print(f"\nRow {i}:")
    print(f"  CSV ip:     {csv_row['ip']}")
    print(f"  CSV class:  {csv_row['class']}")
    print(f"  CSV text:   {csv_row['text']!r}")
    if i < len(train):
        print(f"  JSONL prompt:     {train[i]['prompt']}")
        print(f"  JSONL completion: {train[i]['completion']}")

# Save report
with open('reports/phase_a_jsonl_profile.txt', 'w') as f:
    f.write("=== PHASE A: JSONL PROFILING ===\n\n")
    f.write(f"Training records: {len(train)}\n")
    f.write(f"Testing records:  {len(test)}\n\n")
    f.write("--- LABEL BALANCE (TRAIN) ---\n")
    f.write(train_df['completion'].value_counts().to_string() + "\n\n")
    f.write("--- LABEL BALANCE (TEST) ---\n")
    f.write(test_df['completion'].value_counts().to_string() + "\n")

print("\nReport saved to reports/phase_a_jsonl_profile.txt")
