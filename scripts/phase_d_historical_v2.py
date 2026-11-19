import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------- Load result.csv ----------
df = pd.read_csv('working/working_result.csv')

print("=== PHASE D: HISTORICAL FINE-TUNING ANALYSIS ===")
print(f"Total rows: {len(df)}")
print(f"Final step: {df['step'].max()}")
print(f"Final elapsed_tokens: {df['elapsed_tokens'].iloc[-1]}")
print(f"Final elapsed_examples: {df['elapsed_examples'].iloc[-1]}")

# ---------- Plot training loss vs step ----------
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(df['step'], df['training_loss'], color='steelblue', linewidth=0.8, label='Training loss')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Training Loss')
axes[0].set_title('Training Loss vs Step (full fine-tuning run)')
axes[0].grid(alpha=0.3)
axes[0].legend()

# Validation loss (only 92 non-null values)
val = df.dropna(subset=['validation_loss'])
axes[1].plot(val['step'], val['validation_loss'], color='darkorange', marker='o', markersize=2, linewidth=0.8, label='Validation loss')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Validation Loss')
axes[1].set_title('Validation Loss vs Step (92 recorded points)')
axes[1].grid(alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/phase_d_loss_plot.png')
plt.close()
print("\nSaved: outputs/phase_d_loss_plot.png")

# ---------- Classification metrics at final step ----------
print("\n--- FINAL CLASSIFICATION METRICS (from result.csv) ---")
final = df.dropna(subset=['classification/accuracy']).iloc[-1]
for col in ['classification/accuracy','classification/precision','classification/recall',
            'classification/auroc','classification/auprc','classification/f1.0']:
    print(f"  {col:30s}: {final[col]}")

# ---------- Validation accuracy behaviour ----------
print("\n--- VALIDATION SEQUENCE ACCURACY ---")
print(f"  First recorded validation accuracy: {val['validation_sequence_accuracy'].dropna().iloc[0] if len(val)>0 else 'N/A'}")
print(f"  Final recorded validation accuracy: {val['validation_sequence_accuracy'].dropna().iloc[-1] if len(val)>0 else 'N/A'}")
print(f"  Rows where validation_seq_acc < 1.0: {(val['validation_sequence_accuracy'] < 1.0).sum()}")
print(f"  Rows where validation_seq_acc == 1.0: {(val['validation_sequence_accuracy'] == 1.0).sum()}")

# ---------- Write report ----------
with open('reports/phase_d_historical_v2.txt', 'w') as f:
    f.write("=== PHASE D: HISTORICAL FINE-TUNING EVIDENCE ===\n\n")
    f.write("Source files: fine_tuning_steps.txt (49 steps) and result.csv (733 steps)\n")
    f.write("Note: fine_tuning_steps.txt is a truncated view of the longer result.csv run.\n\n")
    
    f.write("--- RUN LENGTH ---\n")
    f.write(f"Total training steps: {df['step'].max()}\n")
    f.write(f"Total elapsed tokens: {df['elapsed_tokens'].iloc[-1]}\n")
    f.write(f"Total elapsed examples: {df['elapsed_examples'].iloc[-1]}\n\n")
    
    f.write("--- TRAINING vs VALIDATION ---\n")
    f.write("Training accuracy reaches 1.0 by step 5 and stays there for all 733 steps.\n")
    f.write("Validation loss recorded only 92 times (sporadic validation).\n")
    f.write("The 1.0 training accuracy indicates memorisation, not generalisation.\n\n")
    
    f.write("--- FINAL CLASSIFICATION METRICS (from result.csv, step 733) ---\n")
    for col in ['classification/accuracy','classification/precision','classification/recall',
                'classification/auroc','classification/auprc','classification/f1.0']:
        f.write(f"  {col:30s}: {final[col]}\n")
    f.write("\n")
    f.write("IMPORTANT: The final accuracy of 0.9783 matches the fresh ML holdout\n")
    f.write("result. This is because both use the same test set, which contains\n")
    f.write("11 exact prompt overlaps with the training set. The historical\n")
    f.write("accuracy is inflated by the same leakage as our fresh run.\n\n")
    
    f.write("--- WHAT CANNOT BE CONCLUDED ---\n")
    f.write("1. The 1.0 training accuracy does NOT prove real-world reliability.\n")
    f.write("2. The AUROC of 0.9975 is high but measured on a leaky test set.\n")
    f.write("3. The historical 'ada' model used legacy OpenAI API endpoints that\n")
    f.write("   are no longer current. The workflow (upload -> create job -> poll\n")
    f.write("   status -> predict) cannot be reproduced with the supplied scripts\n")
    f.write("   because they contain embedded credentials and obsolete API patterns.\n")
    f.write("4. This is a HISTORICAL result, not a student-run model. It must not\n")
    f.write("   be presented as the student's own fine-tuning output.\n")

print("\nSaved: reports/phase_d_historical_v2.txt")
