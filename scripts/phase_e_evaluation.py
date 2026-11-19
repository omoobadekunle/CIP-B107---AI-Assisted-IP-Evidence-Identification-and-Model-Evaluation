import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 1. Load the data
df = pd.read_csv('working/working_dataset_small.csv')
X = df['ip'].astype(str)
y = df['class']

# 2. Reproduce the exact train/test split from Phase C
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Reproduce the TF-IDF and Model Training
vectorizer = TfidfVectorizer(token_pattern=r'\d+')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# 4. Find the misclassifications
results = pd.DataFrame({
    'Actual_Class': y_test,
    'Predicted_Class': y_pred,
    'IP_Address': X_test
})

misclassified = results[results['Actual_Class'] != results['Predicted_Class']]

print("=== PHASE E: MISCLASSIFICATION ANALYSIS ===")
print(f"Total Test Samples: {len(results)}")
print(f"Correctly Classified: {len(results) - len(misclassified)}")
print(f"Misclassified: {len(misclassified)}\n")

if len(misclassified) > 0:
    print("--- DETAILS OF MISCLASSIFIED IPs ---")
    for index, row in misclassified.iterrows():
        print(f"IP: {row['IP_Address']} | Actual: {row['Actual_Class']} | Predicted: {row['Predicted_Class']}")
else:
    print("No misclassifications found in this test split.")

# 5. Save to report
with open('reports/phase_e_misclassification_analysis.txt', 'w') as f:
    f.write("=== PHASE E: MISCLASSIFICATION ANALYSIS ===\n\n")
    f.write(f"Total Test Samples: {len(results)}\n")
    f.write(f"Correctly Classified: {len(results) - len(misclassified)}\n")
    f.write(f"Misclassified: {len(misclassified)}\n\n")
    
    if len(misclassified) > 0:
        f.write("--- DETAILS OF MISCLASSIFIED IPs ---\n")
        for index, row in misclassified.iterrows():
            f.write(f"IP: {row['IP_Address']} | Actual: {row['Actual_Class']} | Predicted: {row['Predicted_Class']}\n")
    else:
        f.write("No misclassifications found.\n")

print("\nPhase E analysis saved to reports/phase_e_misclassification_analysis.txt")
