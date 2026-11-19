import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create outputs folder if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# 1. Load the data
print("Loading dataset...")
df = pd.read_csv('working/working_dataset_small.csv')

# 2. Prepare Features (X) and Labels (y)
# We will use the 'ip' column as our text feature for the model
X = df['ip'].astype(str)
y = df['class']

# 3. Train-Test Split (80% training, 20% testing)
# We stratify by 'y' to ensure the rare classes (B, C, E) appear in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Feature Extraction (TF-IDF)
# This converts the IP strings (e.g., "192.168.1.1") into numerical arrays
vectorizer = TfidfVectorizer(token_pattern=r'\d+') # Extract just the numbers as tokens
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train the Model
print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_vec, y_train)

# 6. Evaluate the Model
print("Evaluating model...")
y_pred = model.predict(X_test_vec)

# Generate metrics
report = classification_report(y_test, y_pred, zero_division=0)
print("\n--- CLASSIFICATION REPORT ---")
print(report)

# Save the report to a text file
with open('reports/phase_c_classification_report.txt', 'w') as f:
    f.write("=== PHASE C: CLASSIFICATION REPORT ===\n\n")
    f.write(report)
print("Classification report saved to reports/phase_c_classification_report.txt")

# 7. Generate Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix - Phase C Classifier')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.tight_layout()

# Save the plot
plt.savefig('outputs/phase_c_confusion_matrix.png')
print("Confusion matrix plot saved to outputs/phase_c_confusion_matrix.png")

print("\nPhase C Complete.")
