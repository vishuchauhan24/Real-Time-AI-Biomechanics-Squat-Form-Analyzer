import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

print("1. Loading multi-feature dataset...")
try:
    df = pd.read_csv("advanced_biomechanics_data.csv")
except FileNotFoundError:
    print("Error: advanced_biomechanics_data.csv not found. Run video_pipeline.py first.")
    exit()

# Define our 4 geometric features (X) and the classification label (y)
X = df[['left_knee', 'right_knee', 'left_hip', 'right_hip']]
y = df['label']

print(f"Dataset contains {len(df)} total frames across 4 geometric dimensions.")

# 2. Train-Test Split (80% training, 20% validation testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("3. Optimization: Training Advanced Random Forest Classifier...")
# We use 100 decision trees to handle the multi-variable data smoothly
clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
clf.fit(X_train, y_train)

# 4. Comprehensive Evaluation Metrics
accuracy = clf.score(X_test, y_test)
print(f"\n[METRICS] Overall Model Accuracy: {accuracy * 100:.2f}%")

# Generate a detailed classification report (Precision, Recall, F1-Score)
y_pred = clf.predict(X_test)
print("\nDetailed Performance Report:")
print(classification_report(y_test, y_pred))

# 5. Serialization
with open("advanced_squat_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Success! Advanced model saved as 'advanced_squat_model.pkl'.")
