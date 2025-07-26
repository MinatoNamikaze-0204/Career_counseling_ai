# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
from scipy import sparse

# Load dataset
df = pd.read_csv('data/AI-based Career Recommendation System.csv')

# Columns
target_col = 'Recommended_Career_Path'
features_cat = [
    'Gender', 'Field_of_Study', 'University_Location', 'Prior_Employment',
    'Employment_Type', 'Entrepreneurial_Experience', 'Entrepreneurial_Aspirations'
]
features_num = [
    'Age', 'Year_of_Study', 'GPA', 'Startup_Participation', 'Career_Guidance_Satisfaction'
]
text_features = ['Relevant_Coursework', 'Career_Interests']

# Label Encode Categorical Features
label_encoders = {}
X_cat = pd.DataFrame()
for col in features_cat:
    le = LabelEncoder()
    col_data = df[col].fillna('missing').astype(str)
    X_cat[col] = le.fit_transform(col_data)
    label_encoders[col] = le

# Numeric Features — fill missing with 0
X_num = df[features_num].fillna(0)

# Vectorize Text Features
vectorizers = {}
vecs = []
for col in text_features:
    vec = CountVectorizer()
    text_data = df[col].fillna('').astype(str)
    # Skip text features with empty content
    if text_data.str.strip().str.len().sum() == 0:
        print(f"Warning: Skipping text feature '{col}' due to empty content.")
        continue
    try:
        vec_matrix = vec.fit_transform(text_data)
        if vec_matrix.shape[1] == 0:
            print(f"Warning: Skipping text feature '{col}' due to empty vocabulary.")
            continue
        vecs.append(vec_matrix)
        vectorizers[col] = vec
    except ValueError as e:
        print(f"Skipping column '{col}' due to error: {e}")

# Combine all features
X_cat_num = np.hstack([X_cat.values, X_num.values])
X_all = sparse.hstack([sparse.csr_matrix(X_cat_num)] + vecs)

# Encode Target
target_le = LabelEncoder()
y = target_le.fit_transform(df[target_col].fillna('missing').astype(str))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save model and encoders
os.makedirs('model', exist_ok=True)
with open('model/career_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('model/target_encoder.pkl', 'wb') as f:
    pickle.dump(target_le, f)
with open('model/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
with open('model/vectorizers.pkl', 'wb') as f:
    pickle.dump(vectorizers, f)

print("Training complete and models saved.")
