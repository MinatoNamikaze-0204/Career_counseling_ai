# predict_input.py
import pandas as pd
import numpy as np
import pickle
from scipy import sparse

# Load trained model and encoders
with open('model/career_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('model/target_encoder.pkl', 'rb') as f:
    target_le = pickle.load(f)
with open('model/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
with open('model/vectorizers.pkl', 'rb') as f:
    vectorizers = pickle.load(f)

# Features as in training
features_cat = [
    'Gender', 'Field_of_Study', 'University_Location', 'Prior_Employment',
    'Employment_Type', 'Entrepreneurial_Experience', 'Entrepreneurial_Aspirations'
]
features_num = [
    'Age', 'Year_of_Study', 'GPA', 'Startup_Participation', 'Career_Guidance_Satisfaction'
]
text_features = list(vectorizers.keys())  # Use only trained text features

# EXAMPLE input — replace with your own data dictionary
input_data = {
    'Gender': 'Male',
    'Field_of_Study': 'Computer Science',
    'University_Location': 'Urban',
    'Prior_Employment': 'Yes',
    'Employment_Type': 'Internship',
    'Entrepreneurial_Experience': 'No',
    'Entrepreneurial_Aspirations': 'Yes',
    'Age': 21,
    'Year_of_Study': 3,
    'GPA': 8.5,
    'Startup_Participation': 1,
    'Career_Guidance_Satisfaction': 4,
    'Relevant_Coursework': 'AI, Machine Learning, Data Science',
    'Career_Interests': 'Software Development, Artificial Intelligence'
}

df_input = pd.DataFrame([input_data])

# Encode categorical features
X_cat = []
for col in features_cat:
    val = str(df_input[col].fillna('missing').values[0])
    le = label_encoders[col]
    if val in le.classes_:
        encoded = le.transform([val])[0]
    else:
        print(f"Unseen category '{val}' in '{col}' – using 'missing'")
        if 'missing' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'missing')
        encoded = le.transform(['missing'])[0]
    X_cat.append(encoded)

# Numeric features
X_num = df_input[features_num].fillna(0).values.flatten()

# Vectorize text features
vecs = []
for col in text_features:
    if col in df_input.columns:
        text = df_input[col].fillna('').astype(str).values
        vec = vectorizers[col]
        vec_matrix = vec.transform(text)
        vecs.append(vec_matrix)
    else:
        print(f"Text column '{col}' not found in input. Skipping.")

# Combine all features
X_cat_num = np.hstack([X_cat, X_num])
X_all = sparse.hstack([sparse.csr_matrix(X_cat_num.reshape(1, -1))] + vecs)

# Predict
y_pred = model.predict(X_all)
predicted_career = target_le.inverse_transform(y_pred)[0]

print(f"\nPredicted Career Path: {predicted_career}")
