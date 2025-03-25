import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
file_path = "Full_Dataset.csv"
df = pd.read_csv(file_path)
# Encode ASL letters as numeric labels
label_encoder = LabelEncoder()
df["Letter"] = label_encoder.fit_transform(df["Letter"])  # Converts 'A', 'B'... to 0, 1...

# Separate features and target
X = df.drop(columns=["Letter"])
y = df["Letter"]

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize sensor values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
svm_model = SVC(kernel="rbf", C=1, gamma="scale")
svm_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Extract SVM model parameters
support_vectors = svm_model.support_vectors_  # Support vectors
dual_coef = svm_model.dual_coef_  # Dual coefficients
intercept = svm_model.intercept_  # Bias term

# Save parameters to files for ESP32 deployment
np.savetxt("support_vectors.csv", support_vectors, delimiter=",")
np.savetxt("dual_coef.csv", dual_coef, delimiter=",")
np.savetxt("intercept.csv", intercept, delimiter=",")

print("Model parameters saved! Use these files for ESP32 integration.")

while(1):
    Thumb = float(input("Enter Thumb Value: "))
    Index =float(input("Enter Index Value: "))
    Middle = float(input("Enter Middle Value: "))
    Ring = float(input("Enter Ring Value: "))
    Pinky =float(input("Enter Pinky Value: "))
    sensor_values = np.array([[Thumb, Index, Middle, Ring, Pinky]])

    # Create a DataFrame with the same column names as the training data
    sensor_values = pd.DataFrame([[Thumb, Index, Middle, Ring, Pinky]], columns=X.columns)

    # Standardize the input values (same scaling as training data)
    sensor_values_scaled = scaler.transform(sensor_values)

    # Predict ASL gesture
    predicted_class = svm_model.predict(sensor_values_scaled)[0]
    predicted_letter = label_encoder.inverse_transform([predicted_class])[0]

    print(f"Predicted Gesture: {predicted_letter}")