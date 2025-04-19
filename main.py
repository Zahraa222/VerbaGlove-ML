import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = "ASL_Dataset.csv"
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

# Train ONE-VS-REST SVM model
svm_model = OneVsRestClassifier(SVC(kernel="rbf", C=1, gamma="scale"))
svm_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

for i, estimator in enumerate(svm_model.estimators_):
    print(f"Model {i} gamma:", estimator._gamma)
    
# Optionally, visualize the confusion matrix using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()




#Save Scaler
np.savetxt("scaler_mean.csv", scaler.mean_, delimiter=",")
np.savetxt("scaler_std.csv", scaler.scale_, delimiter=",")

#Export and extract each model
for i, estimator in enumerate(svm_model.estimators_):
    np.savetxt(f"support_vectors_{i}.csv", estimator.support_vectors_, delimiter=",")
    np.savetxt(f"dual_coef_{i}.csv", estimator.dual_coef_, delimiter=",")
    np.savetxt(f"intercept_{i}.csv", estimator.intercept_, delimiter=",")

#Save label mapping
#We converted the ASL letters to numeric labels using LabelEncoder
#We need to save this mapping to a file so we can use it later to convert the predicted numeric labels back to ASL letters
labels = list(label_encoder.classes_)
print("Class labels:", labels)

with open("label_mapping.txt", "w") as f:
    for idx, label in enumerate(labels):
        f.write(f"{idx}: {label}\n")



while 1:
    Thumb = float(input("Enter Thumb Value: "))
    Index =float(input("Enter Index Value: "))
    Middle = float(input("Enter Middle Value: "))
    Ring = float(input("Enter Ring Value: "))
    Pinky =float(input("Enter Pinky Value: "))
    indexTouch = float(input("Enter Index Touch Value: "))
    middleTouch = float(input("Enter Middle Touch Value: "))
    thumbTouch = float(input("Enter Thumb Touch Value: "))
    sensor_values = np.array([[Thumb, Index, Middle, Ring, Pinky, indexTouch, middleTouch, thumbTouch]])

    # Create a DataFrame with the same column names as the training data
    sensor_values = pd.DataFrame([[Thumb, Index, Middle, Ring, Pinky, indexTouch, middleTouch, thumbTouch]], columns=X.columns)

    # Standardize the input values (same scaling as training data)
    sensor_values_scaled = scaler.transform(sensor_values)

    # Predict ASL gesture
    predicted_class = svm_model.predict(sensor_values_scaled)[0]
    predicted_letter = label_encoder.inverse_transform([predicted_class])[0]

    print(f"Predicted Gesture: {predicted_letter}")