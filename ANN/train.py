import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# -----------------------------------------------------
# 1) Load Data from the 'data' Folder
# -----------------------------------------------------
data_folder = "data"
files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

X, y = [], []
for file in files:
    label = file.replace(".csv", "")  # Use file name as label
    filepath = os.path.join(data_folder, file)
    
    # Read CSV and drop unnecessary columns
    df = pd.read_csv(filepath)
    
    # Remove "Frame" column if present
    if "Frame" in df.columns:
        df = df.drop(columns=["Frame"])

    # Handle missing values (Replace NaN with 0)
    df = df.fillna(0)  # Replace missing hand data with 0

    X.extend(df.values)  # Add numerical data
    y.extend([label] * len(df))  # Assign label to all rows

X = np.array(X)
y = np.array(y)

# -----------------------------------------------------
# 2) Convert Labels to Numerical Values
# -----------------------------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert labels to numbers

# -----------------------------------------------------
# 3) Preprocess Data (Normalize)
# -----------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Standardize features

# Split into 70% Training, 20% Validation, 10% Blind Test
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)  # 10% test

print(f"Data Split: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")

# -----------------------------------------------------
# 4) Build the ANN Model (Reduce Overfitting)
# -----------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # More neurons
    tf.keras.layers.Dropout(0.3),  # Dropout layer to prevent overfitting
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # Multi-class classification
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# **Early Stopping**
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# -----------------------------------------------------
# 5) Train the Model
# -----------------------------------------------------
print("\n🚀 Training Model...")
history = model.fit(X_train, y_train, epochs=100, batch_size=10, 
                    validation_data=(X_val, y_val), callbacks=[early_stopping])

# -----------------------------------------------------
# 6) Evaluate Model on Blind Test Data
# -----------------------------------------------------
print("\n🔍 Evaluating on Blind Test Data...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# -----------------------------------------------------
# 7) Confusion Matrix
# -----------------------------------------------------
y_pred = np.argmax(model.predict(X_test), axis=1)
conf_matrix = confusion_matrix(y_test, y_pred)



# -----------------------------------------------------
# 8) Save the Model & Scaler
# -----------------------------------------------------
model.save("model.h5")  # Save trained model
np.save("label_encoder.npy", label_encoder.classes_)  # Save label mapping
np.save("scaler_mean.npy", scaler.mean_)  # Save scaler parameters
np.save("scaler_scale.npy", scaler.scale_)

print("\n✅ Model saved as 'model.h5'")
print("✅ Label encoder saved as 'label_encoder.npy'")
print("✅ Scaler saved as 'scaler_mean.npy' & 'scaler_scale.npy'")

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()