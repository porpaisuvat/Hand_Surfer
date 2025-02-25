import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
import datetime

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
# 4) Data Augmentation: Add Synthetic Noise
# -----------------------------------------------------
def add_noise(data, noise_level=0.05):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

X_train = add_noise(X_train, noise_level=0.05)  # Small noise injection

# -----------------------------------------------------
# 5) Build the ANN Model (Stronger Regularization)
# -----------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(48, activation='relu', input_shape=(X_train.shape[1],), 
                          kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # L2 Regularization
    tf.keras.layers.Dropout(0.4),  # Strong Dropout
    tf.keras.layers.Dense(24, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # Multi-class classification
])

# Compile with Label Smoothing to Reduce Overconfidence
model.compile(optimizer='adam', 
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),  # Soft labels to avoid overconfidence
              metrics=['accuracy'])

# **Early Stopping**
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# -----------------------------------------------------
# 6) Train the Model
# -----------------------------------------------------
print("\n🚀 Training Model...")
y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes=len(label_encoder.classes_))
y_val_categorical = tf.keras.utils.to_categorical(y_val, num_classes=len(label_encoder.classes_))

history = model.fit(X_train, y_train_categorical, epochs=100, batch_size=16, 
                    validation_data=(X_val, y_val_categorical), callbacks=[early_stopping])

# -----------------------------------------------------
# 7) Evaluate Model on Blind Test Data
# -----------------------------------------------------
print("\n🔍 Evaluating on Blind Test Data...")
y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=len(label_encoder.classes_))
test_loss, test_acc = model.evaluate(X_test, y_test_categorical)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# -----------------------------------------------------
# 8) Confusion Matrix
# -----------------------------------------------------
y_pred = np.argmax(model.predict(X_test), axis=1)
conf_matrix = confusion_matrix(y_test, y_pred)

# ✅ Create timestamped folder name
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = f"models/{timestamp}"
os.makedirs(save_dir, exist_ok=True)  # Create folder

# ✅ Save Model & Preprocessing Files in Timestamped Folder
model.save(f"{save_dir}/model.h5")  
np.save(f"{save_dir}/label_encoder.npy", label_encoder.classes_)
np.save(f"{save_dir}/scaler_mean.npy", scaler.mean_)
np.save(f"{save_dir}/scaler_scale.npy", scaler.scale_)

# ✅ Generate and Save Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

conf_matrix_path = f"{save_dir}/confusion_matrix.png"
plt.savefig(conf_matrix_path)  # ✅ Save confusion matrix as image
plt.close()  # ✅ Close plot to free memory

print(f"\n✅ Model and confusion matrix saved in folder: {save_dir}")