# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# # --- 1. Load the Processed Data ---
# # This file should be the output from your previous script.
# input_filename = 'z_binned_data.csv'

# try:
#     data = pd.read_csv(input_filename)
# except FileNotFoundError:
#     print(f"Error: The file '{input_filename}' was not found.")
#     print("Please run the 'process_intensity.py' script first to generate this file.")
#     exit()

# # --- 2. Prepare the Data for the Model ---

# # Separate features (the 20 bins) from the target (color_marker)
# # We select all columns that start with 'bin_' as our features.
# features = data.loc[:, data.columns.str.startswith('bin_')]
# target = data['color_marker']

# # Neural networks for binary classification typically expect labels to be 0 and 1.
# # We need to convert our -1 and 1 labels into 0 and 1.
# # We'll map -1 to 0, and keep 1 as 1.
# target_mapped = target.replace(-1, 0)

# # Split the data into a training set (to teach the model) and a testing set (to evaluate it)
# # 80% of the data will be for training, 20% for testing.
# X_train, X_test, y_train, y_test = train_test_split(
#     features, target_mapped, test_size=0.2, random_state=42, stratify=target_mapped
# )

# # It's good practice to scale your input data. This helps the model learn faster and more effectively.
# # We will scale the bin values.
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


# # --- 3. Build the 1-D Artificial Neural Network (ANN) ---

# model = tf.keras.Sequential([
#     # Input Layer: It expects a 1D array of length 20 (the number of bins).
#     tf.keras.layers.Input(shape=(20,)),
    
#     # Hidden Layer 1: A standard layer with 16 neurons. 'relu' is a common activation function.
#     tf.keras.layers.Dense(16, activation='relu'),
    
#     # Hidden Layer 2: Another layer to help the model learn more complex patterns.
#     tf.keras.layers.Dense(8, activation='relu'),
    
#     # Output Layer: It has 1 neuron because we have one output (0 or 1).
#     # 'sigmoid' activation squashes the output to a probability between 0 and 1.
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# # --- 4. Compile the Model ---
# # This step configures the model for training.
# model.compile(
#     optimizer='adam',  # Adam is a popular and effective optimization algorithm.
#     loss='binary_crossentropy',  # This loss function is the standard for binary (0/1) classification.
#     metrics=['accuracy']  # We want to monitor the accuracy during training.
# )

# # Print a summary of the model's architecture
# model.summary()

# # --- 5. Train the Model ---
# print("\nStarting model training...")
# history = model.fit(
#     X_train_scaled,
#     y_train,
#     epochs=50,  # An epoch is one full pass through the entire training dataset.
#     batch_size=16, # The model will update its weights after seeing 16 samples.
#     validation_data=(X_test_scaled, y_test),
#     verbose=2 # Shows one line per epoch.
# )
# print("Training complete.")

# # --- 6. Evaluate the Model ---
# print("\nEvaluating model performance on the test set...")
# loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

# print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
# print(f"Test Loss: {loss:.4f}")

# # --- 7. Generate and Display Confusion Matrix ---
# print("\nGenerating confusion matrix...")
# # Get model predictions for the entire test set
# y_pred_proba = model.predict(X_test_scaled)
# # Convert probabilities to class labels (0 or 1)
# y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# # Calculate the confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# # Extract TP, TN, FP, FN
# # Note: Class 0 (-1) is considered 'Negative', Class 1 (1) is 'Positive'
# tn, fp, fn, tp = cm.ravel()

# print("\n--- Confusion Matrix ---")
# print(f"                 Predicted -1 (0) | Predicted 1 (1)")
# print(f"--------------------------------------------------")
# print(f"Actual -1 (0)    | {tn:^16} | {fp:^15}")
# print(f"Actual 1 (1)     | {fn:^16} | {tp:^15}")
# print(f"--------------------------------------------------")
# print(f"True Negatives (TN):  {tn} (Correctly predicted -1)")
# print(f"False Positives (FP): {fp} (Incorrectly predicted 1)")
# print(f"False Negatives (FN): {fn} (Incorrectly predicted -1)")
# print(f"True Positives (TP):  {tp} (Correctly predicted 1)")
# print("--------------------------")

# # Optional: Visualize the confusion matrix as a heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted -1', 'Predicted 1'], yticklabels=['Actual -1', 'Actual 1'])
# plt.title('Confusion Matrix')
# plt.ylabel('Actual Label')
# plt.xlabel('Predicted Label')
# plt.show()


# # --- 8. Making Predictions (Example) ---
# # You can now use the trained model to predict the class for new, unseen data.
# print("\nExample prediction:")
# # Let's take the first sample from the test set
# sample_to_predict = np.expand_dims(X_test_scaled[0], axis=0)
# prediction = model.predict(sample_to_predict)

# # The output of the sigmoid is a probability. If it's > 0.5, we classify as 1, otherwise 0.
# predicted_class = 1 if prediction[0][0] > 0.5 else 0
# actual_class = y_test.iloc[0]

# # Remap 0 back to -1 for clarity
# predicted_marker = 1 if predicted_class == 1 else -1
# actual_marker = 1 if actual_class == 1 else -1

# print(f"Predicted Color Marker: {predicted_marker}")
# print(f"Actual Color Marker:    {actual_marker}")
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import tf2onnx

# # --- Configuration ---
# # Make sure these filenames match your other scripts
# Binned_Data_Filename = 'z_binned_data.csv'
# Onnx_Model_Filename = 'model.onnx'
# Scaler_Params_Filename = 'scaler_params.txt'

# try:
#     data = pd.read_csv(Binned_Data_Filename)
# except FileNotFoundError:
#     print(f"Error: The file '{Binned_Data_Filename}' was not found.")
#     print("Please run the 'process_intensity.py' script first to generate this file.")
#     exit()

# # --- 1. Prepare Data and Train Model (Identical to training script) ---
# print("Preparing data and training model...")
# features = data.loc[:, data.columns.str.startswith('bin_')]
# target = data['color_marker'].replace(-1, 0)

# X_train, X_test, y_train, y_test = train_test_split(
#     features, target, test_size=0.2, random_state=42, stratify=target
# )

# # --- 2. Create and Save the Scaler ---
# # The scaler MUST be fitted only on the training data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train) # Fit and transform training data

# # Save the scaler's mean and scale (standard deviation) to a file
# # These values are essential for the C++ application
# np.savetxt(Scaler_Params_Filename,
#            np.vstack([scaler.mean_, scaler.scale_]),
#            delimiter=',',
#            header='Line 1: Mean values\nLine 2: Scale (std dev) values',
#            fmt='%f')
# print(f"Scaler parameters saved to '{Scaler_Params_Filename}'")

# # --- 3. Build and Train the Model (ensure architecture matches) ---
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(features.shape[1],)), # Use shape from features
#     tf.keras.layers.Dense(16, activation='relu'),
#     tf.keras.layers.Dense(8, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.fit(
#     X_train_scaled,
#     y_train,
#     epochs=50,
#     batch_size=16,
#     validation_data=(scaler.transform(X_test), y_test), # Transform test data for validation
#     verbose=0 # Suppress training output for export script
# )
# print("Model training complete.")

# # --- 4. Convert and Save the Model to ONNX format ---

# # FIX: Wrap the trained Sequential model in a Functional model to solve the AttributeError.
# # This gives the model the 'output_names' attribute that tf2onnx expects.
# inputs = tf.keras.Input(shape=model.input_shape[1:], name="input_tensor")
# outputs = model(inputs)
# model_for_export = tf.keras.Model(inputs=inputs, outputs=outputs)

# # The input signature specifies the expected input shape and type for the ONNX model
# # This is crucial for the C++ ONNX Runtime to understand the model's input
# input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name='input_tensor')]

# # Use the tf2onnx library to perform the conversion
# # We pass the new wrapper model, 'model_for_export', to the converter.
# onnx_model, _ = tf2onnx.convert.from_keras(model_for_export, input_signature, opset=13)
# with open(Onnx_Model_Filename, "wb") as f:
#     f.write(onnx_model.SerializeToString())

# print(f"\nModel successfully converted and saved to '{Onnx_Model_Filename}'")
# print("You can now move 'model.onnx' and 'scaler_params.txt' to your ROS2 C++ package.")

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tf2onnx

# --- Configuration ---
# Make sure these filenames match your other scripts
Binned_Data_Filename = 'z_binned_data.csv'
Onnx_Model_Filename = 'model.onnx'
Scaler_Params_Filename = 'scaler_params.txt'

try:
    data = pd.read_csv(Binned_Data_Filename)
except FileNotFoundError:
    print(f"Error: The file '{Binned_Data_Filename}' was not found.")
    print("Please run the 'process_intensity.py' script first to generate this file.")
    exit()

# --- 1. Prepare Data and Train Model (Identical to training script) ---
print("Preparing data and training model...")
features = data.loc[:, data.columns.str.startswith('bin_')]
target = data['color_marker'].replace(-1, 0)

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)

# --- 2. Create and Save the Scaler ---
# The scaler MUST be fitted only on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit and transform training data

# Save the scaler's mean and scale (standard deviation) to a file
# These values are essential for the C++ application
np.savetxt(Scaler_Params_Filename,
           np.vstack([scaler.mean_, scaler.scale_]),
           delimiter=',',
           header='Line 1: Mean values\nLine 2: Scale (std dev) values',
           fmt='%f')
print(f"Scaler parameters saved to '{Scaler_Params_Filename}'")

# --- 3. Build and Train the Model (ensure architecture matches) ---
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(features.shape[1],)), # Use shape from features
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nStarting model training...")
model.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=16,
    validation_data=(scaler.transform(X_test), y_test), # Transform test data for validation
    verbose=2 # Show one line of output per epoch during training
)
print("Model training complete.")

# --- 4. Convert and Save the Model to ONNX format ---

# FIX: Wrap the trained Sequential model in a Functional model to solve the AttributeError.
# This gives the model the 'output_names' attribute that tf2onnx expects.
inputs = tf.keras.Input(shape=model.input_shape[1:], name="input_tensor")
outputs = model(inputs)
model_for_export = tf.keras.Model(inputs=inputs, outputs=outputs)

# The input signature specifies the expected input shape and type for the ONNX model
# This is crucial for the C++ ONNX Runtime to understand the model's input
input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name='input_tensor')]

# Use the tf2onnx library to perform the conversion
# We pass the new wrapper model, 'model_for_export', to the converter.
onnx_model, _ = tf2onnx.convert.from_keras(model_for_export, input_signature, opset=13)
with open(Onnx_Model_Filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"\nModel successfully converted and saved to '{Onnx_Model_Filename}'")
print("You can now move 'model.onnx' and 'scaler_params.txt' to your ROS2 C++ package.")

