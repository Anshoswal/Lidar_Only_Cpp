import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# --- MODIFICATION: Import RandomForestClassifier ---
from sklearn.ensemble import RandomForestClassifier
import onnx
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType

# --- Configuration ---
INPUT_CSV = 'z_binned_data.csv'
MODEL_OUTPUT_ONNX = 'cone_model.onnx'
NUM_BINS = 20

def train_and_export_model():
    """
    Loads binned data, trains a Random Forest classifier, evaluates it,
    and exports the model to the ONNX format for C++ inference.
    """
    print(f"--- Starting Model Training ---")

    # 1. Load the dataset
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"Successfully loaded '{INPUT_CSV}' with {len(df)} samples.")
    except FileNotFoundError:
        print(f"Error: The data file '{INPUT_CSV}' was not found.")
        print("Please run the z_binner_script.py first to generate this file.")
        return

    # 2. Prepare the data
    feature_cols = [f'bin_{i+1}' for i in range(NUM_BINS)]
    X = df[feature_cols]
    y = df['color_marker']
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")

    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # 4. --- MODIFICATION: Train the Random Forest model ---
    print("\n--- Training Random Forest Classifier ---")
    # n_estimators is the number of trees in the forest.
    model = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 5. Evaluate the model on the test set
    print("\n--- Evaluating Model Performance ---")
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Type 0 (Yellow)', 'Type 1 (Blue)']))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("[[True Neg, False Pos]]")
    print("[[False Neg, True Pos]]")

    # 6. Export the model to ONNX format
    print(f"\n--- Exporting Model to ONNX ---")
    initial_type = [('float_input', FloatTensorType([None, NUM_BINS]))]
    
    # This conversion is extremely reliable and should work without issue.
    onnx_model = skl2onnx.convert_sklearn(model, initial_types=initial_type)
    
    with open(MODEL_OUTPUT_ONNX, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    print(f"Success! Model has been saved to '{MODEL_OUTPUT_ONNX}'.")
    print("This file is now ready to be used by the C++ application.")

if __name__ == '__main__':
    train_and_export_model()

