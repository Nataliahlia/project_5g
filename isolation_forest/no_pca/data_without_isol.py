import sys
import pandas as pd
import numpy as np
import kagglehub
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from pathlib import Path

WINDOW_SIZE = 10
N_ESTIMATORS = 100

# Function to load and preprocess the data for a specific server
def get_data_for_server(train_file, test_file, label_file):
    # Read train data
    train_data = pd.read_csv(train_file, header=None).values
    
    # Read test data and labels
    test_data = pd.read_csv(test_file, header=None).values
    labels = pd.read_csv(label_file, header=None).values.flatten()
    
    # No need to remove labels since we're not using sliding windows
    y_true = labels
    
    # Perform Standard Scaling on the training data
    scaler = StandardScaler()
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.transform(test_data)
    
    return scaled_train_data, scaled_test_data, y_true

# Function to perform anomaly detection using KNN and return the anomaly scores
def detect_anomalies_iforest(train_data, test_data, n_estimators=N_ESTIMATORS):
    print(f"Fitting Isolation Forest σε {len(train_data)} δείγματα...")
    # Perform the Isolation Forest algorithm on the training data to learn the normal patterns
    model = IsolationForest(n_estimators=n_estimators, contamination='auto', random_state=42)
    model.fit(train_data)
    
    print(f"Υπολογισμός scores για {len(test_data)} test δείγματα...")
    # Calculate the anomaly scores for the test data
    anomaly_scores = -model.decision_function(test_data)
    # Normalize the anomaly scores to the range [0, 1] for better interpretability and thresholding
    anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
    
    return anomaly_scores

# Function to calculate Precision, Recall, and F1-score with Point Adjustment
def get_metrics_with_pa(y_true, y_pred):
    y_pred_pa = y_pred.copy()   # Create a copy of the original predictions to apply Point Adjustment
    # Point Adjustment Logic
    anomaly_state = False
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            # Make the current point and all previous points in the segment 1 until we hit a 0
            for j in range(i, 0, -1):
                if y_true[j] == 0: break
                y_pred_pa[j] = 1
            # Make the current point and all subsequent points in the segment 1 until we hit a 0
            for j in range(i, len(y_true)):
                if y_true[j] == 0: break
                y_pred_pa[j] = 1
        elif y_true[i] == 0:
            anomaly_state = False
            
    # Calculate Precision, Recall, and F1-score for both original and PA-adjusted predictions
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    conf_matrix = confusion_matrix(y_true, y_pred)
    p_pa, r_pa, f1_pa, _ = precision_recall_fscore_support(y_true, y_pred_pa, average='binary')
    conf_matrix_pa = confusion_matrix(y_true, y_pred_pa)

    return (p, r, f1, conf_matrix), (p_pa, r_pa, f1_pa, conf_matrix_pa)

# Function to average metrics across multiple machines
def average_metrics(metrics_list):
    avg_precision = np.mean([m[0] for m in metrics_list])
    avg_recall = np.mean([m[1] for m in metrics_list])
    avg_f1 = np.mean([m[2] for m in metrics_list])
    total_conf_matrix = np.sum([m[3] for m in metrics_list], axis=0).astype(int)
    
    return avg_precision, avg_recall, avg_f1, total_conf_matrix

if __name__ == "__main__":
    try: 
        # Create output directory if it doesn't exist
        output_dir = Path("results")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load the data - download dataset from Kaggle (if not already downloaded)
        path = kagglehub.dataset_download("mgusat/smd-onmiad")
        # Create paths to train, test, and label directories
        base_path_train = Path(path) / "ServerMachineDataset" / "train"
        base_path_test = Path(path) / "ServerMachineDataset" / "test"   
        base_path_labels = Path(path) / "ServerMachineDataset" / "test_label"

        # For each server machine, perform separate training and anomaly detection
        # Dont access the machine ids directly, but find them dynamically from the file names in the train directory
        train_files = sorted([f for f in base_path_train.iterdir() if f.is_file()])
        metrics_list = []       # List to store metrics for each server before PA
        metrics_pa_list = []    # List to store metrics for each server after PA
        results_json = {"servers": {}, "overall": {}}
        server_count = 0
        
        for train_file in train_files:
            # Find corresponding test and label files
            test_file = base_path_test / train_file.name
            label_file = base_path_labels / train_file.name
            
            if not test_file.exists() or not label_file.exists():
                continue
                
            server_count += 1
            server_name = train_file.stem
            print(f"\n[{server_count}] Επεξεργασία για {server_name}")  
            
            # Preprocess the data for the specified server
            scaled_train_data, scaled_test_data, y_true = get_data_for_server(train_file, test_file, label_file)

            # Perform the Isolation Forest algorithm
            anomaly_scores = detect_anomalies_iforest(scaled_train_data, scaled_test_data, n_estimators=N_ESTIMATORS)

            # Use static threshold based on the distribution of anomaly scores (e.g., 99th percentile) to classify anomalies
            threshold = np.percentile(anomaly_scores, 99)  # Choose the 99th percentile as the threshold for anomaly detection
            y_pred = (anomaly_scores > threshold).astype(int)
            metrics, metrics_pa = get_metrics_with_pa(y_true, y_pred)
            # Store the metrics for this server in the lists for later averaging
            metrics_list.append(metrics)
            metrics_pa_list.append(metrics_pa)
            
            # Store the results for this server in the JSON structure
            results_json["servers"][server_name] = {
                "before_pa": {"precision": float(metrics[0]), "recall": float(metrics[1]), "f1": float(metrics[2]), "confusion_matrix": metrics[3].tolist()},
                "after_pa": { "precision": float(metrics_pa[0]), "recall": float(metrics_pa[1]), "f1": float(metrics_pa[2]), "confusion_matrix": metrics_pa[3].tolist()}
            }

            print(f"Πριν PA: Precision={metrics[0]:.4f}, Recall={metrics[1]:.4f}, F1={metrics[2]:.4f}, Confusion Matrix:\n{metrics[3]}")
            print(f"Μετά PA: Precision={metrics_pa[0]:.4f}, Recall={metrics_pa[1]:.4f}, F1={metrics_pa[2]:.4f}, Confusion Matrix:\n{metrics_pa[3]}")
            
        # Calculate and print the overall metrics across all servers
        print(f"\n{'='*80}")
        print(f"Overall Metrics (across {server_count} servers):")
        print(f"{'='*80}")
        overall_metrics = average_metrics(metrics_list)
        overall_metrics_pa = average_metrics(metrics_pa_list)
        print(f"Πριν PA: Precision={overall_metrics[0]:.4f}, Recall={overall_metrics[1]:.4f}, F1={overall_metrics[2]:.4f}, Confusion Matrix:\n{overall_metrics[3]}")
        print(f"Μετά PA: Precision={overall_metrics_pa[0]:.4f}, Recall={overall_metrics_pa[1]:.4f}, F1={overall_metrics_pa[2]:.4f}, Confusion Matrix:\n{overall_metrics_pa[3]}")
        results_json["overall"] = {
                "before_pa": {"precision": float(overall_metrics[0]), "recall": float(overall_metrics[1]), "f1": float(overall_metrics[2]), "confusion_matrix": overall_metrics[3].tolist()},
                "after_pa": {"precision": float(overall_metrics_pa[0]), "recall": float(overall_metrics_pa[1]), "f1": float(overall_metrics_pa[2]), "confusion_matrix": overall_metrics_pa[3].tolist()},
                "total_servers": server_count
            }
        # Write the results to a JSON file in the output directory
        with open(output_dir / "if_without_windows.json", "w", encoding="utf-8") as f:
            json.dump(results_json, f, indent=4)
    except Exception as e:
        print(f"Σφάλμα κατά την εκτέλεση: {e}")
    except KeyboardInterrupt:
        print("Η εκτέλεση διακόπηκε από τον χρήστη.")
        sys.exit(0)