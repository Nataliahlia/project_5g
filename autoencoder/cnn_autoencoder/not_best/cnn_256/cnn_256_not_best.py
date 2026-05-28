import sys
import pandas as pd
import numpy as np
import kagglehub
import json
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from pathlib import Path
import keras
from keras import layers
from sklearn.preprocessing import StandardScaler, RobustScaler

WINDOW_SIZE = 256
# Function to load and preprocess the data for a specific server
def get_data_for_server(train_file, test_file, label_file):
    # Read train data
    train_data = pd.read_csv(train_file, header=None).values
    
    # Read test data and labels
    test_data = pd.read_csv(test_file, header=None).values
    labels = pd.read_csv(label_file, header=None).values.flatten()
    
    # Remove the first (WINDOW_SIZE-1) labels to align with sliding windows
    y_true = labels[WINDOW_SIZE-1:]
    
    # Perform Standard Scaling on the training data
    scaler = StandardScaler()
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_test_data = scaler.transform(test_data)
    
    return scaled_train_data, scaled_test_data, y_true

# Function to create sliding windows from the data
def create_sliding_windows(data, window_size=WINDOW_SIZE):
    num_samples, num_features = data.shape
    num_windows = num_samples - window_size + 1
    
    windows = []
    for i in range(num_windows):
        window = data[i : i + window_size]
        windows.append(window)
        
    return np.array(windows)

# Function to build the CNN Autoencoder model
def build_cnn_autoencoder_model(input_shape):
    # Encoder
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(filters=32, kernel_size=7, padding="same", activation="relu")(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(filters=16, kernel_size=7, padding="same", activation="relu")(x)
    
    # Bottleneck - compress the temporal dimension
    x = layers.Conv1D(filters=16, kernel_size=3, padding="same", activation="relu")(x)
    
    # Decoder
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1DTranspose(filters=16, kernel_size=7, padding="same", activation="relu")(x)
    x = layers.Conv1DTranspose(filters=32, kernel_size=7, padding="same", activation="relu")(x)
    x = layers.Conv1DTranspose(filters=input_shape[1], kernel_size=7, padding="same")(x)
    
    # Create the autoencoder model
    autoencoder = keras.Model(inputs, x)
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    
    return autoencoder

# Function to calculate Precision, Recall, and F1-score with Point Adjustment
def get_metrics_with_pa(y_true, y_pred):
    y_pred_pa = y_pred.copy()
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
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    p_pa, r_pa, f1_pa, _ = precision_recall_fscore_support(y_true, y_pred_pa, average='binary', zero_division=0)
    conf_matrix_pa = confusion_matrix(y_true, y_pred_pa)

    return (p, r, f1, conf_matrix), (p_pa, r_pa, f1_pa, conf_matrix_pa)

# Function to average metrics across multiple servers
def average_metrics(metrics_list):
    avg_precision = np.mean([m[0] for m in metrics_list])
    avg_recall = np.mean([m[1] for m in metrics_list])
    avg_f1 = np.mean([m[2] for m in metrics_list])
    total_conf_matrix = np.sum([m[3] for m in metrics_list], axis=0)
    
    return avg_precision, avg_recall, avg_f1, total_conf_matrix

# Function to create and save plots
def create_plots(server_name, test_mse_loss, y_true, y_pred, threshold, plots_dir):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Reconstruction Error with Threshold
    ax1 = axes[0]
    ax1.plot(test_mse_loss, label='reconstruction error', color='blue', linewidth=1)
    ax1.axhline(y=threshold, color='green', linestyle='-', linewidth=2, label=f'threshold={threshold:.4f}')
    anomalies = np.where(y_pred == 1)[0]
    if len(anomalies) > 0:
        ax1.scatter(anomalies, test_mse_loss[anomalies], color='red', s=30, alpha=0.8, zorder=5)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Ground Truth Labels + Detected Anomalies
    ax2 = axes[1]
    ax2.plot(y_true, label='label', color='gold', linewidth=2)
    anomalies_pred = np.where(y_pred == 1)[0]
    if len(anomalies_pred) > 0:
        ax2.scatter(anomalies_pred, y_pred[anomalies_pred], color='red', s=30, alpha=0.8, zorder=5)
    ax2.set_ylim(-0.05, 1.1)
    ax2.legend(loc='center right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_filename = plots_dir / f"{server_name}_anomaly_detection.png"
    plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Γράφημα αποθηκεύτηκε: {plot_filename}")

def analyze_anomaly(sample_idx, test_windows, x_test_pred, server_name, plots_dir, scaled_test_data=None, feature_names=None, context_size=200, top_n=5):
    # Get the number of features and window size from the test windows shape
    n_features = test_windows.shape[2]
    window_size = test_windows.shape[1]
    # If feature names are not provided, create generic names like F0, F1, ..., Fn
    if feature_names is None:
        feature_names = [f"F{i}" for i in range(n_features)]

    # Per-feature MSE averaged over the window timesteps -> shape (n_features,)
    # Calculate the mean squared error for each feature across the window and then average it to get a single error value per feature
    window_true = test_windows[sample_idx]
    window_pred = x_test_pred[sample_idx]
    per_feature_error = np.mean(np.square(window_pred - window_true), axis=0)
    mean_error = per_feature_error.mean()

    # Sort all features by error descending
    sorted_idx = np.argsort(per_feature_error)[::-1]
    top_idx = sorted_idx[:top_n]

    fig = plt.figure(figsize=(14, 4 + top_n * 3))

    # Bar plot of per-feature errors to identify which features contributed most to the anomaly
    ax_bar = plt.subplot(top_n + 1, 1, 1)
    bar_colors = ['crimson' if i < top_n else 'steelblue' for i in range(n_features)]
    ax_bar.barh(np.arange(n_features), per_feature_error[sorted_idx], color=bar_colors)
    ax_bar.set_yticks(np.arange(n_features))
    ax_bar.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=7)
    ax_bar.invert_yaxis()
    ax_bar.axvline(mean_error, color='orange', linestyle='--', linewidth=1.5, label=f'mean error = {mean_error:.3f}')
    ax_bar.set_xlabel('Mean Reconstruction Error')
    ax_bar.set_title(f'{server_name} Feature Errors')
    ax_bar.legend(fontsize=9, loc='lower right')
    ax_bar.grid(True, alpha=0.3, axis='x')

    # Time series plots for the top contributing features around the detected anomaly
    if scaled_test_data is not None:
        n_test = len(scaled_test_data) # total number of timesteps in the test data
        ctx_start = max(0, sample_idx - context_size) # start of the context window (before the anomaly)
        ctx_end   = min(n_test, sample_idx + window_size + context_size) # end of the context window (after the anomaly)
        timesteps = np.arange(ctx_start, ctx_end) # timesteps for plotting the context around the anomaly
        win_start = sample_idx  # start of the anomaly window
        win_end   = min(sample_idx + window_size, n_test) # end of the anomaly window (based on the window size and total test data length)

        for rank, feat_idx in enumerate(top_idx):
            ax = plt.subplot(top_n + 1, 1, rank + 2)
            feat_vals = scaled_test_data[ctx_start:ctx_end, feat_idx] # values of the current feature in the context window

            ax.plot(timesteps, feat_vals, color='green', linewidth=1, label=f'Feature: {feature_names[feat_idx]}')
            ax.axvspan(win_start, win_end, alpha=0.25, color='gray', label='Region around detected anomaly')

            # Red bar at the bottom marking the anomaly window
            y_min, y_max = feat_vals.min(), feat_vals.max()
            y_range = y_max - y_min if y_max != y_min else 1.0
            red_y = y_min - 0.08 * y_range
            ax.plot([win_start, win_end], [red_y, red_y], color='red', linewidth=5, solid_capstyle='butt', clip_on=False)
            ax.set_ylim(bottom=red_y - 0.02 * y_range)

            ax.set_title(f'Plot of {feature_names[feat_idx]}')
            ax.set_ylabel(feature_names[feat_idx])
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(ctx_start, ctx_end)
            if rank < top_n - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('Timesteps')

    plt.tight_layout()
    plot_filename = plots_dir / f"{server_name}_rca_idx{sample_idx}.png"
    plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"RCA γράφημα αποθηκεύτηκε: {plot_filename}")
    print(f"Top-{top_n} features: {[(feature_names[i], f'{per_feature_error[i]:.4f}') for i in top_idx]}")
    
def normalize_scores_simple(scores, window=200):
    scores = np.array(scores)
    norm_scores = np.zeros_like(scores)

    for i in range(len(scores)):
        start = max(0, i - window)
        local_window = scores[start:i+1]

        mean = local_window.mean()
        std = local_window.std() + 1e-8

        norm_scores[i] = (scores[i] - mean) / std

    return norm_scores

if __name__ == "__main__":
    try: 
        # Create output directory if it doesn't exist
        output_dir = Path("results_cnn_256")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create plots directory
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Load the data - download dataset from Kaggle
        path = kagglehub.dataset_download("mgusat/smd-onmiad")
        
        # Create paths to train, test, and label directories
        base_path_train = Path(path) / "ServerMachineDataset" / "train"
        base_path_test = Path(path) / "ServerMachineDataset" / "test"   
        base_path_labels = Path(path) / "ServerMachineDataset" / "test_label"

        # For each server machine, perform separate training and anomaly detection
        train_files = sorted([f for f in base_path_train.iterdir() if f.is_file()])
        metrics_list = []
        metrics_pa_list = []
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
            # Create sliding windows from the scaled data
            print("Δημιουργία sliding windows...")
            train_windows = create_sliding_windows(scaled_train_data, window_size=WINDOW_SIZE)
            test_windows = create_sliding_windows(scaled_test_data, window_size=WINDOW_SIZE)
            print(f"Shape of train windows: {train_windows.shape}")
            print(f"Shape of test windows: {test_windows.shape}")
            
            # Build the autoencoder model
            print("Κατασκευή CNN Autoencoder μοντέλου...")
            input_shape = (train_windows.shape[1], train_windows.shape[2])
            model = build_cnn_autoencoder_model(input_shape)
            # Train the autoencoder
            print("Εκπαίδευση Autoencoder...")
            history = model.fit(
                train_windows,
                train_windows,
                epochs=50,
                batch_size=128,
                validation_split=0.1,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=0)
                ],
                verbose=0
            )
            print(f"Εκπαίδευση ολοκληρώθηκε. Final training loss: {history.history['loss'][-1]:.6f}")
            
            # Get training reconstruction error (mse) for threshold calculation
            print("Υπολογισμός reconstruction error για train data...")
            x_train_pred = model.predict(train_windows, verbose=0)
            train_mse_loss_before = np.mean(np.square(x_train_pred - train_windows), axis=(1, 2))
            train_mse_loss = normalize_scores_simple(train_mse_loss_before)
            # Get test reconstruction error (mse)
            print("Υπολογισμός reconstruction error για test data...")
            x_test_pred = model.predict(test_windows, verbose=0)
            test_mse_loss_before = np.mean(np.square(x_test_pred - test_windows), axis=(1, 2))
            test_mse_loss = normalize_scores_simple(test_mse_loss_before)

            # Use a static threshold at the 95th percentile of test reconstruction error
            static_thresh = np.percentile(test_mse_loss, 95)
            print(f"Χρήση στατικού threshold (95th percentile): {static_thresh}")
            y_pred = (test_mse_loss > static_thresh).astype(int)
            metrics, metrics_pa = get_metrics_with_pa(y_true, y_pred)

            # Create and save plots for this server
            print("Δημιουργία γραφημάτων...")
            create_plots(server_name, test_mse_loss, y_true, y_pred, static_thresh, plots_dir)
            
            # Root Cause Analysis on the first detected anomaly
            anomaly_indices = np.where(y_pred == 1)[0]
            if len(anomaly_indices) > 0:
                rca_idx = int(anomaly_indices[0])
                print(f"RCA για ανωμαλία στο test index {rca_idx}...")
                analyze_anomaly(rca_idx, test_windows, x_test_pred, server_name, plots_dir, scaled_test_data=scaled_test_data)

            # Store the metrics for this server
            metrics_list.append(metrics)
            metrics_pa_list.append(metrics_pa)
            # Store the results for this server
            results_json["servers"][server_name] = {
                "before_pa": {"precision": float(metrics[0]), "recall": float(metrics[1]), "f1": float(metrics[2]), "confusion_matrix": metrics[3].tolist()},
                "after_pa": {"precision": float(metrics_pa[0]), "recall": float(metrics_pa[1]), "f1": float(metrics_pa[2]), "confusion_matrix": metrics_pa[3].tolist()}
            }
            print(f"Threshold: {static_thresh}")
            print(f"Πριν PA: Precision={metrics[0]:.4f}, Recall={metrics[1]:.4f}, F1={metrics[2]:.4f}")
            print(f"Μετά PA: Precision={metrics_pa[0]:.4f}, Recall={metrics_pa[1]:.4f}, F1={metrics_pa[2]:.4f}")
            
        # Calculate and print the overall metrics across all servers
        print(f"\n{'='*80}")
        print(f"Overall Metrics (across {server_count} servers):")
        print(f"{'='*80}")
        overall_metrics = average_metrics(metrics_list)
        overall_metrics_pa = average_metrics(metrics_pa_list)
        print(f"Πριν PA: Precision={overall_metrics[0]:.4f}, Recall={overall_metrics[1]:.4f}, F1={overall_metrics[2]:.4f}")
        print(f"Μετά PA: Precision={overall_metrics_pa[0]:.4f}, Recall={overall_metrics_pa[1]:.4f}, F1={overall_metrics_pa[2]:.4f}")
        
        results_json["overall"] = {
            "before_pa": {"precision": float(overall_metrics[0]), "recall": float(overall_metrics[1]), "f1": float(overall_metrics[2]), "confusion_matrix": overall_metrics[3].tolist()},
            "after_pa": {"precision": float(overall_metrics_pa[0]), "recall": float(overall_metrics_pa[1]), "f1": float(overall_metrics_pa[2]), "confusion_matrix": overall_metrics_pa[3].tolist()},
            "total_servers": server_count
        }
        # Write the results to a JSON file
        with open(output_dir / "autoencoder_result_cnn_256.json", "w", encoding="utf-8") as f:
            json.dump(results_json, f, indent=4)
        print(f"\Τα αποτελέσματα αποθηκεύτηκαν στο: results/autoencoder_results_cnn_256.json")
        print(f"Γραφήματα αποθηκεύτηκαν στο: results/plots/")
        
    except Exception as e:
        print(f"Σφάλμα κατά την εκτέλεση: {e}")
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        print("Η εκτέλεση διακόπηκε από τον χρήστη.")
        sys.exit(0)