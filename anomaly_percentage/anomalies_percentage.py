import pandas as pd
import numpy as np
import sys
from pathlib import Path
import kagglehub

WINDOW_SIZES = [10, 64, 128, 256]

def main():
	# Κατέβασμα dataset (αν δεν υπάρχει)
	path = kagglehub.dataset_download("mgusat/smd-onmiad")
	base_path_labels = Path(path) / "ServerMachineDataset" / "test_label"

	# Βρες όλα τα αρχεία labels δυναμικά
	label_files = sorted([f for f in base_path_labels.iterdir() if f.is_file()])

	results = []
	for label_file in label_files:
		labels = pd.read_csv(label_file, header=None).values.flatten()
		
		# Χωρίς alignment: όλα τα labels
		y_true_without = labels
		anomaly_percentage_without = np.sum(y_true_without) / len(y_true_without)
		print(f"{label_file.stem}: Anomaly percentage (χωρίς alignment) = {anomaly_percentage_without:.4f} ({np.sum(y_true_without)}/{len(y_true_without)})")
		
		# Με alignment για διάφορα window sizes
		for window_size in WINDOW_SIZES:
			if window_size - 1 < len(labels):
				y_true_with = labels[window_size-1:]
				anomaly_percentage_with = np.sum(y_true_with) / len(y_true_with)
				print(f"{label_file.stem}: Anomaly percentage (με alignment, window_size={window_size}) = {anomaly_percentage_with:.4f} ({np.sum(y_true_with)}/{len(y_true_with)})")
			else:
				anomaly_percentage_with = None
				print(f"{label_file.stem}: Window size {window_size} είναι μεγαλύτερο από τα διαθέσιμα labels")
		
		row = {
			"server": label_file.stem,
			"without_alignment_percentage": anomaly_percentage_without,
			"without_alignment_anomalies": int(np.sum(y_true_without)),
			"without_alignment_total": int(len(y_true_without))
		}
		
		# Προσθήκη αποτελεσμάτων για κάθε window size
		for window_size in WINDOW_SIZES:
			if window_size - 1 < len(labels):
				y_true_with = labels[window_size-1:]
				anomaly_percentage_with = np.sum(y_true_with) / len(y_true_with)
				row[f"with_alignment_ws{window_size}_percentage"] = anomaly_percentage_with
				row[f"with_alignment_ws{window_size}_anomalies"] = int(np.sum(y_true_with))
				row[f"with_alignment_ws{window_size}_total"] = int(len(y_true_with))
			else:
				row[f"with_alignment_ws{window_size}_percentage"] = None
				row[f"with_alignment_ws{window_size}_anomalies"] = None
				row[f"with_alignment_ws{window_size}_total"] = None
		
		results.append(row)

	# Αποθήκευση σε κοινό CSV
	df = pd.DataFrame(results)
	df.to_csv("anomaly_percentages_combined.csv", index=False, encoding="utf-8")
	print(f"\nΑποτελέσματα αποθηκεύτηκαν σε: anomaly_percentages_combined.csv")

if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		print(f"An error occurred: {e}")
	except KeyboardInterrupt:
		print("Process interrupted by user.")
		sys.exit(0)
		