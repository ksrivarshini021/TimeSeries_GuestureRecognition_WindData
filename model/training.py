import os
import glob
from pathlib import Path
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold

# Wrist & Arm Base (2 points): Wrist Root, forearm Stub
# Thumb (5 points): Thumb Tip, Thumb 0, Thumb 1, Thumb 2, Thumb 3
# Index Finger (4 points): Index Tip, Index 1, Index 2, Index 3
# Middle Finger (4 points): Middle Tip, Middle 1, Middle 2, Middle 3
# Ring Finger (4 points): Ring Tip, Ring 1, Ring 2, Ring 3
# Pinky Finger (5 points): Pinky Tip, Pinky 0, Pinky 1, Pinky 2, Pinky 3
# 2 + 5 + 4 + 4 + 4 + 5 = 24 tracking points total
# total feature = 24 * 6 = 144

DATA_DIR = Path("./gesture_data(cleaned)")

class GestureDataLoader:
    def __init__(self, data_root_path, window_size=30, step_size=15):
        # config
        self.data_root = Path(data_root_path)
        self.window_size = window_size
        self.step_size = step_size
        
        # List of all 24 tracking joints on the hand
        self.joints = [
            "Wrist Root", "forearm Stub",
            "Thumb Tip", "Thumb 0", "Thumb 1", "Thumb 2", "Thumb 3",
            "Index Tip", "Index 1", "Index 2", "Index 3",
            "Middle Tip", "Middle 1", "Middle 2", "Middle 3",
            "Ring Tip", "Rign 1", "Ring 2", "Ring 3",  # right
            "Pinky Tip", "Pinky 0", "Pinky 1", "Pinky 2", "Pinky 3"
        ]
        
        # Automatically build our target list of 48 columns (24 Location and 24 Rotation)
        self.target_columns = []
        for joint in self.joints:
            self.target_columns.append(f"L_{joint}_C")  # Local Positions (X, Y, Z)
            self.target_columns.append(f"R_{joint}_C")  # Local Rotations (Pitch, Yaw, Roll)

    def clean_cells(self, cell_value):
        # extracts floating numbers
        if pd.isna(cell_value) or str(cell_value).strip() == "":
            return [0.0, 0.0, 0.0]
        
        found_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(cell_value))
        
        if len(found_numbers) >= 3:
            return [float(found_numbers[0]), float(found_numbers[1]), float(found_numbers[2])]
        return [0.0, 0.0, 0.0]

    def read_single_csv(self, file_path):
        # turns csv into numerical arrays
        df = pd.read_csv(file_path, skiprows=1, header=0)
        df.columns = [name.strip() for name in df.columns]
        
        all_rows_numerical = []
        for _, row in df.iterrows():
            single_frame_data = []
            for col in self.target_columns:
                if col in df.columns:
                    numbers = self.clean_cells(row[col])
                    single_frame_data.extend(numbers)
                else:
                    single_frame_data.extend([0.0, 0.0, 0.0])
            all_rows_numerical.append(single_frame_data)
            
        return np.array(all_rows_numerical, dtype=np.float32)

    def load_dataset(self):
        # Parses filenames, maps Participant -> Referent, and chops data into sliding windows.
        X_windows = []   
        y_labels = []    
        user_groups = [] 
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"Could not find the directory: {self.data_root}")

        for csv_file in sorted(self.data_root.glob("**/*.csv")):
            file_name = csv_file.name  
            
            match = re.search(r'(P\d+)_Rf(\d+)', file_name, re.IGNORECASE)
            if not match:
                print(f"Skipping file (name format doesn't match P#_Rf#): {file_name}")
                continue
                
            participant_name = match.group(1).upper()  
            gesture_label = int(match.group(2))  
            
            # 1. Read and convert the file data from text strings to numbers
            file_data = self.read_single_csv(csv_file)
            total_frames = len(file_data)
            
            # 2. Chop the sequence into sliding windows
            if total_frames >= self.window_size:
                for start in range(0, total_frames - self.window_size + 1, self.step_size):
                    end = start + self.window_size
                    chunk = file_data[start:end]
                    
                    X_windows.append(chunk)
                    y_labels.append(gesture_label)
                    user_groups.append(participant_name)
                            
        return np.array(X_windows), np.array(y_labels), np.array(user_groups)


# execution and cross validation
if __name__ == "__main__":
    # Initialize the pipeline
    data_pipeline = GestureDataLoader(data_root_path=DATA_DIR, window_size=30, step_size=15)
    
    print("--- PIPELINE DATA INITIALIZATION ---")
    print(f"Checking target directory: {data_pipeline.data_root.resolve()}")
    
    # FIXED: Changed to recursive look for checking file list
    all_files = list(data_pipeline.data_root.glob("**/*.csv"))
    print(f"Total CSV files found across all subfolders: {len(all_files)}")
    
    # Load the dataset
    X, y, groups = data_pipeline.load_dataset()
    
    print(f"\n--- DATA LOAD READOUT ---")
    print(f"Unique participants found in dataset: {np.unique(groups)}")
    print(f"Total unique participant count: {len(np.unique(groups))}")
    print(f"X tensor shape (Samples, Window Frames, Features): {X.shape}")
    print(f"y labels shape: {y.shape}")

    # Cross-validation splits (This will now run safely with all your participants)
    print(f"\n--- STARTING CROSS-VALIDATION ---")
    cross_validator = GroupKFold(n_splits=5)
    for fold, (train_indices, test_indices) in enumerate(cross_validator.split(X, y, groups=groups)):
        print(f"\n--- FOLD {fold + 1} ---")
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
        print(f"Testing on unseen participants: {np.unique(groups[test_indices])}")
        
        # Machine Leanring code:
        # my_lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test)) 
        