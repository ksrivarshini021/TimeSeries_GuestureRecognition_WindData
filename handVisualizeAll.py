import pandas as pd
import numpy as np
import re
import os
import json
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone

FILE_PATH = "P1_Rf11.csv"
OUTPUT_DIR = "hand_plots_html"
if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

# Timezone config
MST_ZONE = timezone(timedelta(hours=-7))

def extract_numbers(val):
    if pd.isna(val) or str(val).strip() == "": return None
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
    return [float(nums[0]), float(nums[1]), float(nums[2])] if len(nums) >= 3 else None

def euler_to_vector(angles, segment_length):
    p, y, r = np.radians(angles[0]), np.radians(angles[1]), np.radians(angles[2])
    vx = np.cos(p) * np.cos(y)
    vy = np.cos(p) * np.sin(y)
    vz = np.sin(p)
    return np.array([vx, vy, vz]) * segment_length

# --- LOAD DATA ---
df = pd.read_csv(FILE_PATH, header=0, index_col=False)

# Clean headers strictly (Remove spaces, convert to lowercase)
cleaned_cols = []
for c in df.columns:
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '', str(c).replace(' ', '')).lower()
    cleaned_cols.append(clean_name)

# Dedup columns safely
final_cols = []
col_counts = {}
for col in cleaned_cols:
    if col in col_counts:
        col_counts[col] += 1
        final_cols.append(f"{col}_{col_counts[col]}")
    else:
        col_counts[col] = 0
        final_cols.append(col)

df.columns = final_cols

print(f"Total data rows found to process: {len(df)}")

# Export timestamps file needed for your HUD dashboard layout
ts_list = df['timestamp'].astype(str).tolist()
with open(os.path.join(OUTPUT_DIR, "timestamps.js"), "w") as f:
    f.write(f"const TIMESTAMPS = {json.dumps(ts_list)};\n")

# Process and render individual frames
for i in range(len(df)):
    row = df.iloc[i]
    
    finger_defs = {
        "Thumb":  {"cols": ["r_thumb1_c", "r_thumb2_c", "r_thumb3_c"], "len": 0.8},
        "Index":  {"cols": ["r_index1_c", "r_index2_c", "r_index3_c"], "len": 1.05},
        "Middle": {"cols": ["r_middle1_c", "r_middle2_c", "r_middle3_c"], "len": 1.15},
        "Ring":   {"cols": ["r_rign1_c", "r_ring2_c", "r_ring3_c"], "len": 1.05}, 
        "Pinky":  {"cols": ["r_pinky1_c", "r_pinky2_c", "r_pinky3_c"], "len": 0.85}
    }

    fig = go.Figure()
    
    # Base wrist joint anchor point
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers', 
        marker=dict(size=16, color='lightblue', line=dict(width=4, color='black')), name="Wrist"
    ))

    # Generate 3D paths for finger nodes
    for name, data in finger_defs.items():
        points = [[0, 0, 0]]
        curr = np.array([0.0, 0.0, 0.0])
        for c in data["cols"]:
            angles = extract_numbers(row.get(c))
            if angles:
                curr = curr + euler_to_vector(angles, data["len"])
                points.append(curr.tolist())
        
        pts = np.array(points)
        fig.add_trace(go.Scatter3d(
            x=-pts[:,0], y=pts[:,1], z=pts[:,2], 
            mode='lines+markers', line=dict(width=7), 
            marker=dict(size=6), name=name
        ))

    # --- CHOSEN VIEWING ANGLE CONFIGURATION ---
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-2, 2], backgroundcolor="rgb(240, 240, 240)", gridcolor="white"),
            yaxis=dict(range=[-1, 3], backgroundcolor="rgb(240, 240, 240)", gridcolor="white"),
            zaxis=dict(range=[-2, 2], backgroundcolor="rgb(240, 240, 240)", gridcolor="white"),
            
            # FIXED: Target perspective looking down down on hand gestures clearly
            camera=dict(
                eye=dict(x=-1.25, y=1.25, z=1.65),  # Elevated top-down isometric split
                up=dict(x=0, y=0, z=1),              # Fixed +Z vector vertical locking
                center=dict(x=0, y=0.5, z=0)         # Centers focus tracking down finger lengths
            ),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=0),  # Clean marginless styling for smooth iframe injection
        showlegend=True,
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.02)
    )

    # Save out as the structured frame file names your index dashboard explicitly calls
    fig.write_html(os.path.join(OUTPUT_DIR, f"frame_{i:04d}.html"), full_html=False, include_plotlyjs='cdn')

print(f"Done! Successfully generated all {len(df)} frames using the optimized viewing angle.")