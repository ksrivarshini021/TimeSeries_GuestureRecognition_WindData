import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("P1/VR_RightHandData673.csv", skiprows=1)

df.columns = df.columns.str.strip()

subset = df.iloc[2462:3509].copy()

subset["time_diff"] = subset["Milliseconds"].diff()

subset = subset[subset["time_diff"] > 0]

# plot
plt.figure(figsize=(12, 4))
plt.plot(subset["time_diff"], linewidth=1)

plt.title("Time Differences (Rows 2462–3508)")
plt.xlabel("Frame Index")
plt.ylabel("Milliseconds Difference")
plt.grid()
plt.savefig("time_diff_plot.png", dpi=300)

plt.show()