import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("point_level_data.csv")

# Extract z values
z_values = df["z_coordinate"]

# Plot histogram
plt.figure(figsize=(10,6))
plt.hist(z_values, bins=5000, edgecolor="black")  # adjust bins as needed
plt.title("Frequency Distribution of z_coordinate")
plt.xlabel("z_coordinate")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
