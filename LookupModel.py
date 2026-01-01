import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import eig
from scipy.stats import linregress
import matplotlib.pyplot as plt

# ==========================================
# 1. Load Data
# ==========================================
print("Loading data...")
try:
    stations = pd.read_csv("Station.csv")
    attractions = pd.read_csv("Attraction.csv")
    routes = pd.read_csv("Network.csv")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print(
        "Please ensure Station.csv, Attraction.csv, and Network.csv are in the folder."
    )
    exit()

# ==========================================
# 2. Process Attractiveness (With Penalty)
# ==========================================
print("\n--- STEP 1: CALCULATING ATTRACTIVENESS ---")

# Handle missing 'Last_Mile_Difficulty' by defaulting to 1.0
if "Last_Mile_Difficulty" not in attractions.columns:
    attractions["Last_Mile_Difficulty"] = 1.0

# Calculate Attractiveness Score: (Reviews * Rating) / Penalty
attractions["Raw_Score"] = (
    attractions["Count_Reviews"] * attractions["Rating"]
) / attractions["Last_Mile_Difficulty"]
station_scores = attractions.groupby("Station_ID")["Raw_Score"].sum()

# Log Normalization and Scaling (1-10 range)
station_scores_normalized = np.log(station_scores + 1)
min_score = station_scores_normalized.min()
max_score = station_scores_normalized.max()
attractiveness_final = (
    (station_scores_normalized - min_score) / (max_score - min_score)
) * 9 + 1

# --- DEBUG PRINT: Attractiveness Table ---
attractiveness_table = pd.DataFrame(
    {
        "Raw_Sum_Score": station_scores,
        "Log_Normalized": station_scores_normalized,
        "Final_Score_1_to_10": attractiveness_final,
    }
).sort_values("Final_Score_1_to_10", ascending=False)

print("\n[TABLE 1] Attractiveness Scores (Before vs After Normalization):")
print(attractiveness_table.head(10))  # Print top 10 to keep it readable
print("... (showing top 10 only)")


# ==========================================
# 3. Least Squares Optimization of Cost Weight
# ==========================================
print("\n--- STEP 2: OPTIMIZATION (FINDING COST WEIGHT) ---")

# --- A. Mock Data Generation ---
TRUE_TIME_VALUE_OF_MONEY = 3.5  # The hidden value we want the model to find.
COST_RANGE = np.arange(1.0, 7.1, 0.5)  # Test conversion factors from 1 to 7.


def calculate_likelihood_and_cost(time_value_of_money):
    # Calculate Generalized Cost for this weight
    routes["GCost_Test"] = routes["Time"] + \
        (routes["Fare"] * time_value_of_money)
    G_test = nx.from_pandas_edgelist(
        routes, "From_ID", "To_ID", ["GCost_Test"])
    path_lengths_test = dict(
        nx.all_pairs_dijkstra_path_length(G_test, weight="GCost_Test")
    )

    data = []
    all_stations = station_scores.index.tolist()

    for i in all_stations:
        for j in all_stations:
            if i == j:
                continue  # Ignore self-loops for regression analysis

            attractiveness_j = attractiveness_final.get(j, 1.0)

            # Get shortest path cost
            cost_ij = path_lengths_test.get(i, {}).get(j, float("inf"))

            if cost_ij > 0 and cost_ij != float("inf"):
                likelihood_ratio = attractiveness_j / cost_ij
                data.append(
                    {
                        "From_ID": i,
                        "To_ID": j,
                        "GCost_Value": time_value_of_money,
                        "Attractiveness_J": attractiveness_j,
                        "Generalized_Cost": cost_ij,
                        "Likelihood_Ratio": likelihood_ratio,
                    }
                )
    return pd.DataFrame(data)


# Generate Mock Observation Data (The Y variable)
mock_data_df = calculate_likelihood_and_cost(TRUE_TIME_VALUE_OF_MONEY)
# Add some random noise
mock_data_df["Observed_Traffic"] = mock_data_df["Likelihood_Ratio"] * (
    1 + np.random.normal(0, 0.1, len(mock_data_df))
)

print(f"\n[TABLE 2] Mock Observed Traffic Data (Target for Regression):")
print(f"(Generating assuming True Weight = {
      TRUE_TIME_VALUE_OF_MONEY} mins/RM)")
print(
    mock_data_df[
        ["From_ID", "To_ID", "Generalized_Cost",
            "Likelihood_Ratio", "Observed_Traffic"]
    ].head()
)
print("Minimum Generalized Cost in Mock Data:",
      mock_data_df["Generalized_Cost"].min())

# --- B. Test all Cost Conversions using Least Squares ---
best_r_squared = -1.0
optimal_cost_conversion = 0.0

print(f"\nTesting {len(COST_RANGE)} candidate weights...")

for test_weight in COST_RANGE:
    # Calculate Likelihood Ratios (X variable) for the test weight
    test_data_df = calculate_likelihood_and_cost(test_weight)

    # Merge with Mock Observed Traffic (Y variable)
    merged_data = pd.merge(
        mock_data_df[["From_ID", "To_ID", "Observed_Traffic"]],
        test_data_df[["From_ID", "To_ID", "Likelihood_Ratio"]],
        on=["From_ID", "To_ID"],
    )

    # Perform Linear Regression (Least Squares)
    slope, intercept, r_value, p_value, std_err = linregress(
        merged_data["Likelihood_Ratio"], merged_data["Observed_Traffic"]
    )

    r_squared = r_value**2

    # Check if this test_weight gives a better fit
    if r_squared > best_r_squared:
        best_r_squared = r_squared
        optimal_cost_conversion = test_weight

    print(f"  Testing Weight: {
          test_weight:.1f} mins/RM | R-squared: {r_squared:.4f}")

print(
    f"\n>>> OPTIMIZATION RESULT: Best Weight found is {
        optimal_cost_conversion:.1f} mins/RM (R²={best_r_squared:.4f})"
)

# ==========================================
# 4. Run Final Markov Model
# ==========================================
FINAL_TIME_VALUE_OF_MONEY = optimal_cost_conversion
print(
    f"\n--- STEP 3: MARKOV CHAIN GENERATION (Using Weight {
        FINAL_TIME_VALUE_OF_MONEY
    }) ---"
)

# Recalculate Generalized Cost with the Optimal Weight
routes["Generalized_Cost"] = routes["Time"] + (
    routes["Fare"] * FINAL_TIME_VALUE_OF_MONEY
)

# Build the Graph using NetworkX
G = nx.from_pandas_edgelist(routes, "From_ID", "To_ID", ["Generalized_Cost"])

# Calculate shortest paths
path_lengths = dict(nx.all_pairs_dijkstra_path_length(
    G, weight="Generalized_Cost"))

# Rebuild Transition Matrix P using the optimal weights
all_stations = station_scores.index.tolist()
P = pd.DataFrame(0.0, index=all_stations, columns=all_stations)

for i in all_stations:
    for j in all_stations:
        attractiveness_j = attractiveness_final.get(j, 1.0)

        if i == j:
            likelihood = attractiveness_j / 4.0
        else:
            if i in path_lengths and j in path_lengths[i]:
                cost_ij = path_lengths[i][j]
                if cost_ij <= 0:
                    cost_ij = 0.1
                likelihood = attractiveness_j / cost_ij
            else:
                likelihood = 0.0
        P.loc[i, j] = likelihood

print("\n[TABLE 3] Raw Likelihood Matrix (Before Normalization):")
print("Rows = From, Columns = To. Values = Desire/Pull Strength.")
print(P.round(3))


# Normalize
P_normalized = P.div(P.sum(axis=1), axis=0)
P_normalized = P_normalized.fillna(0)

# Handle dead ends
for i in P_normalized.index:
    if P_normalized.loc[i].sum() == 0:
        P_normalized.loc[i, i] = 1.0

print("\n[TABLE 4] Normalized Transition Matrix (Probabilities):")
print("Rows now sum to 1.0. This is the Markov 'Rulebook'.")
print(P_normalized.round(3))

# ==========================================
# 5. Solve for Steady State
# ==========================================
print("\n--- STEP 4: SOLVING FOR STEADY STATE ---")
A = P_normalized.T.values
eigenvalues, eigenvectors = eig(A)
idx = np.argmin(np.abs(eigenvalues - 1))
steady_state_vector = np.real(eigenvectors[:, idx])
steady_state_vector = steady_state_vector / steady_state_vector.sum()

# Visualization of Final Ranking
final_ranking = pd.Series(steady_state_vector, index=all_stations).sort_values(
    ascending=False
)

# Map IDs to Names for display
try:
    id_to_name = stations.set_index("Station_ID")["Station_name"].to_dict()
    final_ranking.index = final_ranking.index.map(
        lambda x: f"{x} ({id_to_name.get(x, 'Unknown')})"
    )
except:
    pass

print("\n[TABLE 5] FINAL PREDICTION: Top Tourist Hubs")
print(final_ranking)

# Plot
top_10 = final_ranking
plt.figure(figsize=(12, 7))
top_10.plot(kind="bar", color="#4CAF50", edgecolor="black")
plt.title(
    f"Optimized Tourist Distribution (Weight: 1 RM = {
        FINAL_TIME_VALUE_OF_MONEY:.1f} min)",
    fontsize=14,
)
plt.ylabel("Steady State Probability", fontsize=12)
plt.xlabel("Station", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# --- VISUALIZATION 1: The Matrix Heatmap ---
plt.figure(figsize=(10, 8))
# We use log scale for color because some probabilities are very small vs very large
plt.imshow(P_normalized, cmap="viridis", aspect="auto")
plt.colorbar(label="Transition Probability")
plt.title("Transition Matrix Heatmap (Bright = High Probability)", fontsize=14)
plt.xlabel("To Station", fontsize=12)
plt.ylabel("From Station", fontsize=12)

# Set ticks to show station names (rotating them so they fit)
# limiting to top 20 for readability if there are too many
if len(all_stations) <= 20:
    tick_labels = [f"{s}" for s in all_stations]
    plt.xticks(range(len(all_stations)), tick_labels, rotation=90, fontsize=8)
    plt.yticks(range(len(all_stations)), tick_labels, fontsize=8)

    # Annotate each cell with the numeric value
    # Loop over data dimensions and create text annotations.
    for i in range(len(all_stations)):
        for j in range(len(all_stations)):
            val = P_normalized.iloc[i, j]
            # Set color to white for dark cells (low prob) and black for bright cells (high prob)
            # Threshold set to 0.5 for simple visibility check on Viridis map
            text_color = "white" if val < 0.5 else "black"
            plt.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

else:
    plt.xticks([])  # Hide labels if too messy
    plt.yticks([])
    print("Matrix too large to show numeric annotations on plot.")

plt.tight_layout()
plt.show()

comparison_df = pd.DataFrame(
    {
        "Station_ID": all_stations,
        "Attractiveness_Score": [attractiveness_final.get(s, 0) for s in all_stations],
        "Steady_State_Prob": steady_state_vector,
    }
).set_index("Station_ID")

# Map IDs to Names for labels
comparison_df["Station_Name"] = comparison_df.index.map(
    lambda x: id_to_name.get(x, x))

plt.figure(figsize=(12, 8))
x = comparison_df["Attractiveness_Score"]
y = comparison_df["Steady_State_Prob"]

# Scatter Plot
plt.scatter(
    x, y, color="purple", alpha=0.7, s=100, edgecolors="black", label="Stations"
)

# Trend Line (Linear Regression)
m, b = np.polyfit(x, y, 1)
plt.plot(x, m * x + b, color="red", linestyle="--",
         alpha=0.5, label="Trend Line")

# Label key points (Top 5 Prob + Top 5 Attraction)
top_prob = comparison_df.nlargest(14, "Steady_State_Prob").index
top_attr = comparison_df.nlargest(14, "Attractiveness_Score").index
labels_to_show = set(top_prob).union(set(top_attr))

for station_id in labels_to_show:
    row = comparison_df.loc[station_id]
    plt.annotate(
        row["Station_Name"],
        (row["Attractiveness_Score"], row["Steady_State_Prob"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

plt.title("Comparison: Intrinsic Attractiveness vs. Network Probability", fontsize=16)
plt.xlabel("Attractiveness Score (1-10)", fontsize=12)
plt.ylabel("Steady-State Probability (Calculated)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
