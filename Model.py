import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import eig
from scipy.stats import linregress
import matplotlib.pyplot as plt

print("Loading data...")
stations = pd.read_csv("Station.csv")
attractions = pd.read_csv("Attraction.csv")
routes = pd.read_csv("Network.csv")

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

print("\n--- 3. OPTIMIZATION: Finding the Optimal Cost Weight ---")

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
                        "Likelihood_Ratio": likelihood_ratio,  # The X variable for regression
                    }
                )
    return pd.DataFrame(data)


# Generate Mock Observation Data (The Y variable)
# We assume the 'Observed Traffic' (Likelihood) is determined by the TRUE_TIME_VALUE_OF_MONEY
mock_data_df = calculate_likelihood_and_cost(TRUE_TIME_VALUE_OF_MONEY)
# Add some random noise to simulate real-world measurement error
mock_data_df["Observed_Traffic"] = mock_data_df["Likelihood_Ratio"] * (
    1 + np.random.normal(0, 0.1, len(mock_data_df))
)
print("Mock observed traffic data generated.")
print(mock_data_df)


# --- B. Test all Cost Conversions using Least Squares ---
best_r_squared = -1.0
optimal_cost_conversion = 0.0

print(
    f"Mock Data generated assuming True Weight = 1 RM is {
        TRUE_TIME_VALUE_OF_MONEY
    } minutes."
)
print(f"Testing {len(COST_RANGE)} candidate weights...")

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
    # Fit: Observed_Traffic ~ Likelihood_Ratio (for this specific test_weight)
    slope, intercept, r_value, p_value, std_err = linregress(
        merged_data["Likelihood_Ratio"], merged_data["Observed_Traffic"]
    )

    r_squared = r_value**2

    # 4. Check if this test_weight gives a better fit
    if r_squared > best_r_squared:
        best_r_squared = r_squared
        optimal_cost_conversion = test_weight

    print(
        f"  Testing Weight: {
            test_weight:.1f} minutes/RM | R-squared: {r_squared:.4f}"
    )

print(f"\n--- OPTIMIZATION RESULT ---")
print(
    f"The optimal conversion factor (1 RM = X minutes) that best fits the observed traffic is:"
)
print(
    f"Optimal Time Value of Money: 1 RM = {optimal_cost_conversion:.1f} minutes (R² = {
        best_r_squared:.4f})"
)

FINAL_TIME_VALUE_OF_MONEY = optimal_cost_conversion
print(
    f"\n--- 4. MARKOV CHAIN: Running final model with {
        FINAL_TIME_VALUE_OF_MONEY:.1f} minutes/RM ---"
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
            likelihood = attractiveness_j / 2.0
        else:
            if i in path_lengths and j in path_lengths[i]:
                cost_ij = path_lengths[i][j]
                if cost_ij <= 0:
                    cost_ij = 0.1
                likelihood = attractiveness_j / cost_ij
            else:
                likelihood = 0.0
        P.loc[i, j] = likelihood

# Normalize
P_normalized = P.div(P.sum(axis=1), axis=0)
P_normalized = P_normalized.fillna(0)

print(P_normalized)
# Solve for Steady State
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
    id_to_name = stations.set_index("Station_ID")["Station_Name"].to_dict()
    final_ranking.index = final_ranking.index.map(
        lambda x: f"{x} ({id_to_name.get(x, 'Unknown')})"
    )
except:
    pass

print("\n=== FINAL PREDICTION: Tourist Hubs (Optimized Model) ===")
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
