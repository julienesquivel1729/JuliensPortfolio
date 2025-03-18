import pandas as pd
import lightgbm as lgb
import pickle

# Load the cleaned dataset
cleaned_data_path = "cleaned_economic_data.csv"
merged_data = pd.read_csv(cleaned_data_path)

# Convert 'Date' column to datetime
merged_data["Date"] = pd.to_datetime(merged_data["Date"], errors="coerce")

# Select features and target
features = ["UNRATE", "CORESTICKM159SFRBATL", "GDP_Growth", "Yield_Curve_Spread"]
target = "USRECD"

# Drop missing values
merged_data = merged_data.dropna(subset=features + [target])

# Define X (features) and y (target)
X = merged_data[features]
y = merged_data[target]

# Train LightGBM model
model = lgb.LGBMClassifier(
    num_leaves=40,
    learning_rate=0.03,
    n_estimators=300,
    class_weight="balanced"
)
model.fit(X, y)

# Save the trained model
with open("recession_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model saved as 'recession_model.pkl'")
