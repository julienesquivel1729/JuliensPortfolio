import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the cleaned dataset
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

# Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define optimized parameter grid for LightGBM
param_grid = {
    "num_leaves": [31],  # Keeps it efficient while allowing flexibility
    "learning_rate": [0.05],  # Balanced learning rate
    "n_estimators": [200],  # Prevents excessive training time
    "class_weight": ["balanced"]  # Addresses class imbalance
}

# Perform Grid Search with f1_score optimization
grid_search = GridSearchCV(
    lgb.LGBMClassifier(objective="binary", random_state=42),
    param_grid, cv=5, scoring="f1", n_jobs=-1
)

print("🔄 Running LightGBM Hyperparameter Optimization... This may take a few minutes.")
grid_search.fit(X_train, y_train)

# Get the best model
best_lgb_model = grid_search.best_estimator_

# Make predictions
y_pred_best_lgb = best_lgb_model.predict(X_test)

# Evaluate performance
lgb_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_best_lgb),
    "Precision": precision_score(y_test, y_pred_best_lgb),
    "Recall": recall_score(y_test, y_pred_best_lgb),
    "F1-Score": f1_score(y_test, y_pred_best_lgb),
    "AUC-ROC": roc_auc_score(y_test, y_pred_best_lgb),
}

print("✅ LightGBM Model Performance:")
for metric, value in lgb_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\n🔍 Best Hyperparameters Found:", grid_search.best_params_)
