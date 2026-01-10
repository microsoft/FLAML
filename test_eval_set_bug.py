"""Test script to reproduce the eval_set preprocessing bug."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- 1. Create synthetic dataset with numeric + categorical features ---
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    "num1": np.random.randn(n),
    "num2": np.random.rand(n) * 10,
    "cat1": np.random.choice(["A", "B", "C"], size=n),
    "cat2": np.random.choice(["X", "Y"], size=n),
    "target": np.random.choice([0, 1], size=n)
})

# --- 2. Split data ---
X = df.drop(columns="target")
y = df["target"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

# --- 3. Convert categorical columns to pandas 'category' dtype ---
for col in X_train.select_dtypes(include="object").columns:
    X_train[col] = X_train[col].astype("category")
    X_valid[col] = X_valid[col].astype("category")

print("=" * 60)
print("Testing with standard XGBClassifier (should work)")
print("=" * 60)

# --- 4. Test with standard XGBClassifier ---
try:
    from xgboost import XGBClassifier
    
    model = XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        eval_metric="logloss",
        use_label_encoder=False,
        early_stopping_rounds=10,
        random_state=0,
        n_estimators=10
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    print("✓ Standard XGBClassifier: SUCCESS")
except Exception as e:
    print(f"✗ Standard XGBClassifier: FAILED - {e}")

print("\n" + "=" * 60)
print("Testing with FLAML XGBClassifier (should fail with bug)")
print("=" * 60)

# --- 5. Test with FLAML XGBClassifier ---
try:
    import flaml.default as flaml_zeroshot
    
    model = flaml_zeroshot.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        eval_metric="logloss",
        use_label_encoder=False,
        early_stopping_rounds=10,
        random_state=0,
        n_estimators=10
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    print("✓ FLAML XGBClassifier: SUCCESS")
except Exception as e:
    print(f"✗ FLAML XGBClassifier: FAILED - {e}")

print("\n" + "=" * 60)
print("Testing with FLAML XGBRegressor (should also fail with bug)")
print("=" * 60)

# --- 6. Test with FLAML XGBRegressor for regression ---
try:
    import flaml.default as flaml_zeroshot
    
    # Create regression data
    y_reg = df["num1"]  # Use num1 as target for regression
    X_reg = df.drop(columns=["num1", "target"])
    
    X_train_reg, X_valid_reg, y_train_reg, y_valid_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=0
    )
    
    # Convert categorical columns
    for col in X_train_reg.select_dtypes(include="object").columns:
        X_train_reg[col] = X_train_reg[col].astype("category")
        X_valid_reg[col] = X_valid_reg[col].astype("category")
    
    model = flaml_zeroshot.XGBRegressor(
        tree_method="hist",
        enable_categorical=True,
        eval_metric="rmse",
        early_stopping_rounds=10,
        random_state=0,
        n_estimators=10
    )
    
    model.fit(
        X_train_reg, y_train_reg,
        eval_set=[(X_valid_reg, y_valid_reg)],
        verbose=False
    )
    print("✓ FLAML XGBRegressor: SUCCESS")
except Exception as e:
    print(f"✗ FLAML XGBRegressor: FAILED - {e}")
