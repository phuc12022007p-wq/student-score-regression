from pathlib import Path
import joblib
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "student_scores.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "linear_regression.joblib"

FEATURES = ["hours_studied", "attendance", "assignments_completed", "previous_score"]
TARGET = "exam_score"

def load_and_validate() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df[required].isna().any().any():
        raise ValueError("Missing values were found in required columns")
    return df

def main() -> None:
    df = load_and_validate()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42
    )
    
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ])
    
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    
    metrics = {
        "mae": float(mean_absolute_error(y_test, prediction)),
        "rmse": float(mean_squared_error(y_test, prediction) ** 0.5),
        "r2": float(r2_score(y_test, prediction)),
    }
    
    artifact = {
        "model": model,
        "feature_names": FEATURES,
        "target_name": TARGET,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
    }
    
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH, compress=3)
    print(f"Saved model to {MODEL_PATH}")
    for name, value in metrics.items():
        print(f"{name.upper()}: {value:.4f}")

if __name__ == "__main__":
    main()