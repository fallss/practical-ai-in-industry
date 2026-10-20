import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from src.config import INTERACTIONS_DATA_PATH, RANKING_MODEL_PATH

FEATURE_COLUMNS = [
    "price_ratio",
    "price_diff_norm",
    "price_gaussian",
    "star_diff",
    "star_exact_match",
    "star_rating",
    "review_score",
    "rating_scaled",
    "log_review_count",
    "distance_to_center_km",
    "distance_decay",
    "discount_percent",
    "discount_factor",
    "value_score"
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer pairwise and pointwise query-item ranking features."""
    feat = pd.DataFrame(index=df.index)
    feat["price_ratio"] = df["price_per_night"] / (df["user_budget"] + 1e-5)
    feat["price_diff_norm"] = np.abs(df["price_per_night"] - df["user_budget"]) / (df["user_budget"] + 1e-5)
    feat["price_gaussian"] = np.exp(-0.5 * ((feat["price_ratio"] - 1.0) ** 2))
    
    feat["star_diff"] = np.abs(df["star_rating"] - df["pref_stars"])
    feat["star_exact_match"] = (df["star_rating"] == df["pref_stars"]).astype(float)
    feat["star_rating"] = df["star_rating"]
    
    feat["review_score"] = df["review_score"]
    feat["rating_scaled"] = (df["review_score"] - 3.5) / 1.5
    feat["log_review_count"] = np.log1p(df["review_count"])
    
    feat["distance_to_center_km"] = df["distance_to_center_km"]
    feat["distance_decay"] = np.exp(-df["distance_to_center_km"] / 10.0)
    
    feat["discount_percent"] = df["discount_percent"]
    feat["discount_factor"] = df["discount_percent"] / 100.0
    feat["value_score"] = (df["review_score"] * 1000000.0) / (df["price_per_night"] + 1.0)
    
    return feat[FEATURE_COLUMNS]

def train_ranking_model():
    """Train Gradient Boosted Search Ranking model and save artifacts."""
    print(f"Loading interaction data from {INTERACTIONS_DATA_PATH}...")
    df = pd.read_csv(INTERACTIONS_DATA_PATH)
    
    print("Engineering features...")
    X = engineer_features(df)
    y = df["is_booked"].astype(int)
    
    print(f"Dataset shape: {X.shape}, Positive conversions: {y.sum()} ({y.mean():.2%})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training HistGradientBoostingClassifier (production ranking model)...")
    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.06,
        max_depth=6,
        min_samples_leaf=25,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"\n==========================================")
    print(f"Model Evaluation Results:")
    print(f"- ROC-AUC Score: {auc:.4f}")
    print(f"- PR-AUC (Average Precision): {pr_auc:.4f}")
    print(f"==========================================\n")
    
    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": {
            "roc_auc": round(float(auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "test_samples": len(y_test)
        },
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_type": "HistGradientBoostingClassifier (Pointwise Conversion Ranker)"
    }
    
    joblib.dump(artifact, RANKING_MODEL_PATH)
    print(f"Saved ranking model artifact to {RANKING_MODEL_PATH}")

if __name__ == "__main__":
    train_ranking_model()
