import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import os
import json
import optuna

# Suppress Optuna's excessive logging so our terminal stays clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    if optimal_idx < len(thresholds):
        return thresholds[optimal_idx], f1_scores[optimal_idx]
    return 0.5, f1_scores[-1]

def main():
    data_path = "data/processed/train_final.csv"
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/interpretability", exist_ok=True)

    print(f"Loading engineered data from {data_path}...")
    df = pd.read_csv(data_path)

    print("Isolating features and handling data types...")
    y = df['TARGET']
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X = X.astype({col: int for col in X.select_dtypes(include='bool').columns})

    # Advanced Splitting Strategy
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp)

    scale_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    # --- PHASE 1: THE OPTUNA SEARCH ---
    print("\n🔍 Initiating Bayesian Hyperparameter Search (10 Trials)...")
    
    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "scale_pos_weight": scale_weight,
            # Optuna will dynamically test different values for these:
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": 300,
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "early_stopping_rounds": 20
        }
        
        opt_model = xgb.XGBClassifier(**params)
        opt_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_val_proba = opt_model.predict_proba(X_val)[:, 1]
        
        # We tell Optuna to maximize the Precision-Recall AUC
        return average_precision_score(y_val, y_val_proba)

    # Run the search
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10) # We use 10 for time. In FAANG, we'd use 100+.
    
    best_params = study.best_params
    print(f"✅ Search Complete! Best PR-AUC achieved: {study.best_value:.4f}")
    print(f"🏆 Optimal Architecture Found: Depth {best_params['max_depth']}, LR {best_params['learning_rate']:.4f}")

    # --- PHASE 2: PRODUCTION TRAINING & REGISTRY ---
    print("\nTraining final production model with optimal parameters...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("SentinelScore_Credit_Risk")

    with mlflow.start_run() as run:
        # Add our static params back to the best_params dictionary
        best_params["objective"] = "binary:logistic"
        best_params["eval_metric"] = "aucpr"
        best_params["scale_pos_weight"] = scale_weight
        best_params["n_estimators"] = 500
        best_params["random_state"] = 42
        best_params["early_stopping_rounds"] = 25
        
        mlflow.log_params(best_params)

        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # Evaluation & Threshold Optimization
        y_proba = final_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        opt_threshold, opt_f1 = optimize_threshold(y_test, y_proba)
        
        mlflow.log_metrics({"roc_auc": roc_auc, "pr_auc": pr_auc, "optimal_threshold": opt_threshold, "optimal_f1": opt_f1})

        print(f"✅ Final Production Performance - ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
        print(f"🎯 New Optimal Threshold: {opt_threshold:.4f}")

        # Artifact Generation
        plt.figure(figsize=(10, 8))
        xgb.plot_importance(final_model, max_num_features=20, importance_type='gain')
        plt.title("Top 20 Features by Information Gain (Optuna Tuned)")
        plt.tight_layout()
        importance_path = "reports/interpretability/feature_importance.png"
        plt.savefig(importance_path)
        plt.close()
        
        mlflow.log_artifact(importance_path)

        # MLflow & Physical Exports
        signature = mlflow.models.infer_signature(X_train, y_proba)
        mlflow.xgboost.log_model(final_model, artifact_path="xgboost-model", signature=signature, registered_model_name="SentinelScore_Prod")
        
        final_model.save_model("models/production_model.json")
        threshold_config = {"optimal_threshold": float(opt_threshold), "features": X_train.columns.tolist()}
        with open("models/threshold.json", "w") as f:
            json.dump(threshold_config, f)

if __name__ == "__main__":
    main()