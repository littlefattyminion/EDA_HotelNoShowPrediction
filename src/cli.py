import argparse
from pathlib import Path

from .config import Config
from .data_loader import load_data
from .features import clean_and_engineer
from .train import train_eval_with_cv_and_tuning, save_model
from .utils import ensure_dir, save_json, now_iso

def parse_args():
    p = argparse.ArgumentParser(description="Hotel No-Show pipeline (SQLite → ML)")
    p.add_argument("--db-path", default=Config.db_path)
    p.add_argument("--test-size", type=float, default=Config.test_size)
    p.add_argument("--random-state", type=int, default=Config.random_state)
    p.add_argument("--fx-usd-to-sgd", type=float, default=Config.fx_usd_to_sgd)

    p.add_argument("--model", default="all", help="all | logistic | random_forest | hist_gradient_boost")
    p.add_argument("--n-estimators", type=int, default=Config.n_estimators)
    p.add_argument("--max-depth", type=int, default=-1)

    p.add_argument("--outputs-dir", default="outputs")
    
    # CV and tuning options
    p.add_argument("--cv", action="store_true", help="Enable StratifiedKFold CV AUC reporting")
    p.add_argument("--cv-folds", type=int, default=5)

    p.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning (HGB only)")
    p.add_argument("--tune-iter", type=int, default=25)
    p.add_argument("--tune-cv-folds", type=int, default=3)
    return p.parse_args()

def main():
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    ensure_dir(str(outputs_dir))

    df_raw = load_data(args.db_path)
    df = clean_and_engineer(df_raw, fx_usd_to_sgd=args.fx_usd_to_sgd)

    models = ["logistic", "random_forest", "hist_gradient_boost"] if args.model == "all" else [args.model]
    max_depth = None if args.max_depth == -1 else args.max_depth

    all_metrics = {
        "run_timestamp": now_iso(),
        "db_path": args.db_path,
        "rows_after_cleaning": int(df.shape[0]),
        "models": {},
    }

    best_name, best_auc, best_pipe = None, -1.0, None

    for m in models:
        pipe, metrics = train_eval_with_cv_and_tuning(
        df,model_name=m,
        test_size=args.test_size,
        random_state=args.random_state,
        do_cv=args.cv,
        cv_folds=args.cv_folds,
        do_tuning=args.tune,
        tuning_iter=args.tune_iter,
        tuning_cv_folds=args.tune_cv_folds,
        n_estimators=args.n_estimators,
        max_depth=max_depth,
    )
        all_metrics["models"][m] = metrics

        auc = metrics["test"].get("roc_auc")
        if auc is not None and auc > best_auc:
            best_auc, best_name, best_pipe = auc, m, pipe

        print("\n==============================")
        print(f"Model: {m}")
        print("ROC-AUC:", auc)
        print("Confusion matrix:", metrics["test"]["confusion_matrix"])

    # Save metrics
    save_json(str(outputs_dir / "metrics.json"), all_metrics)

    # Save best model (if any has AUC)
    if best_pipe is not None:
        save_model(best_pipe, str(outputs_dir / "best_model.joblib"))
        print(f"\nBest model: {best_name} (ROC-AUC={best_auc:.4f})")
        print(f"Saved: {outputs_dir / 'best_model.joblib'}")
    else:
        # still save the last pipe if for some reason no proba available
        # (unlikely here, all models have predict_proba)
        print("\nNo best model selected (ROC-AUC missing).")

if __name__ == "__main__":
    main()
