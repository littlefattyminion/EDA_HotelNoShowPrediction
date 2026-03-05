
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def make_model(
    name: str,
    *,
    random_state: int,
    n_estimators: int = 400,
    max_depth=None,
    # HGB params (safe defaults)
    hgb_learning_rate: float = 0.03,
    hgb_max_iter: int = 800,
    hgb_max_depth: int = 8,
):
    name = name.lower().strip()

    if name == "logistic":
        return LogisticRegression(max_iter=2000, class_weight="balanced")

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
            max_depth=max_depth,
        )

    if name in ("hist_gradient_boost", "hist_gb", "hgb"):
        # Note: HistGradientBoostingClassifier does not support class_weight directly.
        # We'll rely on better features / tuning, and (optionally) sample_weight later.
        return HistGradientBoostingClassifier(
            learning_rate=hgb_learning_rate,
            max_iter=hgb_max_iter,
            max_depth=hgb_max_depth,
            l2_regularization=1.0,
            random_state=random_state,
        )

    raise ValueError("Unknown model. Use: logistic | random_forest | hist_gradient_boost")