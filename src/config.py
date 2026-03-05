from dataclasses import dataclass

@dataclass
class Config:
    db_path: str = "/Users/smapgal/Documents/EDA_Assessment/Assessment_Hotel_No_Show_Prediction/EDA_HotelNoShowPrediction/data/noshow.db"
    test_size: float = 0.2
    random_state: int = 42

    # price conversion
    fx_usd_to_sgd: float = 1.35

    # model selection: logistic | random_forest | gradient_boost
    model: str = "random_forest"

    # RF params
    n_estimators: int = 400
    max_depth: int | None = None
