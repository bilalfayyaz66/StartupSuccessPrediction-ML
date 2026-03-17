import pandas as pd
import joblib
from config import FINAL_MODEL_PATH, IMPUTER_PATH

model = joblib.load(FINAL_MODEL_PATH)
imputer = joblib.load(IMPUTER_PATH)

sample_data = pd.DataFrame([{
    "relationships": 15,
    "funding_rounds": 3,
    "funding_total_usd": 8000000,
    "milestones": 4,
    "is_CA": 1,
    "is_NY": 0,
    "is_MA": 0,
    "is_TX": 0,
    "is_otherstate": 0,
    "category_code": 5,
    "is_software": 1,
    "is_web": 0,
    "is_mobile": 0,
    "is_enterprise": 1,
    "is_advertising": 0,
    "is_gamesvideo": 0,
    "is_ecommerce": 0,
    "is_biotech": 0,
    "is_consulting": 0,
    "is_othercategory": 0,
    "has_VC": 1,
    "has_angel": 1,
    "has_roundA": 1,
    "has_roundB": 1,
    "has_roundC": 0,
    "has_roundD": 0,
    "avg_participants": 3.5,
    "is_top500": 1
}])

sample_data["funding_per_round"] = sample_data["funding_total_usd"] / (sample_data["funding_rounds"] + 1)
sample_data["participants_per_round"] = sample_data["avg_participants"] / (sample_data["funding_rounds"] + 1)
sample_data["milestones_per_round"] = sample_data["milestones"] / (sample_data["funding_rounds"] + 1)
sample_data["network_funding_strength"] = sample_data["relationships"] * sample_data["avg_participants"]
sample_data["rounds_x_top500"] = sample_data["funding_rounds"] * sample_data["is_top500"]

sample_data = pd.DataFrame(imputer.transform(sample_data), columns=sample_data.columns)

prediction = model.predict(sample_data)[0]
probability = model.predict_proba(sample_data)[0][1]

print("Predicted class:", prediction)
print("Predicted success probability:", round(probability, 4))