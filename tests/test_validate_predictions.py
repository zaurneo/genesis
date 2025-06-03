import json
import tempfile
from tools import validate_predictions

def test_validate_predictions_zero_train_mse(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.chdir(tmpdir)
        predictions = {
            "train_predictions": [1.0, 2.0, 3.0],
            "test_predictions": [1.0, 2.0, 3.0],
            "train_actual": [1.0, 2.0, 3.0],
            "test_actual": [1.0, 2.0, 3.0],
        }
        with open("predictions.json", "w") as f:
            json.dump(predictions, f)
        feature_info = {"features": [], "target": "value"}
        with open("feature_info.json", "w") as f:
            json.dump(feature_info, f)
        result = validate_predictions()
        assert result.get("success")
        ratio = result["validation"]["overfitting_check"]["overfitting_ratio"]
        assert ratio == ratio and ratio not in (float("inf"), float("-inf"))

