import json
import tempfile
import os
import config
from utils.tools import validate_predictions

def test_validate_predictions_zero_train_mse(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        generated_dir = os.path.join(tmpdir, "Generated_Files")
        os.makedirs(generated_dir, exist_ok=True)
        monkeypatch.setattr(config, "GENERATED_FILES_DIR", generated_dir, raising=False)
        monkeypatch.chdir(tmpdir)
        predictions = {
            "train_predictions": [1.1, 2.1, 3.1],
            "test_predictions": [1.0, 2.0, 3.0],
            "train_actual": [1.0, 2.0, 3.0],
            "test_actual": [0.9, 1.9, 2.9],
        }
        with open(os.path.join(generated_dir, "predictions.json"), "w") as f:
            json.dump(predictions, f)
        feature_info = {"features": [], "target": "value"}
        with open(os.path.join(generated_dir, "feature_info.json"), "w") as f:
            json.dump(feature_info, f)
        result = validate_predictions()
        assert result.get("success")
        ratio = result["validation"]["overfitting_check"]["overfitting_ratio"]
        assert ratio == ratio and ratio not in (float("inf"), float("-inf"))

