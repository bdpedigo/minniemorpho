from pathlib import Path

from skops.io import load

current_dir = Path(__file__).resolve().parent
model_dir = current_dir.parent.parent.parent / "models"


def load_model(model_name):
    model_path = model_dir / f"{model_name}.skops"
    return load(model_path)
