from pathlib import Path
import shutil
root = Path(__file__).resolve().parent
nb = root / "02_notebooks"
for name, source in [("data", root / "04_results/reproduction_inputs")]:
    shutil.copytree(source, nb / name, dirs_exist_ok=True)
results = nb / "results"
results.mkdir(exist_ok=True)
for name, source in [("full_week", root / "04_results/01_data_preparation"), ("modeling/production", root / "04_results/02_original_models"), ("validation/production", root / "04_results/03_history_validation")]:
    shutil.copytree(source, results / name, dirs_exist_ok=True)
print("Portable notebook input folders prepared. Model reruns have not been executed.")
