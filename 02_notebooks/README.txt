Reproduction notebooks

01_data_preparation.ipynb — rebuild accounting labels and supplied-week features from the original datasets.
02_original_models.ipynb — reproduce the broader feature-significance study (including the report's entry-price result).
03_final_validation.ipynb — reproduce the report's latest matched-cohort history tests and prediction metrics.

Python 3.9–3.12: pip install -r requirements.txt
Open a notebook here, choose your installed Python 3 kernel, run the first setup cell, then Run All. All analysis code is embedded in the notebooks. The added setup cell only adapts folder paths; original analysis cells and saved UCLA outputs are preserved.

The data and results entries here are relative links into 04_results. Keep all four top-level folders together. Included validated derived tables let notebooks 02–03 run independently; notebook 01 can rebuild those tables from the original supplied records. Reruns write under 04_results/reruns and do not overwrite the preserved results.

Settlement mapping was verified for all 2,661 markets in the final validation. Older unverified-mapping flags/text are historical. Real net profit and historical result-publication times remain unverified. The latest test comparisons reuse an inspected period.
