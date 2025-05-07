# 📂 Scripts Folder

This folder contains the scripts used in the Loan Eligibility Prediction project.

## Files

- **preprocessing.py**:  
  Cleans, encodes, and transforms the raw loan data.  
  Saves processed features (`X`) and target variable (`y`) into CSV files inside the `processed_data/` folder.

## How to Run

1. Ensure the raw data file `loan_eligible.csv` is inside the `data/` folder.
2. Run the script from the project root:

```bash
python scripts/preprocessing.py

