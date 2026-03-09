# Hospital Readmission XAI Backend

A FastAPI backend for your diabetic patient 30-day readmission dashboard.
It trains a leakage-free logistic regression model from the UCI diabetes readmission dataset at startup and exposes prediction, metrics, examples, and explanation endpoints.

## Before you run
Put `diabetes+130-us+hospitals+for+years+1999-2008.zip` in this backend root folder,
or set the `DATASET_ZIP` environment variable.

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
uvicorn app.main:app --reload --port 8000
```

Open:
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/docs

## Endpoints
- `GET /health`
- `GET /metrics`
- `GET /features/top?limit=10`
- `GET /patients/examples?limit=10`
- `GET /explain/test/{sample_index}`
- `POST /predict`
- `GET /dashboard/summary`

## Example POST /predict body
```json
{
  "race": "Caucasian",
  "gender": "Female",
  "age": "[70-80)",
  "admission_type_id": 1,
  "discharge_disposition_id": 1,
  "admission_source_id": 7,
  "time_in_hospital": 6,
  "num_lab_procedures": 41,
  "num_procedures": 0,
  "num_medications": 18,
  "number_outpatient": 0,
  "number_emergency": 0,
  "number_inpatient": 1,
  "diag_1": "428",
  "diag_2": "250.83",
  "diag_3": "401",
  "number_diagnoses": 9,
  "max_glu_serum": "None",
  "A1Cresult": "None",
  "metformin": "No",
  "repaglinide": "No",
  "nateglinide": "No",
  "chlorpropamide": "No",
  "glimepiride": "No",
  "acetohexamide": "No",
  "glipizide": "No",
  "glyburide": "No",
  "tolbutamide": "No",
  "pioglitazone": "No",
  "rosiglitazone": "No",
  "acarbose": "No",
  "miglitol": "No",
  "troglitazone": "No",
  "tolazamide": "No",
  "examide": "No",
  "citoglipton": "No",
  "insulin": "Steady",
  "glyburide-metformin": "No",
  "glipizide-metformin": "No",
  "glimepiride-pioglitazone": "No",
  "metformin-rosiglitazone": "No",
  "metformin-pioglitazone": "No",
  "change": "Ch",
  "diabetesMed": "Yes"
}
```
