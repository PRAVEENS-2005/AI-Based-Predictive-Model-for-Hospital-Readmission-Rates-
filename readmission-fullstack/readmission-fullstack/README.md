# Hospital Readmission XAI Fullstack

This package includes:
- `backend/` FastAPI API that trains the logistic regression model on startup
- `frontend/` React dashboard that fetches live metrics and prediction explanations from the backend

## What are the backend privileges?

This backend is a **development API**. In plain language, the permissions or "privileges" I gave it are:

- It can **load the dataset zip from the backend folder**
- It can **train the model in memory** when the server starts
- It can **serve metrics, feature importance, example patients, predictions, and top-3 explanations**
- It allows **any frontend origin** to call it because CORS is set to `allow_origins=["*"]`
- It has **no login, no authentication, and no user roles**

What it does **not** do:
- it does not write back to your dataset
- it does not modify system files
- it does not create users
- it does not have database privileges
- it does not secure patient data for production use

So the real answer is: **the frontend has permission to call the backend API, and the backend has permission to read the dataset and return model results.** That is all. No magical admin powers, despite how software errors like to sound.

## Folder layout

- `backend/app/main.py` API and model logic
- `frontend/src/App.jsx` dashboard and API integration

## Requirements

- Python 3.10+
- Node.js 18+
- Put the dataset zip file named exactly `diabetes+130-us+hospitals+for+years+1999-2008.zip` inside `backend/`

## Run the backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Check:
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

## Run the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, usually:
- `http://127.0.0.1:5173`

If the backend runs somewhere else, change the **API Base URL** field at the top of the dashboard.

## Main connected features

- Accuracy, ROC-AUC, and recall cards from backend
- global feature driver chart from backend coefficient output
- example test-patient table from backend
- live prediction form
- top 3 risk-driver explanation returned by the backend

## Important

This is suitable for a college project and local demo use.
It is **not production-safe clinical software**.
