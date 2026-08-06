# Diabetes Predictor

A small, modular ML project: train several regression models, keep the
best one, and serve it through a FastAPI endpoint.

## Structure

```
diabetes-predictor/
├── src/
│   ├── config.py         # all hyperparameters + paths
│   ├── data_loader.py    # loads the dataset
│   ├── preprocessing.py  # train/test split
│   ├── models.py         # registry of candidate models
│   ├── evaluator.py      # fit + score one model
│   ├── comparison.py     # run every model, rank by R2
│   ├── train.py          # entry point: trains + saves the best model
│   └── predict.py        # loads saved model, used by the API
├── api/
│   └── main.py           # FastAPI app (imports only predict.py)
├── models/
│   └── diabetes_model.pkl (created after training)
├── requirements.txt
└── README.md
```

Each file has exactly one job. The API never imports `train.py` or
`comparison.py` — only `predict.py`. That means retraining your models
never risks breaking the API, and vice versa.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Train

From the project root:

```bash
python -m src.train
```

This loads the data, trains + compares 8 regression models, prints a
ranked table, and saves the best-performing model to
`models/diabetes_model.pkl`.

## Run the API

```bash
uvicorn api.main:app --reload
```

Then visit `http://127.0.0.1:8000/docs` for interactive Swagger docs,
or test directly:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":0.03,"sex":0.05,"bmi":0.06,"bp":0.02,"s1":-0.04,"s2":-0.03,"s3":-0.02,"s4":0.01,"s5":0.02,"s6":0.01}'
```

## Run the tests

```bash
pytest tests/test_api.py -v
```

This runs the API in-process (no server needed) against several example
patients — a baseline case, higher-risk values, lower-than-average
values, and extreme/boundary values — plus checks that bad input (a
missing field, wrong type) correctly returns a 422 error. Requires a
trained model at `models/diabetes_model.pkl` (run `python -m src.train`
first if you haven't).

## Using your own data

Right now `data_loader.py` loads scikit-learn's built-in diabetes toy
dataset. To use your own CSV, edit `train.py` to call
`load_dataset_from_csv(path, target_column)` instead of `load_dataset()`
(the function already exists in `data_loader.py`) — nothing else needs
to change.

## Next steps (Docker + AWS)

Once you're happy with the model, this project is ready for:
1. A `Dockerfile` that installs `requirements.txt` and runs
   `uvicorn api.main:app --host 0.0.0.0 --port 8000`
2. Pushing the image to ECR and deploying on ECS/Fargate or EC2
3. A simple web UI (React/HTML) that POSTs to `/predict`

Ask for help with any of these when you're ready.
