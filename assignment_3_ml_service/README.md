# Assignment 3 — Production-Style ML Service

I took the stroke-prediction model from Assignment 1 and turned it into an actual local service:
I trained it with proper hyperparameter tuning, saved the whole preprocessing+model pipeline to
disk, wrapped it in a FastAPI app with a health check and a monitoring endpoint, built a small web
page so it's usable without touching a terminal, containerized it with Docker, and finished by
deliberately feeding it shifted data to see how it reacts. Everything below is what I actually did,
in the order I did it, plus what I found along the way.

---

## 1. What's in this folder

```
assignment_3_ml_service/
├── data/
│   ├── stroke-data.csv        # the same data from Assignment 1
│   ├── test_set.csv           # the held-out test split (created by train.py)
│   └── shifted_data.csv       # the drift-simulation dataset (created by drift_check.py)
├── src/
│   ├── train.py                # trains + tunes + persists the model
│   ├── evaluate.py             # re-scores the persisted model on the test set
│   └── drift_check.py          # Part G: drift simulation
├── model/
│   ├── model.joblib            # the persisted preprocessing+model pipeline
│   ├── metadata.json           # everything about the model, in plain text
│   └── evaluation_report.json  # written by evaluate.py
├── api/
│   ├── main.py                 # the FastAPI app
│   └── schemas.py              # Pydantic request/response models
├── monitoring/
│   ├── monitoring.py           # in-memory request/latency/class-distribution tracking
│   └── drift_report.json       # written by drift_check.py
├── static/
│   └── index.html              # the browser-friendly front end (served at /ui)
├── tests/
│   └── test_api.py             # pytest tests for the API
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── concepts.md                 # my own notes on how/why everything works (not for submission)
├── observations.md
└── README.md                   # this file
```

---

## 2. What I did, step by step

### Step 1 — Trained and tuned the model (`src/train.py`)

I reused the stroke dataset from Assignment 1 and the same cleaning I did there (`bmi` converted
from text to numeric, `id` dropped). I split it 60/20/20 into train/validation/test, same as
before, with the same random seed so the split is reproducible.

This time, instead of comparing seven models like Assignment 1, I picked **one** — Random Forest —
and actually tuned it properly with `RandomizedSearchCV` (25 candidate combinations, 5-fold
cross-validation, scored on ROC-AUC because the ~5% stroke rate makes plain accuracy misleading). I
tuned five hyperparameters: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, and
`class_weight`.

**What I observed:** the search's best cross-validation ROC-AUC was **0.8376**. When I checked that
same tuned model against the validation set it had never been scored on during the search, I got
**0.8449** ROC-AUC — close to the CV estimate, which told me the tuning wasn't overfitting to the
CV folds. On the untouched test set, final ROC-AUC came out to **0.8388**. The winning
hyperparameters were `n_estimators=430`, `max_depth=5`, `min_samples_leaf=7`, `max_features=None`,
`class_weight=None`.

One thing I noticed and want to be upfront about: **precision, recall, and F1 all came out to
0.0000 on both validation and test.** At the default 0.5 probability cutoff, the tuned model still
predicts "no stroke" for essentially everyone, exactly like every model in Assignment 1's benchmark
did once the imbalance kicked in. ROC-AUC improved somewhat versus Assignment 1's Random Forest
(0.795 test ROC-AUC there vs 0.839 here), meaning the tuned model genuinely got *better at ranking*
who's at higher risk — but that improvement never crosses the 0.5 threshold needed to flip any
individual prediction to "stroke." I talk through why, and what I'd do about it, in
`observations.md`.

### Step 2 — Confirmed no leakage between train/val/test

The brief specifically asks me to confirm validation/test data only ever gets *transformed*
through the already-fitted pipeline, never *fit* on. I made sure of this by construction: the
`ColumnTransformer` only ever gets fit once, inside `pipeline.fit(X_train)` (which happens inside
`RandomizedSearchCV`). Every later call — scoring on validation, scoring on test — uses
`.predict()` / `.predict_proba()`, which only calls `.transform()` internally on the already-fitted
preprocessor. I never call `fit_transform` on `X_val` or `X_test` anywhere in the codebase. Full
reasoning is in `concepts.md` if I want to remind myself later.

### Step 3 — Persisted the pipeline, not just the classifier

`model/model.joblib` holds the **entire fitted pipeline** — preprocessing steps and classifier
together — so the API never has to reimplement imputing/scaling/encoding by hand. `model/metadata.json`
sits next to it with the model version, algorithm, training date, the exact raw feature names and
order the pipeline expects, the tuning results, validation/test metrics, and the exact package
versions (`scikit-learn`, `pandas`, `numpy`, `joblib`) used to train it — I pinned
`requirements.txt` to match those exactly, since a pickled scikit-learn pipeline isn't always safe
to load with a different scikit-learn version than it was saved with.

### Step 4 — Wrote a standalone evaluation script

`src/evaluate.py` loads `model.joblib` and `data/test_set.csv` independently of `train.py`, re-runs
the same metrics, and cross-checks them against what `metadata.json` recorded at training time.
When I ran it, every metric matched exactly (accuracy 0.9511, ROC-AUC 0.8388, etc.) — confirming
the persisted pipeline behaves identically to how it did right after training, not just in theory.

### Step 5 — Built the FastAPI service

Three real API endpoints, plus a fourth for monitoring:

- `GET /` — a home endpoint that returns a welcome message, the model version, and links to the
  docs and the web UI (so hitting the API root gives you something useful instead of a 404).
- `GET /health` — reports whether the service and the model are up, plus uptime.
- `POST /predict` — takes a patient's raw feature values (validated by a Pydantic model in
  `api/schemas.py`), runs them through the persisted pipeline, and returns the prediction,
  probability, model version, and how long the prediction took.
- `GET /monitoring` — the running counters described below.

The model loads exactly once, at startup (via FastAPI's `lifespan`), not on every request.

**Example response**, matching the shape from the brief:
```json
{ "prediction": 0, "probability": 0.2176, "model_version": "1.0", "latency_ms": 73.746 }
```
(That's a real response I got testing locally with a 67-year-old patient profile — the higher
latency on the very first call includes some one-time warm-up; later calls are consistently faster.)

**Malformed input** gets a clear, structured error instead of a cryptic stack trace. I tested this
by sending a request missing the `age` field and by sending an invalid `gender` value — both came
back as HTTP 422 with a JSON body like:
```json
{
  "error": "validation_error",
  "message": "One or more fields in your request were missing or invalid.",
  "details": [{"type": "missing", "loc": ["body", "age"], "msg": "Field required", ...}]
}
```

### Step 6 — Built a browser page so no one needs curl or Postman

`static/index.html`, served at `/ui`, is a single self-contained page: a home screen with two big
buttons ("Make a Prediction" and "Check Service Health"), a form with plain-English labels for
every patient field, and a results view with a "Back" button. It calls `/health` and `/predict`
with plain JavaScript `fetch()` and renders whatever comes back — no build step, no framework, so
anyone can open it and use the model without knowing what an API is.

### Step 7 — Wrote tests

`tests/test_api.py` uses FastAPI's `TestClient` to check: the home and health endpoints respond
correctly, a valid prediction request succeeds and returns a sane probability, a missing field is
rejected with a clear 422, an invalid category value is rejected, an out-of-range age is rejected,
a missing (optional) `bmi` is *accepted* (the pipeline imputes it, same as at training time), and
the monitoring endpoint reflects the requests that were just made. All 8 tests pass.

### Step 8 — Containerized it

The Dockerfile only copies what the running service actually needs — `api/`, `monitoring/`,
`model/`, `static/`, and `requirements.txt` — not the training scripts or the raw dataset, since
training happens on my machine beforehand, and the container's only job is to serve the already-
trained pipeline. See the Docker section below for the exact commands I used.

### Step 9 — Simulated drift

`src/drift_check.py` takes the real test set and deliberately shifts it — older patients (age
+15 years), higher average glucose (×1.3), higher BMI (+5), more diagnosed hypertension, more
current smokers — then compares feature statistics, category distributions, and the model's
prediction distribution between the original and shifted data, and (since I kept the original
labels attached) checks whether performance actually changed too. Full numbers and reasoning are
in `observations.md`, but the short version: the input distribution shifted clearly (real data
drift), the model's predicted probabilities rose in response, but its 0.5-threshold classification
outputs barely moved because precision/recall were already at zero before the shift — there wasn't
much classification performance left to lose.

---

## 3. Running it locally (without Docker)

### Setup
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Train the model
```bash
python src/train.py
```
This creates `model/model.joblib`, `model/metadata.json`, and `data/test_set.csv`. Takes about
1–2 minutes (most of it is the hyperparameter search).

### Evaluate it independently
```bash
python src/evaluate.py
```

### Run the drift simulation
```bash
python src/drift_check.py
```
This creates `data/shifted_data.csv` and `monitoring/drift_report.json`.

### Start the API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Try it
- Open **http://localhost:8000/ui** in a browser for the point-and-click interface.
- Open **http://localhost:8000/docs** for FastAPI's interactive auto-generated API docs.
- Or from the command line:
```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "gender": "Female", "age": 67, "hypertension": 0, "heart_disease": 1,
        "ever_married": "Yes", "work_type": "Private", "Residence_type": "Urban",
        "avg_glucose_level": 228.69, "bmi": 36.6, "smoking_status": "formerly smoked"
      }'

curl http://localhost:8000/monitoring
```

### Run the tests
```bash
pytest tests/ -v
```

---

## 4. Running it with Docker

I already have Docker installed and running locally, so this is exactly how I built and ran it.

### Build the image
```bash
docker build -t stroke-risk-api:1.0 .
```

### Run the container
```bash
docker run -d --name stroke-risk-api -p 8000:8000 stroke-risk-api:1.0
```
`-p 8000:8000` maps the container's port 8000 to my machine's port 8000, so the API is reachable at
the same `http://localhost:8000` as the local (non-Docker) run above.

### Test it from outside the container
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{...}'
```
Same requests, same responses — the container is running the exact same `model.joblib` that
`train.py` produced locally, since it gets copied into the image at build time.

### View logs
```bash
docker logs -f stroke-risk-api
```

### Stop and remove the container
```bash
docker stop stroke-risk-api
docker rm stroke-risk-api
```

### Using Docker Compose instead (one command for build + run)
```bash
docker compose up --build          # build the image and start the container
docker compose up -d --build       # same, but detached (runs in the background)
docker compose down                # stop and remove the container
docker compose logs -f             # follow logs
```
`docker-compose.yml` already maps port 8000 the same way and sets a restart policy
(`unless-stopped`) plus a basic healthcheck that pings `/health` every 30 seconds.

### A note on rebuilding after retraining
If I re-run `python src/train.py` locally and get a new `model.joblib`, the Docker image needs to
be rebuilt to pick it up — `docker build` / `docker compose up --build` copies whatever is
currently in `model/` at build time; it doesn't watch the folder for changes.

---

## 5. Notes / gotchas

- `requirements.txt` is pinned to the **exact** package versions used to train the model
  (see `model/metadata.json` → `package_versions`). This matters more than it might seem — loading
  a pickled scikit-learn pipeline with a different scikit-learn version than it was saved with can
  fail or silently behave differently.
- `bmi` is the one optional field in `/predict` — leave it out and the pipeline median-imputes it,
  exactly like it did for the ~200 rows with missing `bmi` at training time.
- The monitoring counters in `/monitoring` are in-memory only — they reset if the API process (or
  container) restarts. That's a known, intentional limitation for a project at this scope; see
  `concepts.md` for the reasoning.
- No credentials or secrets are stored in the image or the repo — this model doesn't need any to
  run.

---

## 6. Next step

`observations.md` has my full write-up of the tuning results, what deploying and testing this
actually looked like, what the monitoring counters showed after exercising the API, and the full
drift-simulation findings — all grounded in the real numbers from running this on my machine.
