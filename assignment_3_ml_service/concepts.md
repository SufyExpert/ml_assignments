# concepts.md — What I Built and Why (Assignment 3)

> This file is for me, not for submission. It's the "explain it back to myself"
> document — every concept, every file, and why it exists, so I actually
> understand the flow instead of just having working code. Delete this
> before submitting if it's not wanted in the final package.

---

## 1. The big picture — what does "production-style ML service" actually mean?

Assignments 1 and 2 ended at a notebook. This one asks a different question:
**if someone else needed to use this model right now, without me around,
how would they?** That's the whole point of Assignment 3. The answer has
several layers, and each one maps to a part of the brief:

```
   train the model                A: pipeline, B: tuning
         |
         v
  save it to disk                 C: persistence (model.joblib + metadata.json)
         |
         v
  wrap it in an API                D: FastAPI (/health, /predict, /)
         |
         v
  package it in a container        E: Docker
         |
         v
  watch it while it runs           F: monitoring
         |
         v
  check it's still trustworthy     G: drift simulation
```

Each layer only works because the one below it is solid. If Part A (the
pipeline) doesn't handle preprocessing consistently, nothing downstream can
be trusted, no matter how nice the API or Docker setup looks.

---

## 2. Part A — Why everything lives inside ONE Pipeline object

In Assignment 1, I built a `ColumnTransformer` (numeric imputer+scaler,
categorical imputer+encoder) and put it in a `Pipeline` together with the
classifier. Assignment 3 does the exact same thing, for a very deliberate
reason: **a `Pipeline` object is a single thing that remembers how to
preprocess data, not just how to predict on already-preprocessed data.**

If I had instead saved *only* the trained `RandomForestClassifier`, the API
would need to reimplement the imputing/scaling/encoding logic by hand,
in a completely different codebase (`api/main.py`), written at a different
time, by someone (me) who might make a slightly different choice the second
time around. That's exactly how training/serving skew happens in real
systems — the preprocessing "drifts apart" between training code and
serving code without anyone noticing until predictions come out wrong.

By saving the whole `Pipeline`, `pipeline.predict(raw_dataframe)` in
`api/main.py` does *exactly* what `pipeline.predict(X_test)` did in
`train.py` — same imputer medians, same one-hot categories, same scaler
mean/std, because it's literally the same fitted objects, unpickled from
the same file.

**"Confirm that validation/test data is transformed only through the
fitted pipeline"** — this is about leakage. If I had called
`preprocessor.fit_transform(X_val)` anywhere, the scaler's mean/std and the
imputer's median would have been recalculated using validation data,
meaning the model's preprocessing would have "seen" statistics from data it
's not supposed to know about yet. In `train.py`, `preprocessor` is only
ever fit once, inside `pipeline.fit(X_train)`. Every later call —
`best_pipeline.predict(X_val)`, `best_pipeline.predict(X_test)` — only
*transforms* through the already-fitted preprocessor. I never call
`fit_transform` on val/test data anywhere in the codebase.

---

## 3. Part B — What GridSearchCV / RandomizedSearchCV actually do

A model like `RandomForestClassifier` has knobs (`n_estimators`,
`max_depth`, `min_samples_leaf`, ...). Different combinations of knob
settings produce different models. "Hyperparameter tuning" just means:
try a bunch of combinations, and keep whichever one scores best.

**Why cross-validation, not a single train/val split?** If I tuned by
training once and checking once against a single validation set, I could
get lucky (or unlucky) — a particular random validation split might happen
to favor one set of hyperparameters just by chance. Cross-validation
(`cv=5` here) splits the **training set** into 5 folds, trains on 4 and
validates on the 5th, five times over (each fold gets a turn as the
validation piece), and averages the score. That average is a much more
stable estimate of "how good is this hyperparameter combination, really"
than any single split could give.

**GridSearchCV vs RandomizedSearchCV:** `GridSearchCV` tries *every single
combination* of the hyperparameter grid you give it — thorough, but the
number of combinations multiplies fast (5 values × 5 values × 5 values =
125 combinations × 5 CV folds = 625 model fits). `RandomizedSearchCV`
instead randomly samples a fixed number of combinations (`n_iter=25` here)
from the same space. It won't guarantee finding the single best
combination, but it finds a *very good* one in a fraction of the time —
worth it here because I'm tuning 5 hyperparameters at once (some of them
with wide ranges), and I'd rather get a strong model in 100 seconds than
the theoretically-perfect one in an hour.

**Why I scored on `roc_auc`, not `accuracy`:** stroke cases are only ~5% of
the data. A model that always predicts "no stroke" scores ~95% accuracy
while being completely useless — if I'd told the search to optimize
accuracy, it might have handed me exactly that kind of degenerate model.
ROC-AUC checks how well the model *ranks* stroke cases above non-stroke
cases across every possible threshold, which stays meaningful even when
one class is rare.

**Why I still check validation separately, after the search:** the
`best_score_` the search reports is the *average cross-validation score
across the training folds*. It's a good number, but it's still a number
computed entirely from data the search saw (just split up internally). So
`train.py` takes the winning pipeline and scores it *again*, on the
validation set the search never touched at all, as an independent sanity
check before finally touching the test set.

---

## 4. Part C — model.joblib vs metadata.json: two different jobs

- **`model.joblib`** — the actual fitted Python objects (preprocessor +
  classifier), serialized with `joblib.dump()`. This is what actually
  produces predictions. It's a binary blob; you can't read it as text.
- **`metadata.json`** — a plain-text, human-and-machine-readable summary
  *about* that blob: what algorithm it is, when it was trained, what raw
  feature names it expects, what its tuned hyperparameters were, and what
  its validation/test scores were.

Why not just rely on `model.joblib` alone? Because a `.joblib` file
doesn't tell you *anything* about itself until you load it into the exact
same class definitions and libraries that created it. If someone (or
future-me) needs to know "what version is this, what accuracy did it get,
when was it trained, what scikit-learn version made this" without loading
the model into Python, `metadata.json` answers that instantly. The API
also reads `metadata.json` at startup to know the model version to report
back in every `/predict` response, and to know the exact raw column order
the pipeline expects.

---

## 5. Part D — how the FastAPI app actually works

**Loading the model once, at startup, not per-request.** `api/main.py`
uses FastAPI's `lifespan` context manager. Code before the `yield` runs
once, when the server process starts; code after `yield` runs once, when
it shuts down. `joblib.load(MODEL_PATH)` happens in that startup block and
the result is stashed in `app_state["pipeline"]`. Every `/predict` request
afterwards reuses that already-loaded object — loading a serialized model
from disk takes real time (I/O + deserialization), so doing it once at
startup instead of on every request is the difference between predictions
taking milliseconds vs. potentially much longer under load.

**Pydantic validation.** `api/schemas.py` defines `PatientFeatures` as a
Pydantic model with typed fields (`Literal["Male","Female","Other"]` for
gender, `float` with `ge=0, le=120` for age, and so on). FastAPI uses this
to validate every incoming request body *before* my `/predict` function
even runs. If a field is missing, has the wrong type, or falls outside an
allowed range/category, FastAPI raises a `RequestValidationError`
automatically — my code doesn't need a single `if` statement to check for
that.

**Why there's a custom exception handler for validation errors.** By
default, FastAPI's validation error response is technically correct but
not very readable, and it doesn't tell my monitoring layer that an invalid
request happened. `validation_exception_handler` in `api/main.py`
intercepts that error, returns a clearer JSON shape (`error`, `message`,
`details`), and calls `stats.record_invalid_input()` so it shows up in
`/monitoring` as an invalid-input request — not silently ignored, not
counted as a normal failure either.

**Where latency is measured.** Inside `/predict`, I record
`time.perf_counter()` right before building the input DataFrame and right
after getting the prediction back — that window is "how long did this
specific prediction actually take", not including FastAPI's own request
routing overhead. That's what ends up in the `latency_ms` field of the
response, matching the example in the brief.

**The `/ui` static frontend.** `app.mount("/ui", StaticFiles(...))`
serves `static/index.html` (and any other static files I add later)
directly, so the same FastAPI process serves both the JSON API and a
human-friendly web page, without needing a second server. The page itself
is plain HTML/CSS/JavaScript — no framework, no build step — using
`fetch()` to call `/health` and `/predict` from the browser and rendering
whatever comes back.

---

## 6. Part E — what Docker is doing here

**The mental model:** a Dockerfile is a recipe for building an image (a
frozen snapshot of an environment — OS, Python, installed packages, my
code). A container is a running instance of that image. Docker Compose
(`docker-compose.yml`) is a shortcut so I don't have to remember a long
`docker run` command with all its flags every time.

**Why training happens outside Docker, not inside it.** The Dockerfile
only copies `api/`, `monitoring/`, `model/`, and `static/` — never `data/`
or `src/`. That's intentional: the container's only job is to *serve* a
model that already exists as `model/model.joblib`. Training happens once,
on my machine, via `python src/train.py`, producing that file; Docker then
just packages and serves it. This keeps the image small (no need to ship
the raw dataset or training libraries-that-aren't-needed-at-inference-time
inside it) and keeps "training" and "serving" as clearly separate
concerns — which is exactly how real ML services are usually built.

**Why `.dockerignore` exists.** Even though the Dockerfile only explicitly
`COPY`s specific folders, `.dockerignore` is still worth having: it stops
things like `__pycache__/`, `.git/`, or a stray virtual environment folder
from ever being sent to the Docker daemon in the first place (the "build
context"), which keeps builds faster and avoids accidentally leaking
anything into an intermediate build layer.

**No secrets in the image.** There's nothing to leak here — the model
doesn't call any external API or database that would need a password or
key. `docker-compose.yml` has a placeholder `APP_ENV` environment variable
just to show where real configuration (if this grew to need any) would be
injected at *run* time via `-e` flags or an env file, rather than being
hardcoded into the image at *build* time.

---

## 7. Part F — what "basic monitoring" means here, and its real limits

`monitoring/monitoring.py` defines a `MonitoringStats` class holding plain
Python counters (`total_requests`, `successful_requests`, a
`collections.Counter` for the prediction class distribution, a
`collections.deque` for recent latencies) protected by a `threading.Lock`
so concurrent requests don't corrupt the counts. `/monitoring` in
`api/main.py` just returns a snapshot of these counters as JSON.

**The honest limitation:** this is all in-memory. Restart the API process
(or the Docker container) and every counter resets to zero. That's fine
for what this assignment asks — a real production system would usually
push these same kinds of metrics to something like Prometheus, or log them
to a persistent store, so history survives restarts and multiple container
replicas can be aggregated together. I'm noting the gap here on purpose
rather than pretending this is more durable than it is.

**Why raw patient input is never logged.** `logger.info(...)` calls in
`monitoring.py` only ever include the prediction, probability, latency,
and model version — never the raw `PatientFeatures` (age, glucose level,
etc.). Even though this is synthetic/public data with no real privacy
stakes, I built it the way I'd want a real health-data service to behave:
logs and metrics should never casually contain the exact input someone
submitted.

---

## 8. Part G — what "drift" means, and what my simulation actually shows

**Data drift** = the *input* data the model sees in production looks
statistically different from the data it was trained on (different means,
different category proportions), regardless of whether the model's
predictions are still accurate.

**Performance degradation** = the model's actual predictive accuracy gets
worse — which can happen *because of* drift, but isn't the same thing.
You can have drift with no performance drop (the shift happens to be along
a direction the model doesn't care about), or a performance drop with very
little visible drift (e.g. the world genuinely got noisier).

**What `src/drift_check.py` does:** it takes the real, untouched test set
and deliberately shifts several features to simulate an older, higher-risk
population — age +15 years, glucose level ×1.3, bmi +5, more diagnosed
hypertension, more current smokers — while keeping the original stroke
labels attached to each row (see the caveat spelled out in
`drift_check.py`'s own docstring: this means "performance on shifted data"
here is a controlled check of model *behavior*, not a real clinical claim,
since changing someone's age would really change their true risk too, not
just the model's guess about their risk).

**What actually happened when I ran it** — real numbers are in
`monitoring/drift_report.json` and discussed with reasoning in
`observations.md`. The short version: the input features shifted
substantially (clear data drift), the model's predicted probabilities
shifted upward in response (so the model *did* notice), but its 0.5-
threshold classification outputs barely changed and its ROC-AUC dropped
only modestly — because this model's precision/recall were already at
zero on the original test set (same imbalance problem as Assignment 1),
so there wasn't much "classification performance" left to degrade further
at the default threshold. That's a genuinely useful, slightly
counterintuitive finding, not a flaw in the simulation — full reasoning is
in `observations.md`.

---

## 9. Design choices I made on purpose (things I could have done differently)

- **Model choice:** I picked Random Forest to tune, not XGBoost or
  Gradient Boosting, even though those had competitive test ROC-AUC in
  Assignment 1. Random Forest ships with scikit-learn (one less dependency
  to pin/install in Docker), is easy to reason about, and had the best
  test-set ROC-AUC generalization of the "no extra library needed" models
  in Assignment 1's benchmark.
- **No threshold tuning:** every model here (and in Assignment 1) still
  predicts "no stroke" for almost everyone at the default 0.5 probability
  cutoff, because stroke cases are so rare. Lowering the decision threshold
  (e.g. flagging anyone above 20% predicted probability, not 50%) would
  likely raise recall substantially at the cost of more false positives.
  I didn't implement that here to keep the API's behavior simple and
  literally matching what `metadata.json` reports, but it's the natural
  next improvement if this were a real clinical tool.
- **No retraining-on-drift logic:** Part G asks me to *detect and
  document* drift, not to automatically retrain when it's found. Real
  systems sometimes do trigger retraining pipelines from drift alerts —
  out of scope here on purpose.
