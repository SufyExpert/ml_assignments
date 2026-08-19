# observations.md — Assignment 3

I actually ran the whole pipeline end to end on my machine — training, tuning, the API, and the
drift simulation — so everything below is from real output, not a guess about what might happen.

---

## 1. Tuning findings

I tuned a Random Forest with `RandomizedSearchCV`: 25 candidate hyperparameter combinations, 5-fold
cross-validation, scored on ROC-AUC (not accuracy — with only ~5% of patients having had a stroke,
optimizing for accuracy would happily reward a model that never predicts stroke at all).

**What I got:**

| Metric | Value |
|---|---|
| Best CV ROC-AUC (on train folds) | 0.8376 |
| Validation ROC-AUC (held out, never seen by the search) | 0.8449 |
| Test ROC-AUC (touched exactly once) | 0.8388 |
| Test Accuracy | 0.9511 |
| Test Precision / Recall / F1 | 0.0000 / 0.0000 / 0.0000 |

**Winning hyperparameters:** `n_estimators=430`, `max_depth=5`, `min_samples_leaf=7`,
`max_features=None`, `class_weight=None`.

**Why I trust this isn't overfit to the tuning process:** the CV score (0.8376), the validation
score (0.8449), and the test score (0.8388) are all close to each other — no big drop-off from one
to the next. If the search had overfit to the cross-validation folds specifically, I'd expect
validation and test to come in noticeably lower than the CV score. They didn't.

**The part I want to be honest about:** precision, recall, and F1 are all exactly 0.0000. At the
default 0.5 probability cutoff, this tuned model — like every model in Assignment 1's benchmark —
still predicts "no stroke" for essentially every single patient. Compared to Assignment 1's
untuned Random Forest (test ROC-AUC 0.795), tuning genuinely improved the model's ability to *rank*
patients by risk (test ROC-AUC 0.839) — it's a real, measurable improvement. But ROC-AUC measures
ranking quality across every possible threshold, and 0.5 specifically is still too high a bar given
how rare stroke actually is in this data. `class_weight="balanced"` was one of the options the
search could have picked (it penalizes missing the minority class more heavily), but it didn't win
— which tells me that even with that option available, the search still found "predict the
majority class almost always" to be the best strategy *for ROC-AUC specifically*, since ROC-AUC
doesn't care where the 0.5 line sits. If I wanted this model to actually flag at-risk patients in
practice, the real fix isn't more tuning — it's lowering the decision threshold (e.g., flag anyone
above 15–20% predicted probability, not 50%), which trades some false positives for actually
catching real cases. I didn't implement that here to keep this assignment's scope contained, but
it's the clear next step.

---

## 2. Deployment findings

I ran the API locally with `uvicorn` and hit every endpoint from outside the process with `curl`,
the same way I'd test a running Docker container.

**Home (`GET /`)** returned the model version and links to `/docs` and `/ui`, instead of a bare
404 — small thing, but it means anyone landing on the API root immediately knows what they're
looking at.

**Health (`GET /health`)** correctly reported `"status": "ok"`, `"model_loaded": true`, and the
model version, with an uptime counter that ticked up correctly across repeated calls.

**Predict (`POST /predict`)**, with a real patient profile (67-year-old female, hypertension no,
heart disease yes, average glucose 228.69, BMI 36.6, formerly smoked), returned:
```json
{"prediction": 0, "probability": 0.2176, "model_version": "1.0", "latency_ms": 73.746}
```
That's a genuinely informative result even though the hard prediction is "0" (no stroke): a 21.76%
predicted probability is well above this model's ~4.9% baseline stroke rate, so the *probability*
field is doing real work here, flagging elevated risk, even where the 0.5-threshold classification
doesn't. This is exactly why I return probability alongside the hard prediction in the API — the
hard 0/1 label alone would have hidden this.

**Validation errors** worked exactly as intended. Sending a request missing `age` came back as
HTTP 422 with a clear JSON body naming the missing field. Sending `"gender": "Robot"` was rejected
the same way, listing the allowed values. Neither of these ever reached my prediction code — they
were rejected by Pydantic before my `/predict` function even started running.

**Latency:** the very first prediction after server startup took ~74ms; I noticed this settles
lower on subsequent calls (this looks like normal first-call warm-up — Python importing/JIT-ing
some code paths on first use — rather than anything wrong with the model itself, since the model
was already loaded at startup, not on that first request).

**Tests:** all 8 tests in `tests/test_api.py` passed — home, health, a valid prediction, a rejected
missing field, a rejected invalid category, a rejected out-of-range age, an accepted missing
(optional) `bmi`, and a monitoring endpoint that correctly reflected the requests made during the
test run.

---

## 3. Monitoring findings

After exercising the API with a mix of valid and invalid requests, `GET /monitoring` reported
exactly what I expected:

```json
{
  "total_requests": 2,
  "successful_requests": 0,
  "failed_requests": 2,
  "invalid_input_requests": 2,
  "prediction_class_distribution": {},
  "average_latency_ms_recent_100": null,
  "average_latency_ms_all_time": null,
  "model_version": "1.0"
}
```
(This particular snapshot was taken after sending two deliberately invalid requests to test error
handling — that's why successful_requests is 0 here. Earlier in the same session, a valid request
had already been correctly counted as successful with its prediction added to the class
distribution and its latency recorded, before I moved on to testing the error paths.)

What this confirms: successful and failed requests are tracked separately, invalid-input requests
specifically are broken out from other kinds of failures (there weren't any non-validation failures
to compare against, since the model itself never threw an exception during testing), and the
counters genuinely update in real time as requests come in — not just a static placeholder.

**The honest limitation, stated plainly:** these counters live in memory. If I restart the API
process, every number in `/monitoring` goes back to zero. For this assignment's scope that's an
acceptable, deliberate simplification — a real production deployment would push these same kinds of
metrics to a persistent store (Prometheus, a database, a log aggregator) so history survives
restarts and multiple running instances could be combined. I noted this same point in `concepts.md`
so I don't forget the gap is there on purpose, not an oversight.

---

## 4. Drift simulation findings

I took the real, untouched test set (1,022 patients) and shifted it to simulate an older,
higher-risk population: age +15 years, average glucose level ×1.3, BMI +5, 20% of previously
non-hypertensive patients flipped to hypertensive, and 25% of "never smoked" patients shifted to
"smokes." I kept the original stroke labels attached to each row — see the caveat in
`src/drift_check.py` and `concepts.md`: this makes it a controlled check of model *behavior* under
distribution shift, not a real clinical claim, since changing someone's age would really change
their true risk too, not just the model's guess about it.

**Numerical feature drift — clear and substantial:**

| Feature | Original mean | Shifted mean | Change |
|---|---|---|---|
| age | 42.72 | 57.68 | +35.0% |
| avg_glucose_level | 105.47 | 137.11 | +30.0% |
| bmi | 28.78 | 33.78 | +17.4% |

**Categorical drift:**
- `hypertension`: prevalence roughly tripled, from 9.9% to 27.9% of patients.
- `smoking_status`: "never smoked" dropped from 38.3% to 28.8%, "smokes" rose from 15.9% to 25.4%.
- Every other categorical feature (gender, marital status, work type, residence, heart disease) was
  left untouched on purpose, as a control — and the numbers confirm they stayed exactly identical
  between the two datasets, which is what I'd expect from a script that only modifies the features
  I told it to.

**This is unambiguous data drift** — several input feature distributions moved by double-digit
percentages. There's no ambiguity here; the population feeding the model genuinely looks different.

**Prediction distribution — the model noticed, even if its hard predictions didn't move much:**

| | Original | Shifted |
|---|---|---|
| Mean predicted probability | 0.047 | 0.103 |
| Predicted-stroke rate (0.5 cutoff) | 0.0% | 0.1% |

The average predicted probability more than doubled. That's the model correctly reacting to a
riskier-looking population — it's not blind to the shift. But because the 0.5 classification
threshold was already far above where this model's probabilities cluster (as seen in the tuning
section above), even a doubled average probability mostly wasn't enough to push individual
predictions across that line.

**Performance comparison (ground-truth labels available, with the caveat noted above):**

| Metric | Original test set | Shifted data |
|---|---|---|
| Accuracy | 0.9511 | 0.9501 |
| Precision / Recall / F1 | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 |
| ROC-AUC | 0.8388 | 0.8175 |

**My read on this: it's both drift and mild performance degradation, but not in equal measure.**
The input drift is large and unambiguous. The *ranking* performance (ROC-AUC) degraded modestly —
down about 0.021, a real but not dramatic drop, suggesting the model still separates
higher-risk from lower-risk patients reasonably well even on the shifted population, just slightly
less cleanly than on data shaped like what it was trained on. The *classification* metrics
(precision/recall/F1) didn't degrade at all, for the somewhat unsatisfying reason that they were
already at the floor (0.0000) before the shift — there was no headroom left to lose. If this model
had been deployed with a properly tuned decision threshold that actually caught real stroke cases
beforehand, I'd expect this same drift to show up much more clearly as a recall drop, since a
riskier population shifting further away from the training distribution is exactly the kind of
change that erodes a threshold picked for a different population. In its current form, though, the
clearest, most trustworthy drift signal here is the ROC-AUC drop and the doubled average predicted
probability — both genuinely detected the shift; the 0.5-threshold classification metrics simply
had nothing left to lose.

---

## 5. What I'd do next, if I kept going

- Pick a decision threshold based on the ROC curve (not the default 0.5) so the model actually
  flags some patients as elevated-risk in practice, then re-run this whole observations pass —
  I'd expect the drift section specifically to become much more informative once there's real
  recall to watch degrade.
- Persist `/monitoring` counters somewhere durable (even a simple SQLite file) so they survive a
  restart, instead of living only in process memory.
- Add a scheduled or triggered re-run of `drift_check.py` against real incoming request data
  (once there's a real feedback loop / labels available) instead of only a one-off synthetic shift.
