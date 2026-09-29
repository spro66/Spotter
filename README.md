# Spotter
# Freight Rate ML Pipeline

Predicts freight `posted_rate` for the given dataset — training, scoring new data, and filling fixed-format submission templates, all through one script and CLI.

---



## 1. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```


 Files in this project

     a. Input data schema

        Training/validation data is expected to have these columns:

        ```
        load_id, pickup, delivery, pickup_lat, pickup_lon, delivery_lat,
        delivery_lon, distance, equipment, weight, date, market_index,
        quote_signal, posted_rate
        ```

        - `posted_rate` is the target — required for **training**, absent for **prediction**.
        - If your column names differ, pass `--column-map '{"raw_name": "expected_name"}'` (any subcommand) to rename before processing.

---

    b. Run instructions

        The CLI has three subcommands: `train`, `predict`, `fill-template`.

        Train a model

        ```bash
        python freight_pipeline.py train \
            --data train-test.csv \
            --target posted_rate \
            --output freight_rate_model.joblib
        ```

        Produces `freight_rate_model.joblib` (the fitted pipeline) and `freight_rate_model.metrics.json` (test metrics, feature importances, and the date-feature ablation results).


    c. Predict on new data (reduced output: `id, predicted_rate`)

        ```bash
        python freight_pipeline.py predict \
            --model freight_rate_model.joblib \
            --data validation.csv \
            --output validation_predictions.csv
        ```

        Use `--id-column` if your identifier column isn't named `load_id`.

### 3. Fill a fixed-format template in place (all original columns preserved)

```bash
python freight_pipeline.py fill-template \
    --model freight_rate_model.joblib \
    --template december_chart_inputs.csv \
    --output december_chart_inputs.csv
```

Use this instead of `predict` whenever a downstream consumer (a scorer, a chart, a fixed submission format) requires every original column and column order to survive untouched — `predict` intentionally reduces its output to just `[id, predicted_rate]`, which will break a strict format check.



## 4. Quick end-to-end example

```bash
pip install -r requirements.txt

python freight_pipeline.py train --data train-test.csv --output freight_rate_model.joblib
python freight_pipeline.py predict --model freight_rate_model.joblib --data validation.csv --output validation_predictions.csv
python freight_pipeline.py fill-template --model freight_rate_model.joblib --template december_chart_inputs.csv --output december_chart_inputs.csv
```

