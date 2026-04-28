# Databricks notebook source
# MAGIC %md
# MAGIC # Homework #5 — F1 Model Deployment

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuration

# COMMAND ----------

# MAGIC %pip install --upgrade --force-reinstall mlflow typing_extensions

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install mlflow

# COMMAND ----------

import mlflow
import mlflow.sklearn

# COMMAND ----------

USER    = "jw4853"         
CATALOG = "gr5069"
SCHEMA  = USER
VOLUME  = "takehome"

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

RF_TABLE_NAME = "predictions_random_forest"
GB_TABLE_NAME = "predictions_gradient_boosting"

TABLE_RF      = f"{CATALOG}.{SCHEMA}.{RF_TABLE_NAME}"
TABLE_GB      = f"{CATALOG}.{SCHEMA}.{GB_TABLE_NAME}"
TABLE_RF_PATH = f"{VOLUME_PATH}/{RF_TABLE_NAME}"
TABLE_GB_PATH = f"{VOLUME_PATH}/{GB_TABLE_NAME}"

DATA_PATH = "/Volumes/gr5069/raw/f1_data"

print(f"Schema:   {CATALOG}.{SCHEMA}")
print(f"Volume:   {VOLUME_PATH}")
print(f"RF table: {TABLE_RF}  →  {TABLE_RF_PATH}")
print(f"GB table: {TABLE_GB}  →  {TABLE_GB_PATH}")

# COMMAND ----------

# Spark imports (matching the lab style)
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import col, when

# Python ML stack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create schema + takehome volume (Rubric #1 setup)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
spark.sql(f"USE {CATALOG}.{SCHEMA}")

print(f"Schema ready: {CATALOG}.{SCHEMA}")
print(f"Volume ready: {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load F1 data with Spark
# MAGIC
# MAGIC Same pattern as the Airbnb lab: `spark.read.csv(..., header=True, inferSchema=True)`.

# COMMAND ----------

import os
print("Files in volume:")
for f in sorted(os.listdir(DATA_PATH)):
    print(" -", f)

# COMMAND ----------

resultsDF      = spark.read.csv(f"{DATA_PATH}/results.csv",      header=True, inferSchema=True)
racesDF        = spark.read.csv(f"{DATA_PATH}/races.csv",        header=True, inferSchema=True)
driversDF      = spark.read.csv(f"{DATA_PATH}/drivers.csv",      header=True, inferSchema=True)
constructorsDF = spark.read.csv(f"{DATA_PATH}/constructors.csv", header=True, inferSchema=True)

print("results:     ", resultsDF.count())
print("races:       ", racesDF.count())
print("drivers:     ", driversDF.count())
print("constructors:", constructorsDF.count())

# COMMAND ----------

display(resultsDF.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Spark joins + feature engineering
# MAGIC
# MAGIC **Target:** `podium` = 1 if driver finished in positions 1–3, else 0.
# MAGIC We use `positionOrder` (always populated) instead of `position` (NaN for DNFs).

# COMMAND ----------

# Join all four tables in Spark
joinedDF = (
    resultsDF
    .join(
        racesDF.select("raceId", "year", "round", "circuitId"),
        on="raceId", how="left",
    )
    .join(
        driversDF.select(col("driverId"), col("nationality").alias("driver_nationality")),
        on="driverId", how="left",
    )
    .join(
        constructorsDF.select(col("constructorId"), col("nationality").alias("constructor_nationality")),
        on="constructorId", how="left",
    )
)

# Cast + create target column (matches lab's withColumn(...cast()) idiom)
modelDF = (
    joinedDF
    .withColumn("grid",   col("grid").cast(DoubleType()))
    .withColumn("podium", (col("positionOrder") <= 3).cast(IntegerType()))
    .dropna(subset=["grid", "year", "round", "circuitId", "constructorId", "driverId"])
    .select(
        # identifiers (kept for the predictions table)
        "raceId", "driverId", "constructorId", "year",
        # features
        "grid", "round", "circuitId",
        "driver_nationality", "constructor_nationality",
        # target
        "podium",
    )
)

print("Rows after cleaning:", modelDF.count())
display(modelDF.limit(5))

# COMMAND ----------

# Class balance check
display(
    modelDF.groupBy("podium").count()
           .withColumn("pct", col("count") / modelDF.count())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Train/test split with Spark `randomSplit`
# MAGIC
# MAGIC Matches the lab's pattern. Seed fixes the split so it's reproducible.

# COMMAND ----------

(trainDF, testDF) = modelDF.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {trainDF.count():,}")
print(f"Test:  {testDF.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Hand off to pandas + sklearn
# MAGIC
# MAGIC Matches the lab's pattern: `.toPandas()` once at the sklearn boundary.

# COMMAND ----------

trainPDF = trainDF.toPandas()
testPDF  = testDF.toPandas()

# Label-encode the two string columns. Fit on combined train+test categories
# so the test set never sees an unknown category at predict time.
cat_cols = ["driver_nationality", "constructor_nationality"]
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([trainPDF[c], testPDF[c]]).astype(str))
    trainPDF[c] = le.transform(trainPDF[c].astype(str))
    testPDF[c]  = le.transform(testPDF[c].astype(str))
    encoders[c] = le

feature_cols = [
    "grid", "year", "round", "circuitId",
    "driver_nationality", "constructor_nationality",
]
id_cols = ["raceId", "driverId", "constructorId", "year"]

X_train, y_train = trainPDF[feature_cols], trainPDF["podium"]
X_test,  y_test  = testPDF[feature_cols],  testPDF["podium"]

print("X_train:", X_train.shape, "  X_test:", X_test.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Helpers: artifact plots + write-to-takehome

# COMMAND ----------

def save_confusion_matrix(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Not podium", "Podium"],
        yticklabels=["Not podium", "Podium"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_feature_importance(model, feature_names, path, title):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(importances)), importances[order])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(np.array(feature_names)[order], rotation=45, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_predictions_table(pred_pdf, table_fqn, table_path):
    """Convert pandas predictions back to Spark, write Delta to /takehome, register catalog table."""
    sdf = spark.createDataFrame(pred_pdf)
    spark.sql(f"DROP TABLE IF EXISTS {table_fqn}")
    (
        sdf.write
           .format("delta")
           .mode("overwrite")
           .option("overwriteSchema", "true")
           .option("path", table_path)
           .saveAsTable(table_fqn)
    )
    print(f"Wrote {sdf.count():,} rows")
    print(f"  catalog table: {table_fqn}")
    print(f"  files at:      {table_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Model 1 — Random Forest (MLflow run)

# COMMAND ----------

with mlflow.start_run(run_name="random_forest_podium") as run:
    rf_params = {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    mlflow.log_params(rf_params)

    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)

    y_pred_rf  = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    rf_metrics = {
        "accuracy":  accuracy_score(y_test, y_pred_rf),
        "precision": precision_score(y_test, y_pred_rf),
        "recall":    recall_score(y_test, y_pred_rf),
        "f1":        f1_score(y_test, y_pred_rf),
    }
    mlflow.log_metrics(rf_metrics)
    print("RF metrics:", rf_metrics)

    mlflow.sklearn.log_model(rf, artifact_path="model")

    cm_path = "/tmp/rf_confusion_matrix.png"
    fi_path = "/tmp/rf_feature_importance.png"
    save_confusion_matrix(y_test, y_pred_rf, cm_path, "Random Forest — Confusion Matrix")
    save_feature_importance(rf, feature_cols, fi_path, "Random Forest — Feature Importance")
    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(fi_path)

    rf_run_id = run.info.run_id
    print("RF run_id:", rf_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Random Forest predictions → Spark → Delta in /takehome (Rubric #3)
# MAGIC
# MAGIC Lab pattern: add `prediction` column to the test pandas DF, hand back to Spark with
# MAGIC `spark.createDataFrame()`, then save.

# COMMAND ----------

rf_pred_pdf = testPDF[id_cols].copy()
rf_pred_pdf["actual_podium"]    = y_test.values
rf_pred_pdf["predicted_podium"] = y_pred_rf
rf_pred_pdf["predicted_proba"]  = y_proba_rf
rf_pred_pdf["model_run_id"]     = rf_run_id

def save_confusion_matrix(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Not podium", "Podium"],
        yticklabels=["Not podium", "Podium"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_feature_importance(model, feature_names, path, title):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(importances)), importances[order])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(np.array(feature_names)[order], rotation=45, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_predictions_table(pred_pdf, table_path):
    """Write predictions as a Delta table to the /takehome volume."""
    sdf = spark.createDataFrame(pred_pdf)
    (
        sdf.write
           .format("delta")
           .mode("overwrite")
           .option("overwriteSchema", "true")
           .save(table_path)
    )
    print(f"Wrote {sdf.count():,} rows to {table_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Model 2 — Gradient Boosting (MLflow run)

# COMMAND ----------

with mlflow.start_run(run_name="gradient_boosting_podium") as run:
    gb_params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.9,
        "random_state": 42,
    }
    mlflow.log_params(gb_params)

    gb = GradientBoostingClassifier(**gb_params)
    gb.fit(X_train, y_train)

    y_pred_gb  = gb.predict(X_test)
    y_proba_gb = gb.predict_proba(X_test)[:, 1]

    gb_metrics = {
        "accuracy":  accuracy_score(y_test, y_pred_gb),
        "precision": precision_score(y_test, y_pred_gb),
        "recall":    recall_score(y_test, y_pred_gb),
        "f1":        f1_score(y_test, y_pred_gb),
    }
    mlflow.log_metrics(gb_metrics)
    print("GB metrics:", gb_metrics)

    mlflow.sklearn.log_model(gb, artifact_path="model")

    cm_path = "/tmp/gb_confusion_matrix.png"
    fi_path = "/tmp/gb_feature_importance.png"
    save_confusion_matrix(y_test, y_pred_gb, cm_path, "Gradient Boosting — Confusion Matrix")
    save_feature_importance(gb, feature_cols, fi_path, "Gradient Boosting — Feature Importance")
    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(fi_path)

    gb_run_id = run.info.run_id
    print("GB run_id:", gb_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Gradient Boosting predictions → Spark → Delta in /takehome (Rubric #3)

# COMMAND ----------

gb_pred_pdf = testPDF[id_cols].copy()
gb_pred_pdf["actual_podium"]    = y_test.values
gb_pred_pdf["predicted_podium"] = y_pred_gb
gb_pred_pdf["predicted_proba"]  = y_proba_gb 
gb_pred_pdf["model_run_id"]     = gb_run_id   

write_predictions_table(gb_pred_pdf, TABLE_GB_PATH)
display(spark.read.format("delta").load(TABLE_GB_PATH).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Verify both tables (catalog + volume)

# COMMAND ----------

gb_pred_pdf = testPDF[id_cols].copy()
gb_pred_pdf["actual_podium"]    = y_test.values
gb_pred_pdf["predicted_podium"] = y_pred_gb
gb_pred_pdf["predicted_proba"]  = y_proba_gb
gb_pred_pdf["model_run_id"]     = gb_run_id

write_predictions_table(gb_pred_pdf, TABLE_GB_PATH)
display(spark.read.format("delta").load(TABLE_GB_PATH).limit(5))

# COMMAND ----------

print(f"Files in {VOLUME_PATH}:")
for f in sorted(os.listdir(VOLUME_PATH)):
    print(" -", f)

print("\nRandom Forest predictions table:")
display(spark.read.format("delta").load(TABLE_RF_PATH).limit(5))

print("\nGradient Boosting predictions table:")
display(spark.read.format("delta").load(TABLE_GB_PATH).limit(5))

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS gr5069.jw4853.predictions_random_forest")
spark.read.format("delta").load("/Volumes/gr5069/jw4853/takehome/predictions_random_forest") \
    .write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("gr5069.jw4853.predictions_random_forest")

spark.sql("DROP TABLE IF EXISTS gr5069.jw4853.predictions_gradient_boosting")
spark.read.format("delta").load("/Volumes/gr5069/jw4853/takehome/predictions_gradient_boosting") \
    .write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("gr5069.jw4853.predictions_gradient_boosting")

print("Done!")
spark.sql("SHOW TABLES IN gr5069.jw4853").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Quick comparison

# COMMAND ----------

comparison = pd.DataFrame({"random_forest": rf_metrics, "gradient_boosting": gb_metrics})
print(comparison.round(4))
