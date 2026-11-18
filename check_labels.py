import pandas as pd

files = [
    "evaluator_candidates.parquet",
    "results/candidate_predictions_evaluator.parquet"
]

for f in files:
    df = pd.read_parquet(f)
    print("\nFILE:", f)
    print("Rows:", len(df))

    for c in ["is_match", "ref_arc_id", "prediction", "prediction_probability"]:
        if c in df.columns:
            print(c, "non-null:", df[c].notna().sum(),
                  "null:", df[c].isna().sum())

    if "prediction" in df.columns:
        print("Positive predictions:", int(df["prediction"].sum()))
