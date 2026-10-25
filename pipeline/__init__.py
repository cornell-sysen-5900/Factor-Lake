"""WRDS data pipeline: raw WRDS/FRED downloads -> cleaned Parquet files for Factor-Lake.

Run from the repo root:
    uv run --group pipeline python -m pipeline.run --data-dir "<data folder>" --stage all

See pipeline/README.md for the rules every stage follows and pipeline/PLAN.md for the stages.
"""
