"""Pipeline stages, run in this order. Each module has STAGE and run(paths) -> summary dict."""

STAGES = (
    "s00_inventory",
    "s01_membership",
    "s02_prices",
    "s03_fundamentals_q",
    "s04_fundamentals_a",
    "s05_ibes",
    "s06_market",
    "s07_panel",
    "s08_export",
    "s09_validate",
)
