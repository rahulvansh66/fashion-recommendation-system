"""
⚠️ REFERENCE PROJECT DISCLAIMER ⚠️

THIS IS ARCHIVED/REFERENCE CODE FROM A PREVIOUS IMPLEMENTATION

- DO NOT USE unless explicitly asked to reference old code
- CURRENT IMPLEMENTATION is in system-design/ directory
- This file is for REFERENCE ONLY to understand legacy approaches
- All new development should follow current system design specifications
"""

import polars as pl


def extract_articles_df() -> pl.DataFrame:
    return pl.read_csv("https://repo.hops.works/dev/jdowling/h-and-m/articles.csv", try_parse_dates=True)


def extract_customers_df() -> pl.DataFrame:
    return pl.read_csv("https://repo.hops.works/dev/jdowling/h-and-m/customers.csv", try_parse_dates=True)


def extract_transactions_df() -> pl.DataFrame:
    return pl.read_csv(
        "https://repo.hops.works/dev/jdowling/h-and-m/transactions_train.csv", try_parse_dates=True
    )

"""
⚠️ END OF REFERENCE PROJECT FILE ⚠️

Remember: This is archived code. Use system-design/ for current implementation.
"""
