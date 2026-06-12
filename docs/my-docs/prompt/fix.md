remove train, val, test split completely and  apply feature engineering on entire dataset, also do following changes in feature enginerring. 

- at very first merge, user and item related features in transaction table 

- Rename item_recent_to_lifetime_ratio  to item_recent_to_last_180d_ratio, since Numerator is item 30d; denominator is category 180d.

- for item_seasonality_strength, compare 7 days pop with same 7 days last year

- let there be user_color_pref_1y_rank1 and user_color_pref_1y_rank2 only, not 3.

- for, txn_month_sin are txn_month_cos, you need to use that row's purchase date's month.

- every window/decay/recency formula should be wrt to transaction date of respective row, as we are calculating at that point of time of purchase along with consiering, purchase histroy before that, not wrt cutoff date. like wise, make sure, all user and item's related feature should be engineered wrt respective that rows purchase date, no features should be caluclated wrt cutoff date, it doesnt make sense, as it global value at the end, but while training model row by row, we need value considering history upto that row's purchase date. 

- no need of feature_cutoff feature, remove. and there is nothing like feature cutoff.

once done, do smoke test to see if eveyrhint is working