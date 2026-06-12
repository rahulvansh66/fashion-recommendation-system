create notebook for feature engineering to capture trend, seasonality, recency and other features that you find useful to train/experiment with catboost and lightgbm as ranker model.

here are some of my suggestions, you can sugget more features :

Features related to summarized older behavior.
| Feature                             |      Lookback | Purpose                   |
| ----------------------------------- | ------------: | ------------------------- |
| `item_pop_7d`                       |        7 days | Current trend             |
| `item_pop_30d`                      |       30 days | Recent popularity         |
| `item_category_pop_30d`                      |       30 days | Recent popularity of item's category     |
| `item_pop_180d`                     |      6 months | Stable demand             |
| `item_category_pop_180d`           |        6 months | Stable popularity of item's category     |
| `item_pop_same_month_last_year`     | previous year | Seasonality               |
| `item_recent_to_lifetime_ratio` = item_pop_30d / item_category_pop_180d |  | item trend strength |
| `item_category_recent_to_lifetime_ratio` = item_category_pop_30d / item_category_category_pop_180d |  | category trend strength |
| `item_category_recent_to_lifetime_ratio` = item_pop_7d / item_pop_same_month_last_year |  | Seasonality strength |
| `user_category_pref_1y_rank1`       |        1 year | user most bought category (garment_group_name) |
| `user_category_pref_1y_rank2`       |        1 year | user second most bought category |
| `user_category_pref_1y_rank3`       |        1 year | user third most bought category |
| `user_color_pref_1y_rank1`         |      1 year | most preferred color        |
| `user_color_pref_1y_rank2`         |      1 year | second most preferred color        |
| `user_days_since_last_purchase`         | all history | activity recency        |
| `user_purchase_count_30d`         | 30 days | user recent purchase frequency        |
| `user_purchase_count_180d`         | 6 months | user stable purchase frequency        |
| `user_item_repurchase`      | all history | No of times user repurchase current item       |


| `txn_month_sin` = sin(month x 2pi / 12) |  | Cyclical encoding so December and January are close in feature space. Applied this transformation on `t_dat`. Since its continues it'll follow intuition of after 12th month next month will be 1st month, rather thinking both are far a part which is case if we just take month|
| `txn_month_cos` = cos(month x 2pi / 12) |  | Paired with `month_sin` to preserve seasonal continuity without treating month as an ordinal integer. |

note: always Use garment_group_name for category 


Use decayed historical features:
- Recent transactions receive high weight. Older transactions still contribute, but less.
- This lets older transactions influence the model without dominating recent intent.
- Do **not** duplicate hard-window count features (`item_pop_7d/30d/180d`, `user_purchase_count_*`) with decayed pops — use decay only where hard windows are a poor fit (repurchase timing, price preference, personal-vs-global demand).
- All decayed features below use **half-life = 180 days** at observation date `T₀`. Only transactions with `t_dat < T₀` are included (no label leakage).


**Decay weight** (per transaction):

```
days_ago = T₀ - t_dat

w = exp( -ln(2) × days_ago / 180 )
same as
w = e^(-λ × days_ago)
λ = ln(2) / 180
same as
w = 2^(-days_ago / 180)
```

180-day half-life means: after every 180 days, weight is halved.

```
days_ago = 0   →  w = 2^0 = 1.0
days_ago = 180  →  w = 2^(-180/180) = 2^(-1) = 0.5 
days_ago = 360  →  w = 2^(-2) = 0.25
```

| Feature | Description | Formula |
| ------- | ----------- | ------- |
| `user_item_decayed_repurchase` | Recency-weighted repurchase of the candidate item; complements hard `user_item_repurchase` | `Σ w` over user purchases where `article_id` = candidate item |
| `user_item_decayed_interaction_ratio` | Personal vs global demand — user recently re-buys this item relative to how popular it is overall | `user_item_decayed_repurchase / (item_pop_180d + 1)` |
| `user_decayed_price_avg` | User's recent price preference (recent purchases weighted more) | `Σ (price × w) / Σ w` over all user purchases before `T₀` |
| `user_decayed_price_std` | Spread of user's recent price preference; used to compute price z-score | `sqrt( Σ (w × (price - user_decayed_price_avg)²) / Σ w )`; use `max(std, 1e-6)` if near zero |
| `user_item_price_decayed_zscore` | How candidate item price fits user's recent budget (positive = above typical spend) | `(candidate_price - user_decayed_price_avg) / user_decayed_price_std` |
| `user_days_since_last_purchase` | Hard recency companion — days since user's most recent purchase of any item (see table above) | `T₀ - max(t_dat)` over all user purchases before `T₀` |