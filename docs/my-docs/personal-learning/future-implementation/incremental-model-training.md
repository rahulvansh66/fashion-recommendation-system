Yes — but in industry, this is usually **progressive training / staged scaling**, not blindly “add more sample and continue training forever.”

For your setup:

```text
Stage 1: Two-tower candidate generator
Stage 2: LightGBM ranker
```

the best practice is different for each model.

---

# 1. Core idea

You should separate two concepts:

| Concept                        | Meaning                                                   | Recommended?                   |
| ------------------------------ | --------------------------------------------------------- | ------------------------------ |
| **Progressive data scaling**   | Train on 5%, then 10%, then 25%, then 50%, then full data | Yes                            |
| **Warm-start training**        | Continue training a previous model on more data           | Sometimes                      |
| **Full retraining**            | Rebuild model from scratch on larger data                 | Often best                     |
| **Online incremental updates** | Continuously update model as new data arrives             | Advanced; not needed initially |

For your project, I recommend:

```text
Two-tower model:
    Warm-start allowed and useful.

LightGBM ranker:
    Prefer retraining from scratch on larger candidate data.
    Warm-start only for quick experiments, not final model.
```

LightGBM technically supports continued training with `init_model`, and its Python API exposes `init_model` and `keep_training_booster` for continuing training from an existing model. ([LightGBM Documentation][1]) But for recommender ranking, continued boosting can easily overfit to the newly added sample or become biased toward the old sample. So use it carefully.

---

# 2. Industry-grade training loop

A practical loop should look like this:

```text
1. Train on small deterministic sample.
2. Evaluate on fixed temporal validation set.
3. Identify bottleneck:
   - candidate recall problem?
   - ranker ordering problem?
   - feature problem?
   - cold-user problem?
4. Increase only the part that is bottlenecked.
5. Re-train or warm-start depending on model type.
6. Compare against previous model on the same validation set.
7. Promote only if metrics improve consistently.
```

Do **not** change training sample, validation sample, negative sampling, and features all at once. You will not know what caused the improvement.

---

# 3. Recommended staged scaling plan

Use deterministic user buckets.

```text
bucket = hash(customer_id) % 100
```

Then scale like this:

| Stage | User buckets | Approx data | Purpose                 |
| ----- | -----------: | ----------: | ----------------------- |
| S0    |            0 |          1% | pipeline smoke test     |
| S1    |          0-4 |          5% | feature/debug iteration |
| S2    |          0-9 |         10% | first reliable model    |
| S3    |         0-24 |         25% | stronger model          |
| S4    |         0-49 |         50% | near-final model        |
| S5    |         0-99 |        100% | final training          |

The key rule:

```text
Each larger sample must contain the previous smaller sample.
```

Good:

```text
5% = buckets 0-4
10% = buckets 0-9
25% = buckets 0-24
```

Bad:

```text
5% = random sample A
10% = random sample B
25% = random sample C
```

Why? Because nested samples make results comparable.

---

# 4. Keep validation fixed

Your validation set should stay fixed while you scale training data.

Example:

```text
Validation cutoff:
    Train features until 2020-09-15
    Validation labels from 2020-09-16 to 2020-09-22

Validation users:
    fixed buckets 90-99
```

Then every experiment is compared on the same validation problem.

```text
Model A trained on 5% users  -> evaluate on same validation users
Model B trained on 10% users -> evaluate on same validation users
Model C trained on 25% users -> evaluate on same validation users
```

This avoids fake improvement caused by easier validation samples.

---

# 5. Two-tower incremental training strategy

For the two-tower model, warm-starting is reasonable.

## Stage A: train initial model

Train on 5% users:

```text
users = buckets 0-4
train window = last 12 weeks
embedding_dim = 64
loss = sampled softmax / in-batch negatives
```

Save:

```text
customer tower weights
article tower weights
embedding vocabulary mappings
optimizer state
training config
cutoff date
feature schema
```

## Stage B: expand data and continue training

Train on 10% users:

```text
users = buckets 0-9
initialize model from 5% checkpoint
lower learning rate
train for fewer epochs
shuffle full expanded data
```

Important: do not train only on “new users” from buckets 5-9.

Bad:

```text
Continue training only on new 5% users.
```

Better:

```text
Continue training on full 10% data = old 5% + new 5%.
```

Otherwise, the model may forget or bias toward the latest sample.

---

## Two-tower warm-start recipe

```yaml
two_tower_progressive_training:
  stage_1:
    user_buckets: [0, 1, 2, 3, 4]
    epochs: 5
    learning_rate: 0.001

  stage_2:
    user_buckets: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    init_from: stage_1_checkpoint
    epochs: 2
    learning_rate: 0.0003

  stage_3:
    user_buckets: 0-24
    init_from: stage_2_checkpoint
    epochs: 2
    learning_rate: 0.0001
```

General rule:

```text
As data increases, reduce learning rate and reduce epochs.
```

---

# 6. Two-tower: when to warm-start vs retrain

## Warm-start is good when:

```text
Architecture is unchanged.
Feature schema is unchanged.
User/item ID mappings are stable.
Training objective is unchanged.
New sample is an expansion of the old sample.
Validation Recall@K improves or stays stable.
```

## Retrain from scratch when:

```text
You changed embedding dimensions.
You changed user/item vocabulary handling.
You changed loss function.
You changed negative sampling heavily.
You changed feature encoders.
You changed training window.
You suspect overfitting or bad local optimum.
```

For serious experiments, compare both:

```text
25% warm-started model
vs
25% from-scratch model
```

If from-scratch performs better, use from-scratch for later stages.

---

# 7. LightGBM incremental training strategy

For LightGBM, industry practice is more conservative.

Even though LightGBM supports continued training from an existing model through `init_model`, the safer recommender practice is:

```text
Use warm-start only for fast experiments.
Use full retraining for promoted/final models.
```

Why?

Because LightGBM adds more trees on top of the existing trees. If your new data distribution changes, it does not truly “rethink” earlier splits. It just appends additional trees.

That can cause:

```text
overfitting to old sample,
overfitting to new sample,
poor calibration,
too many trees,
harder comparison between stages,
bias from earlier negative sampling.
```

---

## Recommended LightGBM approach

At each scale, rebuild the LightGBM training table and train from scratch.

```text
Stage 1:
    two-tower top 200 candidates
    50 negatives/user
    train LightGBM from scratch

Stage 2:
    two-tower top 500 candidates
    100 negatives/user
    train LightGBM from scratch

Stage 3:
    two-tower top 1000 candidates
    300 negatives/user
    train LightGBM from scratch
```

This is more expensive than warm-starting, but much safer.

---

# 8. Why LightGBM should usually be retrained

Your ranker dataset changes every time you improve the two-tower model.

For example:

```text
Two-tower v1 candidates:
    user U -> [A, B, C, D]

Two-tower v2 candidates:
    user U -> [A, E, F, G]
```

That means the LightGBM training distribution changed.

The ranker is not just seeing “more rows.” It is seeing a different candidate universe.

So the correct practice is:

```text
When candidate generator changes materially,
rebuild ranker candidates,
relabel candidates,
recompute features,
retrain LightGBM.
```

---

# 9. Recommended incremental workflow for your full system

## Stage 0: smoke test

```yaml
stage: S0_smoke
user_sample: 1%
train_window: 4 weeks
two_tower_top_k: 100
ranker_negatives_per_user: 20
goal:
  - pipeline runs end-to-end
  - no leakage
  - MAP@12 code works
```

Do not optimize heavily here.

---

## Stage 1: small useful model

```yaml
stage: S1_small
user_sample: 5%
train_window: 8 weeks
two_tower_top_k: 200
ranker_negatives_per_user: 50
two_tower_training:
  from_scratch: true
lightgbm_training:
  from_scratch: true
```

Evaluate:

```text
candidate Recall@100
candidate Recall@200
MAP@12
cold-user MAP@12
warm-user MAP@12
```

---

## Stage 2: expanded model

```yaml
stage: S2_medium
user_sample: 10%
train_window: 12 weeks
two_tower_top_k: 500
ranker_negatives_per_user: 100
two_tower_training:
  init_from: S1_small
  learning_rate: lower
lightgbm_training:
  from_scratch: true
```

Important:

```text
Train two-tower on full 10%, not only the new 5%.
Retrain LightGBM from scratch on new candidates.
```

---

## Stage 3: serious model

```yaml
stage: S3_large
user_sample: 25%
train_window: 16 weeks
two_tower_top_k: 500
ranker_negatives_per_user: 200
two_tower_training:
  compare:
    - warm_start_from_S2
    - from_scratch
lightgbm_training:
  from_scratch: true
```

At this point, compare warm-start vs from-scratch two-tower.

---

## Stage 4: final candidate model

```yaml
stage: S4_final_candidate
user_sample: 50% to 100%
train_window: 24 weeks
two_tower_top_k: 1000
ranker_negatives_per_user: 300
two_tower_training:
  from_scratch_or_best_warm_start: based_on_S3_result
lightgbm_training:
  from_scratch: true
```

---

# 10. Do not add new sample only when performance is bad

This is important.

If performance is not good, adding more data may not fix the problem.

First diagnose the bottleneck.

## Case 1: Candidate Recall@K is low

Example:

```text
Recall@500 = 0.18
MAP@12 = 0.015
```

Problem is candidate generation.

Actions:

```text
increase two-tower data,
increase train window,
improve negative sampling,
add user/item features,
increase top-K,
add hard-negative fine-tuning.
```

Do not spend time tuning LightGBM if the correct items are missing from candidates.

---

## Case 2: Candidate Recall@K is high but MAP@12 is low

Example:

```text
Recall@500 = 0.65
MAP@12 = 0.018
```

Problem is ranker.

Actions:

```text
improve LightGBM features,
improve hard-negative sampling,
increase negatives per user,
tune lambdarank parameters,
fix group construction,
check label leakage,
add recency/popularity/user-affinity features.
```

---

## Case 3: Train metric improves but validation metric drops

Problem is overfitting.

Actions:

```text
increase regularization,
reduce epochs,
reduce tree depth/leaves,
use early stopping,
increase validation folds,
check temporal leakage,
reduce overly specific ID features.
```

---

## Case 4: Overall MAP improves but cold-user MAP is poor

Problem is cold-start.

Actions:

```text
add demographic features,
add popularity-by-age features,
add recent popularity fallback,
use article metadata,
use content features,
segment-specific fallback blending.
```

---

# 11. Use fixed experiment gates

Industry teams usually define promotion gates.

Example:

```yaml
promotion_gate:
  candidate_generator:
    recall_at_500_must_improve_by: 1.0%
    no_drop_in_cold_user_recall: true

  ranker:
    map_at_12_must_improve_by: 0.5%
    no_drop_in_cold_user_map: true
    no_drop_in_heavy_user_map: true

  system:
    inference_latency_not_worse_than: 20%
    candidate_generation_cost_acceptable: true
```

This prevents promoting a model that improves one metric but damages an important segment.

---

# 12. Version everything

Every progressive training stage should produce a versioned artifact.

Track:

```text
data sample version
user buckets
time window
feature version
candidate generator version
ranker version
negative sampling config
training parameters
validation metrics
model artifact path
ANN index version
```

Example:

```yaml
experiment_id: hm_s2_10pct_12w_tt500_lgbm100neg_v3

data:
  user_buckets: 0-9
  train_start: 2020-06-24
  cutoff_date: 2020-09-15
  label_window: 2020-09-16_to_2020-09-22

candidate_generator:
  model_type: two_tower
  init_from: hm_s1_5pct
  embedding_dim: 64
  top_k: 500

ranker:
  model_type: lightgbm_lambdarank
  candidates_from: hm_s2_two_tower
  negatives_per_user: 100
  hard_negative_ratio: 0.60

metrics:
  recall_at_500: 0.42
  map_at_12: 0.027
```

---

# 13. Avoid catastrophic forgetting in two-tower training

If you warm-start the two-tower model on new data, do not train only on new data.

Bad:

```text
Stage 1: train buckets 0-4
Stage 2: continue training only buckets 5-9
```

This can cause the model to adapt too much to the new users.

Better:

```text
Stage 2: continue training on buckets 0-9
```

If full replay is too expensive, use a replay buffer:

```text
new data: 70%
old sampled replay data: 30%
```

Example:

```yaml
two_tower_continual_training_batch_mix:
  new_sample_interactions: 70%
  replay_interactions_from_previous_sample: 30%
```

---

# 14. Early stopping strategy

Use early stopping at every stage.

For two-tower:

```text
monitor Recall@500 on fixed validation users
stop if no improvement for N evaluations
```

For LightGBM:

```text
monitor MAP@12 or NDCG@12 on validation candidates
stop if no improvement after N rounds
```

Do not train for fixed epochs blindly.

---

# 15. Learning-rate strategy

## Two-tower

When continuing from a checkpoint:

```text
new stage learning rate = previous learning rate / 3 to /10
```

Example:

```yaml
stage_1_lr: 0.001
stage_2_lr: 0.0003
stage_3_lr: 0.0001
```

TensorFlow/Keras fine-tuning practice commonly uses freezing/unfreezing and lower learning rates when adapting pretrained weights, especially to avoid destroying useful learned representations. ([TensorFlow][2])

For your two-tower model, you may also freeze article embeddings briefly if the item tower is stable, then unfreeze.

---

# 16. Incremental hard-negative mining

A very useful industry-grade loop is:

```text
1. Train two-tower v1 on small data.
2. Retrieve top-K candidates.
3. Find high-scoring false positives.
4. Add them as hard negatives.
5. Fine-tune two-tower.
6. Regenerate candidates.
7. Train LightGBM.
```

Example:

```text
User U bought article A.

Two-tower v1 retrieves:
A, B, C, D, E

B, C, D, E were not purchased.
They are hard negatives because the model thought they were relevant.
```

These are more valuable than random negatives.

---

# 17. Practical decision table

| Situation                      | Two-tower action                    | LightGBM action                              |
| ------------------------------ | ----------------------------------- | -------------------------------------------- |
| Need quick experiment          | Warm-start from previous checkpoint | Optionally continue, but prefer from scratch |
| Increased user sample only     | Warm-start okay                     | Retrain from scratch                         |
| Changed candidate top-K        | No need to retrain two-tower        | Retrain ranker                               |
| Changed ranker features        | No need to retrain two-tower        | Retrain from scratch                         |
| Changed two-tower architecture | Retrain from scratch                | Retrain after new candidates                 |
| Changed item/user vocab        | Usually retrain from scratch        | Retrain                                      |
| Added new time window          | Warm-start possible with replay     | Retrain ranker                               |
| Final model                    | Compare warm-start vs scratch       | Retrain from scratch                         |

---

# 18. Recommended approach for your project

Use this as your default:

```text
1. Start with 5% deterministic user sample.
2. Train two-tower from scratch.
3. Generate top 200 candidates.
4. Train LightGBM from scratch.
5. Evaluate on fixed temporal validation set.

If performance is not enough:

6. Expand to 10% users.
7. Warm-start two-tower from 5% checkpoint.
8. Train on full 10% data, not just new 5%.
9. Generate top 500 candidates.
10. Retrain LightGBM from scratch.
11. Evaluate on same validation set.

If still not enough:

12. Expand to 25% users.
13. Compare two-tower warm-start vs from-scratch.
14. Use the better candidate model.
15. Retrain LightGBM from scratch.
16. Evaluate MAP@12 and segment metrics.
```

---

# 19. Best practical policy

For your case, I would not use fully incremental training as the main strategy.

I would use:

```text
Progressive data scaling + controlled warm-start for two-tower + full LightGBM retraining.
```

In one sentence:

```text
Warm-start the neural retrieval model when scaling data, but rebuild the LightGBM ranker from freshly generated candidates at each stage.
```

That is the safest industry-grade practice for your two-stage H&M recommender.

[1]: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html?utm_source=chatgpt.com "lightgbm.train — LightGBM 4.6.0.99 documentation"
[2]: https://www.tensorflow.org/guide/keras/transfer_learning?utm_source=chatgpt.com "Transfer learning & fine-tuning - TensorFlow Core"
