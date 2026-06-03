For example lets take a tiny two-tower model with only **2-dimensional embeddings** so it is easy to see.

In real systems, embeddings may be 64, 128, 256, or 768 dimensions, and weights are much larger.

## Example user and item

```text
User:
  user_id = U101
  age_group = 30s
  plan_type = premium
  interest = security

Item:
  item_id = I555
  category = VPN
  brand = Norton
  price_bucket = medium
```

A two-tower model has:

```text
User tower weights
Item tower weights
```

These weights are learned during training.

---

## 1. User tower example

Let’s say the user has a simple feature vector:

```text
user_features = [1.0, 0.5, 0.2]
```

Meaning, for example:

```text
1.0 = premium user
0.5 = medium activity
0.2 = some security interest score
```

The **user tower weights** might look like this:

```text
User tower weights W_user =

[
  [0.8,  0.1],
  [0.4,  0.6],
  [0.2,  0.9]
]
```

These are not the user embedding yet. These are the learned numbers inside the model.

Now the model multiplies:

```text
user_features × W_user = user_embedding
```

Calculation:

```text
[1.0, 0.5, 0.2] × [
  [0.8, 0.1],
  [0.4, 0.6],
  [0.2, 0.9]
]
```

Result:

```text
user_embedding = [1.04, 0.58]
```

So:

```text
User tower weights = the learned model parameters
User embedding     = [1.04, 0.58]
```

---

## 2. Item tower example

Now the item has a feature vector:

```text
item_features = [0.7, 1.0, 0.3]
```

Meaning, for example:

```text
0.7 = VPN category signal
1.0 = Norton brand signal
0.3 = medium price signal
```

The **item tower weights** might look like this:

```text
Item tower weights W_item =

[
  [0.9,  0.2],
  [0.3,  0.8],
  [0.5,  0.4]
]
```

Again, these are learned model parameters, not the item embedding.

Now the model multiplies:

```text
item_features × W_item = item_embedding
```

Calculation:

```text
[0.7, 1.0, 0.3] × [
  [0.9, 0.2],
  [0.3, 0.8],
  [0.5, 0.4]
]
```

Result:

```text
item_embedding = [1.08, 1.06]
```

So:

```text
Item tower weights = the learned model parameters
Item embedding     = [1.08, 1.06]
```

---

## What FAISS stores

FAISS stores the **item embedding**, not the item tower weights.

For this item:

```text
item_id = I555
item_embedding = [1.08, 1.06]
```

FAISS index may contain many items like:

```text
I111 -> [0.20, 0.90]
I222 -> [1.40, 0.10]
I555 -> [1.08, 1.06]
I777 -> [0.95, 0.80]
```

When user `U101` comes in, the user tower creates:

```text
user_embedding = [1.04, 0.58]
```

Then FAISS searches for item embeddings closest to:

```text
[1.04, 0.58]
```

---

## Important point

The model does **not** have separate weights for every user and every item in this simplified feature-based example.

Instead:

```text
Same user tower weights
  are used to generate embeddings for many users

Same item tower weights
  are used to generate embeddings for many items
```

For example:

```text
User U101 features -> same user tower weights -> U101 embedding
User U202 features -> same user tower weights -> U202 embedding
User U303 features -> same user tower weights -> U303 embedding
```

And:

```text
Item I555 features -> same item tower weights -> I555 embedding
Item I777 features -> same item tower weights -> I777 embedding
Item I999 features -> same item tower weights -> I999 embedding
```

So the clean distinction is:

```text
Weights:
  stored inside the trained model

Embeddings:
  produced by the trained model for a specific user or item
```

In this example:

```text
User tower weights:
[
  [0.8, 0.1],
  [0.4, 0.6],
  [0.2, 0.9]
]

User U101 embedding:
[1.04, 0.58]

Item tower weights:
[
  [0.9, 0.2],
  [0.3, 0.8],
  [0.5, 0.4]
]

Item I555 embedding:
[1.08, 1.06]
```
