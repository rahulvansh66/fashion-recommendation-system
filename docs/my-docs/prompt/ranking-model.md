**Popularity correction** in `recsys-v2` fixes a bias from **in-batch negatives**.

### The problem
During training, negatives are only the **other items in the same batch**. Popular articles show up in batches more often, so the model sees them as negatives more often and learns to **push them down too aggressively**, even when they are often the right answer.

### What v2 does
1. Before training, compute each article’s **popularity**:  
   `P(article) = (# times bought in train) / (total train transactions)`
2. In the training loss, for the true item on each row, **subtract `log(P(article))` from its logit/score**.

So frequent items get a small score boost before softmax, canceling out the fact that they appear more often as accidental negatives in batches.

### In one line
It **rebalances the loss so popular items are not unfairly penalized** just because they appear more often in random batches.

**Note:** This is applied **only during training**. At eval time, v2 scores against the **full catalog** with no popularity correction.