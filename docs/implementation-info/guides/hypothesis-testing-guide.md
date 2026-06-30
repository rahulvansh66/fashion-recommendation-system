
# Numeric vs. Target (Hypothesis Testing)

Use to check if feature means differ significantly across target classes

The term **assertions** (more commonly called **assumptions**) refers to the conditions that should be reasonably satisfied before applying a statistical test.

### Important Considerations Before Testing

Before applying any hypothesis test, you must always **check if its assumptions are satisfied**. 

If the assumptions are **not satisfied**, you should:
1. **Transform the Data:** Check if you can use data transformation techniques (e.g., Log transformation, Square Root, or Box-Cox transformation) such that the transformed data satisfies the assumptions of the hypothesis test.
2. **Use Alternative Tests:** If transformations do not work or are inappropriate for your use case, proceed with the alternative non-parametric tests listed for each method below.

# t-tests

## Assumptions of the t-Test

A t-test is used to compare means (one-sample, independent-samples, or paired-samples).

#### Common Assumptions

1. **Dependent variable is continuous**

   * Measured on an interval or ratio scale (e.g., height, weight, exam score).

2. **Random sampling**

   * Observations should be randomly selected.

3. **Independence of observations**

   * One participant's measurement should not influence another's.

4. **Normality**

   * The dependent variable should be approximately normally distributed within each group.
   * Especially important for small sample sizes.
   * Commonly tested using Shapiro-Wilk Test for small to medium datasets (typically n < 2000) or Kolmogorov-Smirnov (KS) Test for larger datasets, though it has lower statistical power compared to Shapiro-Wilk Test.

5. **Homogeneity of variance** (Independent t-test only)

   * The variances of the two groups should be approximately equal.
   * Often tested using Levene's Test.

### What if Assumptions Fail? (Alternative Tests)

*   **If Normality fails:** Use the **Mann-Whitney U Test** (also known as the Wilcoxon rank-sum test), which is a non-parametric alternative.
*   **If Homogeneity of Variance fails:** Use **Welch's t-test**, which is designed for unequal variances.

# Paired t-test

#### Assumption for Paired t-test

* Same as t-test, Additional, the **differences between paired observations** should be normally distributed.

### What if Assumptions Fail? (Alternative Tests)

*   **If Normality of differences fails:** Use the **Wilcoxon Signed-Rank Test**, a non-parametric alternative.

# ANOVA

### Assumptions of ANOVA (Analysis of Variance)

ANOVA compares means across three or more groups.

#### Assumptions

1. **Dependent variable is continuous**

   * Interval or ratio scale.

2. **Independent variable is categorical**

   * Groups/categories being compared.

3. **Random sampling**

4. **Independence of observations**

   * Each observation belongs to only one group.

5. **Normality**

   * The dependent variable is approximately normally distributed within each group.
   * Commonly tested using Shapiro-Wilk Test for small to medium datasets (typically n < 2000) or Kolmogorov-Smirnov (KS) Test for larger datasets, though it has lower statistical power compared to Shapiro-Wilk Test.

6. **Homogeneity of variance**

   * Variances across groups are equal.
   * Commonly tested using Levene's Test or Bartlett's Test.

### Example

Comparing average exam scores among students taught by three different teaching methods.

### What if Assumptions Fail? (Alternative Tests)

*   **If Normality fails:** Use the **Kruskal-Wallis H Test**, a non-parametric alternative to one-way ANOVA.
*   **If Homogeneity of Variance fails:** Use **Welch's ANOVA**.

# Categorical vs. Target:

Use Chi-Square Test of Independence


## 3. Assumptions of the Chi-Square Test

Chi-square tests are used for categorical data.

### Assumptions

1. **Data are categorical**

   * Frequencies or counts, not means.

2. **Random sampling**

3. **Independence of observations**

   * Each subject contributes to only one cell of the contingency table.

4. **Expected cell frequencies are sufficiently large**

   * General rule:

     * Expected frequency in each cell ≥ 5.
     * Alternatively, no more than 20% of cells should have expected counts less than 5, and no cell should have expected count less than 1.

### Example

Testing whether gender is associated with preference for a product.

| Gender | Like | Dislike |
| ------ | ---- | ------- |
| Male   | 40   | 20      |
| Female | 35   | 25      |

A Chi-square test determines whether preference depends on gender.

### What if Assumptions Fail? (Alternative Tests)

*   **If Expected Cell Frequencies are < 5:** Use **Fisher's Exact Test**. It is exact and does not rely on large-sample approximations (most commonly used for 2x2 contingency tables, though it can be extended).

---



# Quick Comparison

| Assumption                  | t-Test                 | ANOVA | Chi-Square |
| --------------------------- | ---------------------- | ----- | ---------- |
| Random sampling             | ✓                      | ✓     | ✓          |
| Independence                | ✓                      | ✓     | ✓          |
| Normal distribution         | ✓                      | ✓     | ✗          |
| Equal variances             | ✓ (independent t-test) | ✓     | ✗          |
| Continuous outcome variable | ✓                      | ✓     | ✗          |
| Categorical outcome/counts  | ✗                      | ✗     | ✓          |
| Expected frequencies ≥ 5    | ✗                      | ✗     | ✓          |


# Easy way to remember

* **t-test** → Compare **2 means** → needs **normality + equal variance**. *(Fails? → Mann-Whitney U or Welch's)*
* **Paired t-test** → Compare **paired means** → needs **normal differences**. *(Fails? → Wilcoxon Signed-Rank)*
* **ANOVA** → Compare **3 or more means** → same assumptions as t-test. *(Fails? → Kruskal-Wallis or Welch's ANOVA)*
* **Chi-square** → Compare **frequencies/proportions** → needs **categorical data and adequate expected counts**. *(Fails? → Fisher's Exact Test)*