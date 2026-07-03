One sample Z-test:
- Used to determine whether the population mean is significantly different from an
assumed value.
- It uses Standard normal distribution as the baseline.
- Assumptions: Either the population's standard deviation should be known or we should
estimate them well when the sample size is not too small (n>30)
- Test statistic = z =
x − μ
0
σ
n

Two sample Z-test:
- Used to compare the means of two populations.
- Assumption: Either the standard deviation (σ ) of the populations should be known

1
, σ
2

or we should estimate them when the sample sizes are not too small (n ).

1
, n
2 ≥ 30

- Test statistic = z =
( x
1− x
2
)−0
σ
1
2
n
1
+
σ
2
2
n
2
One sample Z-proportion test:
- Used to assess if the proportion of a single sample is significantly different from a given
value.
- Assumptions:

- The sample is randomly selected and the sample size is large enough (usually
when n * p and are both greater than 10).

0
n * (1 − p
0
)

- The population can be assumed to be normally distributed or the sample size is
large enough for the Central Limit Theorem to apply.

- Test Statistic = z =
p
^−p
0
(p
0
(1−p
0
)/n
Two-sample Z-proportion test:
- Employed to compare the proportions of two independent samples.
- Assumptions:
- The samples are randomly selected and independent of each other and the
sample sizes are large enough (usually when (n , ,

1
* p
^
1
) (n
1
* (1 − p
^
1
))

(n , and are all greater than 10).
2
* p
^
2
) (n
2
* (1 − p
2
))
^

- The populations can be assumed to be normally distributed or the sample sizes
are large enough for the Central Limit Theorem to apply.

- Test Statistic = z =
p
1
^ −p
2
^
p
^
(1−p
^
) + (
1
n
1
+
1
n
2
)

Note: p is the pooled sample proportion.
^ =
x
1+x
2
n
1 + n
2
One sample t-test:
- The test statistic follows a t-distribution
- It is used when the sample size is too small (n < 30) and/or the population standard
deviation (σ ) is unknown.
- Test statistic = t =
x − μ
0
s
n
- Degree of freedom = n -1

Two sample t-test:
- It is used when the sample sizes are too small (n ) and/or the population

1
, n
2 < 30

standard deviations (σ ) are unknown.

1
, σ
2
- Test statistic = t =
( x
1− x
2
)−0
s
1
2
n
1
+
s
2
2
n
2
- Degree of freedom = n1 + n2 - 2
Paired t-test:

- Used to compare the means of two related groups (e.g., before and after treatment).
- Assumptions:
- The differences between the paired samples are normally distributed.
- The differences are independent of each other.

Chi-square goodness of fit test:
- Used to determine if the distribution of categorical data fits a theoretical distribution
(expected behavior).
- Formula: ChiSquare Statistic =
i
∑
(Oi−E
i
)
2
E
i

- Assumptions:
- Categorical data (data that can be divided into categories).
- Random sample & Independent observations.
- Expected frequencies in each category ≥ 5.

Chi-square test of independence:
- Used to assess whether there is a significant association between two categorical
variables.
- The assumptions for this test are similar to the goodness of fit test.
One way-ANOVA (Analysis of variance):
- Used to determine if there is a statistically significant difference between two or more
categorical groups by testing for differences of means using variance.
- Test Statistic Formula: F statistic = , where:
MSB
MSW

- MSB is the mean square between groups (measures variability between group
means)
- MSW is the mean square within groups (measures variability within each group)
- Assumptions:
- Normality: The data within each group is normally distributed.
- To check normality we perform the Shapiro-Wilk test
- Homogeneity of variances: The variances of the groups are equal.
- To check the homogeneity of variances we perform Levene’s test
- Independence: The observations within each group are independent of each
other.
Kruskal-Wallis test:
- A non-parametric test is used to determine if there are statistically significant differences
between two or more independent groups.

- If One-way ANOVA’s assumption of normality fails, we can perform the Kruskal-Wallis
test.
- Instead of using sample means to compare the groups, it uses sample medians
Two-way ANOVA:
- Used to analyze the influence of two categorical independent variables on a dependent
variable.
- Assumptions:
● The populations from which the samples are drawn should be approximately
normally distributed.
● Homogeneity of variances within each combination of the two independent
variables.
● Independence of observations.

KS (Kolmogorov - Smirnov) test:
- It is a non - parametric test used for determining whether the distributions of two samples
are the same or not.
- The test statistic T follows a distribution called the Kolmogorov Distribution.

ks

TKS = the maximum absolute value of the difference in the CDFs of the two samples X and Y.
- Assumptions:
- The data is continuous.
- The data is independent and identically distributed