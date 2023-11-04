import pandas as pd
import numpy as np
import scipy.stats as sts
from scipy.stats import ttest_ind # Calculate the T-test for the means of two independent samples of scores.
from scipy.stats import ttest_rel # This is a test for the null hypothesis that two related or repeated samples have identical average (expected) values.
from statsmodels.stats.power import TTestIndPower

"""sumary_line

There are nine people joined a cholestrol reduction diet program.
Cholestrol data are recorded and categorized into first 30 day mean and second 30 day mean.

Compute T test statistic and p value

Compute 95% confidence interval

Compute Cohen's d

Compute statistical power and 0.8-power requried sample number

"""

# only for len(d1) == len(d2)
def compute_cohend(d1, d2):
    assert(len(d1) == len(d2))
    return (np.mean(d1) - np.mean(d2)) / \
        (np.sqrt((np.var(d1, ddof=1) + np.var(d2, ddof=1)) / 2))

# load data
ls_choles = [None for x in range(9)]
ls_choles[0] = [1, 190.0, 180.0]
ls_choles[1] = [2, 183.0, 178.0]
ls_choles[2] = [3, 170.0, 155.0]
ls_choles[3] = [4, 200.0, 195.0]
ls_choles[4] = [5, 178.0, 173.0]
ls_choles[5] = [6, 185.0, 182.0]
ls_choles[6] = [7, 193.0, 189.0]
ls_choles[7] = [8, 165.0, 161.0]
ls_choles[8] = [9, 186.0, 183.0]

df_choles = pd.DataFrame(ls_choles)
df_choles.rename(    
    columns={0: "Participant", 1: "CholesterolFirst30Days", 2: "CholesterolSecond30Days"},
    inplace=True)

# ddof: degree of freedom: n-1
df_choles_first30_mean = df_choles['CholesterolFirst30Days'].mean()
df_choles_first30_std = df_choles['CholesterolFirst30Days'].std(ddof=1)
df_choles_second30_mean = df_choles['CholesterolSecond30Days'].mean()
df_choles_second30_std = df_choles['CholesterolSecond30Days'].std(ddof=1)

# perform relative (difference) two sample t-test
# greater: the mean of the distribution underlying the first sample is greater than that of the second sample.
relative_t_test_results = ttest_rel(df_choles['CholesterolFirst30Days'], df_choles['CholesterolSecond30Days'], 
                                    alternative="greater")
paired_t_test_choles_stats = relative_t_test_results.statistic
paired_t_test_choles_pvalue = relative_t_test_results.pvalue

df_choles_gap = df_choles['CholesterolFirst30Days'] - df_choles['CholesterolSecond30Days']

df_choles_gap_mean = df_choles_gap.mean()

# confidence interval
ci_choles = sts.t.interval(confidence=0.95, 
              df=df_choles_gap.size-1,
              loc=np.mean(df_choles_gap),
              scale=sts.sem(df_choles_gap)
              )

cohens_d_choles = compute_cohend(df_choles['CholesterolFirst30Days'], df_choles['CholesterolSecond30Days'])
cohens_d_choles_fromGPower = 0.5046206

# perform power analysis to find sample size
# for given effect 
alpha_significance = 0.05
analysis = TTestIndPower()

stats_choles_power = analysis.solve_power(effect_size=cohens_d_choles, 
                    alpha=alpha_significance, 
                    power=None,
                    nobs1=df_choles_gap.size,
                    ratio=1.0, 
                    alternative='larger') 
stats_choles_power_fromGPower = 0.2677945 # from G*Power

n_choles_samples = analysis.solve_power(effect_size=cohens_d_choles, 
                    alpha=alpha_significance, 
                    power=0.8,
                    nobs1=None,
                    ratio=1.0, 
                    alternative='larger') 
n_choles_samples_fromGPower = 100 / 2# from G*Power