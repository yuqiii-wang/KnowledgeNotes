import pandas as pd
import numpy as np

"""
    Food scientist proposes a healthy gum (memory gum) that can temporarily boost memory and cognitive skill.

    Experiment includes three groups of people for comparison: Memory Gum, Regular Gum, No Gum.
    People are tested in a memorization-based exam immediately after having consumed the gum, and scores are recorded.

    Find ANOVA and check f and p value to see if the gum is effective in boosting memory.
"""

extract_df = pd.read_csv("gum.csv")

group_ls = np.unique(extract_df["Group"])

extract_df_group = extract_df.groupby(by="Group")
extract_df_group_mean = extract_df_group.mean().reset_index()

extract_df["InGroupVar"] = np.nan
extract_df["BetweenGroupVar"] = np.nan

for group in group_ls:
    group_diff = extract_df[extract_df["Group"] == group]["Scores"] - \
        extract_df_group_mean[extract_df_group_mean["Group"] == group]["Scores"].values[0]
    group_var = np.square(group_diff)
    extract_df.loc[list(group_diff.index),'InGroupVar'] = list(group_var.values)

    group_var = np.square(extract_df_group_mean[extract_df_group_mean["Group"] == group]["Scores"].values[0] \
              - extract_df["Scores"].mean())
    extract_df.loc[list(group_diff.index),'BetweenGroupVar'] = group_var
    
    del group_diff
    del group_var
    
ssr = np.sum(extract_df["BetweenGroupVar"])
sse = np.sum((extract_df["InGroupVar"]))
msr = ssr / (len(group_ls)-1)
mse = sse / (len(extract_df)-len(group_ls))
f_val = msr / mse
f_report_str = """
ssr: {ssr},     sse: {sse}
msr: {msr},     mse: {mse}
f_val: {f_val}
"""
print(f_report_str.format(
    ssr=ssr, sse=sse,
    msr=msr, mse=mse,
    f_val=f_val))

# get ANOVA table as R like output
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Ordinary Least Squares (OLS) model
model = ols('Scores ~ C(Group)', data=extract_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

"""
	        sum_sq	            df	    F	                PR(>F)
C(Group)	653.7283950617322	2.0	    2.4406169118950682	0.09373405536974168
Residual	10446.296296296296	78.0		
"""
print("For PR(>F) = 0.0937 > 0.05, H_1 is rejected.")


########## tukey_hsd

from scipy.stats import tukey_hsd

extract_df_col_by_group = pd.DataFrame()

for group in group_ls:
    extract_df_col_by_group[group] = extract_df[extract_df["Group"] == group]["Scores"].reset_index(drop="index")

# confidence_interval(confidence_level=0.95):
tukey_hsd_table = tukey_hsd(extract_df_col_by_group['Memory Gum'], 
                extract_df_col_by_group['No Gum'], 
                extract_df_col_by_group['Regular Gum'])
print("\n")
print(tukey_hsd_table)

"""
Tukey's HSD Pairwise Group Comparisons (95.0% Confidence Interval)
Comparison  Statistic  p-value  Lower CI  Upper CI
 (0 - 1)      6.889     0.080    -0.637    14.414
 (0 - 2)      4.296     0.365    -3.229    11.822
 (1 - 0)     -6.889     0.080   -14.414     0.637
 (1 - 2)     -2.593     0.690   -10.118     4.933
 (2 - 0)     -4.296     0.365   -11.822     3.229
 (2 - 1)      2.593     0.690    -4.933    10.118
"""