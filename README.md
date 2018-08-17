# FragileFamiliesChallenge
Mass collaboration and open research effort using predictive modeling, causal inference, and in-depth interviews to generate insights that can improve the lives of disadvantaged children in the United States. Project for COS 424.

Abstract
----------------------------------------------
As a part of the Fragile Families Challenge, this project allowed us to use multiple machine learning techniques to explore how best to use a given set of data to predict both continuous and binary outcomes. For continuous outcomes, we used regularized linear regressions (Ridge/Lasso), and for binary we used the logistic regression. We tested several supported theories in the social science field regarding the correlation between certain features and our 6 target outcomes (grit, GPA, material hardship, eviction, job loss, and job training). We looked at these subsets to see which theories the Fragile Families Challenge best supports and provides us with the most accurate predictions.

Deployment
----------------------------------------------
This directory comprises the following files

-MissingDataScript.py: performs missing data imputation

-regress.py: runs OLS with ridge and lasso regularization for predicting continuous variables (GPA, grit, material hardship) and logistic regression for classifying binary vairables (eviction, layoff, job training).

Authors
----------------------------------------------
- Peter Chen
- Brandon Lanchang
