# BCaOLSpy

BCaOLSpy is a simple `statsmodels` wrapper that provides a convenient way to compute Bias-Corrected and Accelerated (BCa) confidence intervals for OLS regression estimates. 

## Table of Contents

- [Overview](#overview)
- [References][#references]

## Overview

The BCa (Bias-Corrected and Accelerated) confidence interval is a non-parametric approach that adjusts for both bias and skewness in the data. Traditional OLS confidence intervals assume normally distributed residuals, but BCa intervals are better suited for cases where this assumption does not hold, such as with skewed or heavy-tailed distributions. Additionally, the package also implements Bias-Corrected and percentile intervals, as well as the bias-adjusted OLS point estimate. See `example.ipynb` for an example.


## References

- Efron, B. (1982). The Jackknife, the Bootstrap, and Other Resampling Plans. Society for Industrial and Applied Mathematics.
- Efron, B., & Tibshirani, R. (1987). Better Bootstrap Confidence Intervals. Journal of the American Statistical Association, 82(397), 171-185.
- Hansen, B. E. (2020). Econometrics. Princeton University Press.
- Zivot, E. (2021). [Introduction to Computational Finance and Financial Econometrics with R.](https://bookdown.org/compfinezbook/introcompfinr/) 
