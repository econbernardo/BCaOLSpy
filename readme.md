# BCaOLSpy

BCaOLSpy is a simple `statsmodels` wrapper that provides a convenient way to compute Bias-Corrected and Accelerated (BCa) confidence intervals for OLS regression estimates. 

## Table of Contents

- [Overview](#overview)
- [Contributing](#contributing)
- [References](#references)
  

## Overview

The BCa (Bias-Corrected and Accelerated) confidence interval is a non-parametric approach that adjusts for both bias and skewness in the data. Traditional OLS confidence intervals assume normally distributed residuals, but BCa intervals are better suited for cases where this assumption does not hold, such as with skewed or heavy-tailed distributions. Additionally, the package also implements Bias-Corrected and percentile intervals, as well as the bias-adjusted OLS point estimate. See `example.ipynb` for an example.

## Contributing

We welcome contributions to BCaOLSpy! If you have ideas, suggestions, or would like to help with development, please follow these steps:

### Reporting Issues
If you encounter any issues, bugs, or have feature requests, please create an issue in the [Issues](https://github.com/econbernardo/BCaOLSpy/issues) section of the GitHub repository.

### Pull Requests
We encourage you to contribute code to the project by following these steps:
1. **Fork the Repository**: Click the "Fork" button at the top of this page to create your own copy of the repository.
2. **Clone Your Fork**: 
    ```bash
    git clone https://github.com/econbernardo/BCaOLSpy.git
    ```
3. **Create a Branch**: Create a new branch for your feature or bugfix.
    ```bash
    git checkout -b feature-name
    ```
4. **Make Changes**: Make your changes or add your new feature.
5. **Test Your Changes**: Ensure your changes work as expected.
6. **Commit and Push**: Commit your changes and push them to your forked repository.
    ```bash
    git commit -m "Add feature name"
    git push origin feature-name
    ```
7. **Submit a Pull Request**: Go to the original repository on GitHub and click the "New Pull Request" button. Please provide a clear description of your changes and reference any related issues.

### Code Style and Guidelines
- Please ensure your code adheres to PEP 8 standards.
- Document your code where applicable to improve readability and maintainability.
- Make sure all tests pass before submitting a pull request.

Thank you for helping improve BCaOLSpy!


## References

- Efron, B. (1982). The Jackknife, the Bootstrap, and Other Resampling Plans. Society for Industrial and Applied Mathematics.
- Efron, B., & Tibshirani, R. (1987). Better Bootstrap Confidence Intervals. Journal of the American Statistical Association, 82(397), 171-185.
- Hansen, B. E. (2020). Econometrics. Princeton University Press.
- Zivot, E. (2021). [Introduction to Computational Finance and Financial Econometrics with R.](https://bookdown.org/compfinezbook/introcompfinr/)
