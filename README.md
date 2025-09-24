# MixLS: Mixed-Effects Location Scale Models

<p align="center">
  <img src="logo.svg" alt="MixLS logo" width="200">
</p>

[![R](https://img.shields.io/badge/R-%3E%3D4.0.0-blue.svg)](https://www.r-project.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

**MixLS** provides tools for fitting mixed-effects location scale models to longitudinal and clustered data. These models extend traditional mixed-effects models by allowing covariates to influence both the mean structure and the variance components (both within-subject and between-subject variances).

### Key Features

- **Three-stage estimation** for robust parameter estimation
- **Flexible variance modeling** for both between- and within-subject effects  
- **Multiple association types** between location and scale effects
- **Adaptive quadrature** for improved numerical integration
- **Ridge regularization** for stable parameter estimation and handling convergence issues
- **Comprehensive diagnostics** including empirical Bayes estimates

### Applications

This approach is particularly useful for:

- Intensive longitudinal data (e.g., ecological momentary assessment)
- Understanding heteroscedasticity in repeated measures
- Modeling both location (mean) and scale (variance) effects
- Identifying subjects with different patterns of variability
- Personalized interventions and understanding heterogeneity in longitudinal processes

## Installation

### Install from GitHub

You can install the development version of MixLS from GitHub using `devtools`:

```r
# Install devtools if you haven't already
if (!require("devtools")) {
  install.packages("devtools")
}

# Install MixLS from GitHub
devtools::install_github("Jiananrui/MixLS", build_vignettes = TRUE, dependencies = TRUE)
```

### Dependencies

MixLS depends on the following R packages, which will be automatically installed:

#### Required Dependencies
```r
# Core dependencies (automatically installed)
install.packages(c(
  "dplyr",           # For data manipulation
  "statmod",         # For statistical modeling functions
  "Rcpp",            # For C++ integration
  "RcppArmadillo"    # For linear algebra operations
))
```

#### Suggested Dependencies (for examples and vignettes)
```r
# Optional packages for enhanced functionality
install.packages(c(
  "ggplot2",   # For data visualization
  "dplyr",     # For data manipulation
  "knitr",     # For vignettes
  "rmarkdown"  # For vignettes
))
```

### System Requirements

- **R version**: ≥ 4.0.0
- **Operating Systems**: Windows, macOS, Linux
- **Memory**: Recommended ≥ 4GB RAM for large datasets

## Quick Start

```r
library(MixLS)

# Load example dataset
data("riesby", package = "MixLS")

# Fit a mixed-effects location scale model
model <- mixregls(
  data = riesby,
  mean_formula = ~ week + endog + endweek,
  var_bs_formula = ~ endog,
  var_ws_formula = ~ week + endog,
  id_var = "id",
  response_var = "hamdep",
  nq = 11,
  maxiter = 200,
  tol = 1e-5,
  stage = 3,
  stage3_model = "linear",
  adaptive = 1,
  ridge_stage1 = 0,
  ridge_stage2 = 0.1,
  ridge_stage3 = 0.2
)

# View results
summary(model)
```
## Model Overview

### Original Mixed-Effects Location Scale Model

The mixed-effects location scale model begins with:

$$y_{ij} = x_{ij}^T\beta + v_i + \epsilon_{ij}$$

where:
- **Mean model**: $x_{ij}^T\boldsymbol{\beta}$ represents fixed effects
- **Between-subject variance**: $\sigma^2_{v_{ij}} = \exp(\mathbf{u}_{ij}^T\boldsymbol{\alpha})$
- **Within-subject variance**: $\sigma^2_{\epsilon_{ij}} = \exp(w_{ij}^T\boldsymbol{\tau} + \omega_i)$
- $v_i \sim N(0, \sigma^2_{v_{ij}})$ are random location effects
- $\omega_i \sim N(0, \sigma^2_\omega)$ are random scale effects

### Standardization to Computational Form

For computational efficiency and model interpretability, the original random effects are standardized using Cholesky factorization:

```math
\begin{bmatrix} v_i \\ \omega_i \end{bmatrix} = \begin{bmatrix} s_{1ij} & 0 \\ s_{2ij} & s_{3ij} \end{bmatrix} \begin{bmatrix} \theta_{1i} \\ \theta_{2i} \end{bmatrix} = \begin{bmatrix} \sigma_{v_{ij}} & 0 \\ \frac{\sigma_{v\omega}}{\sigma_{v_{ij}}} & \sqrt{\sigma_{\omega}^2 - \frac{\sigma_{v\omega}^2}{\sigma_{v_{ij}}^2}}\end{bmatrix} \begin{bmatrix} \theta_{1i} \\ \theta_{2i} \end{bmatrix}
```

where $\theta_{1i}, \theta_{2i} \sim N(0,1)$ are standardized normal random effects.

This standardization transforms the model into the computational form used by our package:

$$y_{ij} = x_{ij}^T\beta + \sigma_{vi}\theta_{1i} + \epsilon_{ij}$$

where the within-subject variance becomes:

$$\sigma^2_{\epsilon_{ij}} = \exp(w_{ij}^T\tau + \text{association terms})$$

### Location-Scale Association Types

The package supports four different association structures between location and scale effects:

1. **Independent**: No association between location and scale effects
   - Within-subject variance: $\sigma^2_{\epsilon} = \exp(w_{ij}^T\tau + \sigma_\omega\theta_{2i})$

2. **Linear association**: $\tau_l\theta_{1i}$
   - Within-subject variance: $\sigma^2_{\epsilon} = \exp(w_{ij}^T\tau + \tau_l\theta_{1i} + \sigma_\omega\theta_{2i})$
   - Direct proportional relationship between random location and scale effects

3. **Quadratic association**: $\tau_q\theta_{1i}^2$
   - Within-subject variance: $\sigma^2_{\epsilon} = \exp(w_{ij}^T\tau + \tau_l\theta_{1i} + \tau_q\theta_{1i}^2 + \sigma_\omega\theta_{2i})$
   - Non-linear quadratic relationship between location and scale effects

4. **Interaction**: $\tau_k\theta_{1i}\theta_{2i}$
   - Within-subject variance: $\sigma^2_{\epsilon} = \exp(w_{ij}^T\tau + + \tau_l\theta_{1i} + \tau_k\theta_{1i}\theta_{2i} + \sigma_\omega\theta_{2i})$
   - Captures interaction between location and scale random effects

The model is estimated in three sequential stages:

1. **Stage 1**: Mean model + between-subject variance effects
2. **Stage 2**: Adds within-subject variance effects  
3. **Stage 3**: Adds random scale effects and location-scale association

## Example Datasets

The package includes two example datasets:

### 1. Riesby Depression Study (`riesby`)
Hamilton Depression Rating Scale scores for 66 patients measured weekly over 6 weeks.

```r
data("riesby", package = "MixLS")
head(riesby)
```

Variables:
- `id`: Patient identifier
- `week`: Week of assessment (0-5)
- `hamdep`: Depression Score
- `endog`: Endogenous depression indicator (1 = Endogenous, 0 = Non-Endogenous)
- `endweek`: Interaction term (endog × week)

### 2. Positive Mood Study (`posmood`)
Ecological momentary assessment data from adolescents with multiple daily mood measurements.

```r
data("posmood", package = "MixLS")
head(posmood)
```

Variables:
- `id`: Participant identifier
- `posmood`: Positive mood rating
- `alone`: Social context indicator (1 = with others, 0 = alone)
- `genderf`: Gender (1 = female, 0 = male)

## Key Functions

### Main Modeling Function

- `mixregls()`: Fit mixed-effects location scale models

### Model Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data` | Data frame containing variables | Required |
| `mean_formula` | Formula for mean structure | Required |
| `var_bs_formula` | Formula for between-subject variance | Required |
| `var_ws_formula` | Formula for within-subject variance | Required |
| `id_var` | Column name for subject ID | Required |
| `response_var` | Column name for outcome variable | Required |
| `nq` | Number of quadrature points | 11 |
| `maxiter` | Maximum iterations | 200 |
| `tol` | Convergence tolerance | 1e-5 |
| `stage` | Modeling stage (1, 2, or 3) | 3 |
| `adaptive` | Use adaptive quadrature (0/1) | 1 |
| `ridge_stage1` | Ridge parameter for stage 1 | 0 |
| `ridge_stage2` | Ridge parameter for stage 2 | 0.1 |
| `ridge_stage3` | Ridge parameter for stage 3 | 0.2 |

## Documentation

### Getting Help

```r
# Package overview
help(package = "MixLS")

# Function documentation
?mixregls

# View vignettes
browseVignettes("MixLS")
```

### Vignettes

The package includes comprehensive vignettes:

```r
# Introduction and examples
vignette("MIXLS-introduction", package = "MixLS")
```

## Troubleshooting

### Common Issues

1. **Convergence Problems**
   - Increase `maxiter`
   - Adjust ridge parameters
   - Try different `nq` values

2. **Memory Issues with Large Datasets**
   - Reduce `nq` 
   - Use ridge regularization
   - Consider data subsampling

3. **Installation Issues**
   ```r
   # Update R and packages
   update.packages()
   
   # Install dependencies manually
   install.packages(c("dplyr", "statmod", "Rcpp", "RcppArmadillo"))
   
   # For RcppArmadillo compilation issues on some systems:
   # Make sure you have appropriate C++ compiler
   # On Windows: Install Rtools
   # On macOS: Install Xcode command line tools
   # On Linux: Install build-essential
   ```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development

```r
# Clone repository
git clone https://github.com/Jiananrui/MixLS.git

# Install development version
devtools::install_local("path/to/MixLS")
```

## Citation

If you use MixLS in your research, please cite:

```bibtex
@Manual{mixls,
  title = {MixLS: Mixed-Effects Location Scale Models},
  author = {Jianan Rui},
  year = {2025},
  note = {R package},
  url = {https://github.com/Jiananrui/MixLS}
}
```

## References

Hedeker, D., Mermelstein, R. J., & Demirtas, H. (2008). An application of a mixed-effects location scale model for analysis of ecological momentary assessment (EMA) data. *Biometrics*, 64(2), 627-634.

Hedeker, D., & Nordgren, R. (2013). MIXREGLS: A program for mixed-effects location scale analysis. *Journal of statistical software*, 52, 1-38.

Rabe-Hesketh, S., Skrondal, A., & Pickles, A. (2002). Reliable estimation of generalized linear mixed models using adaptive quadrature. *The Stata Journal*, 2(1), 1-21.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Jianan Rui
- **Email**: jiananrui@uchicago.edu
- **GitHub**: [Jiananrui](https://github.com/Jiananrui)
- **Issues**: [GitHub Issues](https://github.com/Jiananrui/MixLS/issues)

---

**Note**: This package is under active development. Please report any bugs or feature requests via GitHub Issues.
