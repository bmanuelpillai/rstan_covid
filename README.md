# Estimating COVID-19 Epidemiological Parameters using R and Stan / Python and NumPyro

This project fits a Bayesian SIR (Susceptible-Infected-Recovered) model to
county-level COVID-19 case data from the United States using either:

- [RStan](https://mc-stan.org/rstan/) / [CmdStanR](https://mc-stan.org/cmdstanr/) and [Quarto](https://quarto.org/)
- [NumPyro](https://num.pyro.ai/) with JAX in Python

## Repository Structure

```
01_data/            Raw data files
02_cleaning/        Data cleaning and preparation (clean_data.qmd)
03_modeling/        Stan and NumPyro model workflows
```

## Workflow

1. **Data cleaning** – Run `02_cleaning/clean_data.qmd` to load the raw CSV
   files, join case counts with census population sizes, and compute daily
   incidence per county.
2. **Model fitting** – Run `03_modeling/fit.qmd` to compile the Stan SIR model,
   prepare the data for a chosen county, run MCMC sampling, and inspect
   posterior diagnostics and visualizations.
3. **NumPyro model fitting** – Install Python dependencies from
   `03_modeling/requirements-numpyro.txt`, then run
   `python 03_modeling/fit_numpyro.py --county "Young, Texas"` to fit an
   equivalent Bayesian SIR model in NumPyro and print posterior + posterior
   predictive summaries.

## Data Sources

- **COVID-19 case data**: USA county-wise case counts from Kaggle  
  <https://www.kaggle.com/datasets/imdevskp/corona-virus-report/data?select=usa_county_wise.csv>

- **US Census demographic data**: County population sizes from Kaggle  
  <https://www.kaggle.com/datasets/muonneutrino/us-census-demographic-data>

## Dependencies

R package dependencies are managed with [renv](https://rstudio.github.io/renv/).
Restore the project library with:

```r
renv::restore()
```

Key R packages used: `tidyverse`, `cmdstanr`, `bayesplot`, `posterior`.

For NumPyro:

```bash
python -m pip install -r 03_modeling/requirements-numpyro.txt
```
