#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive


@dataclass
class CountySeries:
    county: str
    dates: np.ndarray
    incidence: np.ndarray
    population: float


def normalize_area(name: str) -> str:
    return name.replace(", US", "").replace(" County", "").strip()


def load_population(pop_path: Path) -> dict[str, int]:
    populations: dict[str, int] = {}
    with pop_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for row in reader:
            area = normalize_area(row["geographic_area"].strip().strip('"'))
            pop = int(row["population_size"].strip().strip('"').replace(",", ""))
            populations[area] = pop
    return populations


def load_county_series(cases_path: Path, county: str, population: int) -> CountySeries:
    grouped: dict[str, list[tuple[dt.date, int]]] = defaultdict(list)
    with cases_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            admin2 = (row.get("Admin2") or "").strip()
            if not admin2:
                continue
            area = normalize_area((row.get("Combined_Key") or "").strip().strip('"'))
            if not area:
                continue
            date = dt.datetime.strptime(row["Date"], "%m/%d/%y").date()
            confirmed = int(float(row["Confirmed"]))
            grouped[area].append((date, confirmed))

    if county not in grouped:
        available = ", ".join(sorted(list(grouped.keys()))[:15])
        raise ValueError(f"County '{county}' not found. Example available counties: {available}")

    rows = sorted(grouped[county], key=lambda x: x[0])
    dates = np.array([d for d, _ in rows], dtype="datetime64[D]")
    confirmed = np.array([c for _, c in rows], dtype=np.int64)
    incidence = np.diff(np.concatenate(([0], confirmed)))
    incidence = np.maximum(incidence, 0)

    return CountySeries(
        county=county,
        dates=dates,
        incidence=incidence,
        population=float(population),
    )


def sir_incidence(beta: jnp.ndarray, gamma: jnp.ndarray, s0_frac: float, i0_frac: float, population: float, T: int) -> jnp.ndarray:
    def step(carry, _):
        s, i, r = carry
        new_inf = jnp.clip(beta * s * i, a_min=1e-8)
        new_rec = jnp.clip(gamma * i, a_min=1e-8)

        s_next = jnp.clip(s - new_inf, a_min=1e-8, a_max=1.0)
        i_next = jnp.clip(i + new_inf - new_rec, a_min=1e-8, a_max=1.0)
        r_next = jnp.clip(r + new_rec, a_min=1e-8, a_max=1.0)

        return (s_next, i_next, r_next), jnp.maximum(new_inf * population, 1e-6)

    y0 = (jnp.array(s0_frac), jnp.array(i0_frac), jnp.array(1.0 - s0_frac - i0_frac))
    _, incidence = jax.lax.scan(step, y0, xs=None, length=T)
    return incidence


def model(cases: jnp.ndarray, population: float, s0_frac: float, i0_frac: float):
    T = cases.shape[0]

    beta = numpyro.sample("beta", dist.LogNormal(jnp.log(0.4), 0.5))
    gamma = numpyro.sample("gamma", dist.LogNormal(jnp.log(0.14), 0.2))
    rho = numpyro.sample("rho", dist.Beta(2.0, 5.0))
    phi = numpyro.sample("phi", dist.Exponential(1.0))

    incidence = sir_incidence(beta, gamma, s0_frac, i0_frac, population, T)
    mu = jnp.maximum(rho * incidence, 1e-6)

    numpyro.sample("cases", dist.NegativeBinomial2(mu=mu, concentration=phi), obs=cases)
    numpyro.sample("y_rep", dist.NegativeBinomial2(mu=mu, concentration=phi))
    numpyro.deterministic("R0", beta / gamma)
    numpyro.deterministic("incidence", incidence)


def run_inference(series: CountySeries, warmup: int, samples: int, chains: int, seed: int):
    i0 = 1.0
    s0 = max(series.population - i0, 1.0)
    s0_frac = s0 / series.population
    i0_frac = i0 / series.population

    cases = jnp.asarray(series.incidence.astype(np.int32))

    kernel = NUTS(model, target_accept_prob=0.95)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(seed), cases=cases, population=series.population, s0_frac=s0_frac, i0_frac=i0_frac)

    posterior_samples = mcmc.get_samples()
    predictive = Predictive(model, posterior_samples=posterior_samples, return_sites=["y_rep", "R0"]) 
    ppc = predictive(
        jax.random.PRNGKey(seed + 1),
        cases=cases,
        population=series.population,
        s0_frac=s0_frac,
        i0_frac=i0_frac,
    )

    return mcmc, ppc


def print_summary(mcmc: MCMC, ppc: dict[str, jnp.ndarray], observed: np.ndarray):
    mcmc.print_summary(exclude_deterministic=False)

    y_rep = np.asarray(ppc["y_rep"])
    pred_median = np.median(y_rep, axis=0)
    pred_low = np.percentile(y_rep, 5, axis=0)
    pred_high = np.percentile(y_rep, 95, axis=0)

    print("\nPosterior predictive check (first 14 days):")
    print("day\tobs\tmedian\tp05\tp95")
    for i in range(min(14, observed.shape[0])):
        print(f"{i+1}\t{int(observed[i])}\t{pred_median[i]:.1f}\t{pred_low[i]:.1f}\t{pred_high[i]:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Fit a Bayesian SIR model with NumPyro")
    parser.add_argument("--county", default="Young, Texas", help="County in '<County>, <State>' form")
    parser.add_argument("--cases-path", default="01_data/usa_county_wise.csv")
    parser.add_argument("--population-path", default="01_data/population_sizes.csv")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cases_path = root / args.cases_path
    pop_path = root / args.population_path

    populations = load_population(pop_path)
    if args.county not in populations:
        sample = ", ".join(sorted(list(populations.keys()))[:15])
        raise ValueError(f"County '{args.county}' not found in population file. Example counties: {sample}")

    series = load_county_series(cases_path, args.county, populations[args.county])
    mcmc, ppc = run_inference(series, args.warmup, args.samples, args.chains, args.seed)

    print(f"\nCounty: {series.county}")
    print(f"Population: {int(series.population)}")
    print(f"Time points: {series.incidence.shape[0]}")
    print_summary(mcmc, ppc, series.incidence)


if __name__ == "__main__":
    numpyro.set_host_device_count(4)
    main()
