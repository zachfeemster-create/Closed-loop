# Frozen data snapshot (25 Oct 2026)

The exact inputs the locked Run 009 used, frozen so the demo runs even when the WHO or UNICEF APIs are down or have been updated.

- `who_modeled_series.csv`: the 1,376 WHO modeled series the scorer trains on (indicator, ISO3, 2000-2019), in training order.
- `raw_infant_deaths.csv`: WHO Mortality Database registered deaths under age 1, national, all causes, for countries complete 2000-2019.
- `igme_infant_deaths.csv`: UN IGME estimated infant deaths, 2000-2019.

Built by `make_snapshot.py` from a full `fetch_data.py` download. `python run_demo.py` uses this snapshot by default. `python run_demo.py --live` uses a fresh download in `./data` instead.
