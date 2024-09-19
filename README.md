# PIOFE-Unrolling

Add a valid description here

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── ml
|   └── models         <- Trained and serialized models, model predictions, or model summaries. Effectively also acts as a model registry pertaining to this project
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for PIOFE-Unrolling
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials, including research papers (if applicable).
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file (if applicable)
│
└── src                <- Source code for use in this project.
    │
    ├── PIOFE-Unrolling
    |   └── __init__.py    <- Makes PIOFE-Unrolling a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── features.py
    │
    ├── modeling         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict.py
    │   ├── train.py
    |   └── custom-ml  <- Custom objects required for predict.py and train.py to function (if applicable)
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── plots.py
```

--------
