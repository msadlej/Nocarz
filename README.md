# Nocarz

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

W momencie w którym dodawana jest oferta należy wypełnić wiele pól. Może dałoby się jakoś usprawnić ten proces?


## Installation

1. Clone the repository and navigate to the project directory
```bash
git clone [repo]
cd [repo]
```

3. Install the project
```bash
just install
```

4. Run the microservice
```bash
just run
```


## Project Organization

```
├── Justfile           <- Justfile with convenience commands
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Documentation for the project.
│
├── models             <- Trained and serialized models, model predictions, or model summaries.
│
├── notebooks          <- Jupyter notebooks.
|
├── logs               <- Generated logs.
│
├── pyproject.toml     <- Configuration file for mypy.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment.
│
├── setup.cfg          <- Project configuration file.
│
└── nocarz             <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes nocarz a Python module.
    ├── api                     <- API endpoints for the microservice.
    ├── src                     <- Source code for the project.
    ├── config.py               <- Configuration file for the project.
```

--------
