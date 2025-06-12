# Nocarz

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**PL:** W momencie w którym dodawana jest oferta należy wypełnić wiele pól. Może dałoby się jakoś usprawnić ten proces?

**EN:** When adding an offer, many fields need to be filled in. Perhaps this process could be streamlined?


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
|
├── logs               <- Generated logs.
│
├── models             <- Trained and serialized models, model predictions, or model summaries.
│
├── notebooks          <- Jupyter notebooks.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment.
│
├── setup.cfg          <- Pip project configuration file.
│
└── nocarz             <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes nocarz a Python module.
    ├── api                     <- API endpoints for the microservice.
    ├── src                     <- Source code for the project.
    ├── config.py               <- Configuration file for the project.
```

--------
