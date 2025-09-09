# The WoMAD Code

Let's go through the WoMAD code: Usage, Installation, and the Structure of our Repository.

## Installation and Usage of the WoMAD Package

## The Repository

Look at this refreshing tree of chaos, the structure of the repo:

```bash
.
├── data
│   ├── model-ready
│   └── processed
├── docs
│   ├── img
│   ├── index.md
│   ├── logs
│   │   └── index.md
│   ├── the-code.md
│   └── the-science.md
├── figures
├── LICENSE
├── Makefile
├── mkdocs.yml
├── pyproject.toml
├── README.md
├── tests
│   └── test-all-modules.py
├── utils.py
├── WoMAD
│   ├── data-module.py
│   ├── hyperparameter-module.py
│   ├── __init__.py
│   ├── model-setup-module.py
│   ├── model-train-module.py
│   ├── model-valid-module.py
│   ├── result-interp-module.py
│   └── WoMAD-config.py
└── WoMAD-main.py

9 directories, 20 files
```

Now, let's go through these files one by one.
