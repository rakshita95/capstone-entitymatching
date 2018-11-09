# capstone-entitymatching

### Project Organization

------------
    │
    ├── modules
    │   ├── preprocessing
    │   │   ├── __init__.py       <- create preprocessing class and prepare cleaned data in matrices by
    │   │   │                        calling sub-preprocessing functinos
    │   │   ├── word_embedding.py <- functions to generate word embeddings matrices
    │   │   └── process_text.py   <- preprocessing functions for generic text fields
    │   │
    │   ├── feature_generation    <- generate features for to feed in modeling
    │   │   └── gen_similarities.py <- calculates different pair-wise similarities based on input data
    │   │
    │   └── modeling              <- train test split and ensembles
    │
    ├── run_scripts        <- Data set speicifc execution scripts that calls class & functions from modules
    │
    ├── exploratory        <- Jupyter notebooks for exploratory purposes
    │
    └── unit_tests         <- Pytest testing suites
--------

### Package Management

Use conda to maintain a virtual environment.

#### Documentation

Use numpy-style documentation.
