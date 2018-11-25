# capstone-entitymatching

### Project Organization

------------
    │
    ├── modules
    │   ├── preprocessing
    │   │   ├── __init__.py         <- Create preprocessing class and preprocess data
    │   │   ├── word_embedding.py   <- Functions to generate word embeddings matrices
    │   │   └── process_**.py       <- Preprocessing functions for text and special fields
    │   │
    │   └── feature_generation      <- generate features for to feed in modeling
    │       └── gen_similarities.py <- calculates different pair-wise similarities based on input data
    │
    │
    ├── **_exploratory              <- Run scripts that evaluate our models on different datasets
    │
    └── unit_tests                  <- Pytest testing suites
--------

### Package Management

Use conda to maintain a virtual environment.

#### Documentation

Use numpy-style documentation.
