## DeX - Data explorer


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.9.21-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)



This application equips data scientists with essential tools for dataset exploration, cleaning, and preparation. It simplifies the extraction of insights and ensures data is ready for machine learning algorithms, streamlining the entire workflow and enhancing productivity.

#### DeX - Analysis
Explore your data to gather the most relevant insights
![alt text](./assets/demo_1.gif "Analysis")

#### DeX - Pre-processing
Apply state-of-the-art preprocessing operations to fit your data to Machine Learning algorithms.
(Normalization, Standardization, Encoding ..)
![alt text](./assets/demo_2.gif "Pre-pocessing")

#### Installation

1. Clone the repository:

```bash
  git clone https://github.com/bilelsgh/dataset_exploration
  cd dataset_exploration
```
2. Install the libraries
```bash
pip install requirements.txt
```

#### Run

1. Run:
```bash
  streamlit run src/home.py
```

---

##### todo
    - analyse the statistics and return insights
    - dim reduction
    - replace values
    - rename columns
