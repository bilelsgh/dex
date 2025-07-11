## What's in my data?


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.9.21-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)



This application equips data scientists with essential tools for dataset exploration, cleaning, and preparation. It simplifies the extraction of insights and ensures data is ready for machine learning algorithms, streamlining the entire workflow and enhancing productivity.

#### Project Structure

- `components/`: Contains Python classes used in this project.
      - `id_dataset.py`: Defines the IDDataset class for data preprocessing and manipulation.

- `dashboards/`: Simple and intuitive dashboards to visualize traffic data

- `helpers/`:
    - `data_preparation.py`: Preparation of every dataset used in the RL algorithm
    - `utils.py`: Some useful functions used in the project


- `.gitignore`: Specifies files and directories to be ignored by Git.
- `requirements.txt`: Lists the project dependencies.
- `conf.yaml`: General parameters for the Q-Learning algorithm (dataset to use, number of episodes, learning rate etc.).
- `config.py`: Set env variables



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

1. Set the parameters
```bash
conf.yaml
```

###### Dashboard
2. Run:
```bash
  streamlit run dashboards/front.py
```

#### Demo
![alt_text](https://github.com/bilelsgh/dataset_exploration/blob/master/demo.gif)
