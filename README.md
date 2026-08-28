# Programming for Economists Portfolio

This portfolio contains our data analysis project, model analysis project, and exam project for the Programming for Economists course.

## Portfolio Contents

### 01_dataproject

The data project analyses income inequality in Denmark. It includes:

- `Project_1.ipynb`: the main project notebook
- `Analysis.ipynb`: additional analysis
- `Income_model.py` and `model 2.1.py`: supporting model code
- Excel files with the data used in the analysis
- Figures illustrating inequality and income distributions
- `DataProject.pdf`: the written project report

The project combines data management, descriptive statistics, visualisation, and economic analysis.

### 02_modelproject

The model project studies household consumption and government policy. It includes:

- `Model.py`: the main model implementation
- `Consumer.py`: the consumer-side model
- `Government.py`: the government-side model
- `ModelProject.pdf`: the written project report

The project uses Python to formulate, solve, and analyse an economic model.

### Exam project

The exam project contains our solutions to the three parts of the summer 2026 exam:

- **Real GDP across US states:** downloading and analysing real GDP and population data from FRED
- **The Solow model with a time-varying savings rate:** simulation, welfare analysis, grid search, and numerical optimisation
- **A portfolio with a risky and a safe asset:** vectorised Monte Carlo simulation, portfolio rebalancing, transaction costs, and expected utility

The exam files are located in the `Exam` folder:

For the final submission, this folder should be included as `03_examproject` in the zip file, as required by the exam assignment.

- `Exam.ipynb`: data analysis project for Question 1
- `examproject.ipynb`: exam project notebook
- `Exam2.ipynb` and `Exam3.ipynb`: model and portfolio analyses
- `states.py`, `SolowModel.py`, and `PortfolioModel.py`: supporting code

The exam analysis uses the seed `2026` for simulations where required. The FRED part requires an API key when the download cells are run. A cached data copy may be used if the API is unavailable, as specified in the exam assignment.

## Software and Packages

The Python code uses standard scientific Python packages, including:

- NumPy
- pandas
- Matplotlib
- SciPy
- requests

The notebooks can be opened and run in VS Code with the Jupyter and Python extensions installed.

## Reproducibility

Run the notebooks from top to bottom so that imports, data, calculations, figures, and reported results are generated in the intended order. The working directory should be the `AA_Project_1` folder, or the relevant notebook folder when importing local modules.

## Use of Generative AI

Generative AI tools were used as permitted by the exam instructions for coding support, debugging, checking calculations, and improving explanations. We reviewed, tested, and adapted all generated suggestions ourselves. A complete GAI declaration must be included with the submission.
