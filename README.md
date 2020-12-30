# Finding the Higgs Boson using Machine Learning

The purpose of this project is to implement machine learning algorithms for binary classification of the Higgs boson dataset. The resulting predictions are submitted to [AIcrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs).

In this README you will find general information about the methods, and a more detailed documentation is given within the functions.

## Getting Started

The following libraries are needed:

    numpy
    matplotlib
    seaborn

## Project Structure
The project structure is described as the following:

```markdown
.

├── data: place datasets into this folder
        ├── train.csv: train dataset
        └── test.csv: test dataset
├── report: contains LaTeX and pdf version of the report
├── README.md: this file
├── impl_proj1.py: auxiliary functions including data preprocessing, accuracy, feature augmentation and cross validation
├── implementations.py: contains **all the implementations** required as given in the project description
├── implementations_modified.py: modified implementations.py file for testing our pipeline
├── proj1_helpers.py: provided functions and modified predicted_labels function for logistic regression
├── project1.ipynb: Jupyter notebook of project
├── run.py: contains the **final code** to train the model
```

The datasets are not included in the repo due to its file size, but they can be found on [AIcrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs). The datasets will need to be placed in the ```data``` folder.

## Files
### run.py

This script produces a csv file containing the predictions `sample-submission.csv`. The following are executed:

  a) loading the data <br>
  b) data processing <br>
  c) feature augmentation <br>
  d) models <br>
  e) cross validation <br>
  f) obtain weight vector <br>
  g) compute predictions and create the .csv file


### implementations.py

The following functions were implemented:

| Function            | Details |
|-------------------- |-----------|
| `least_squares_GD(y, tx, initial_w, max_iters, gamma)`  | `Linear regression using gradient descent`  |
| `least_squares_SGD(y, tx, initial_w, max_iters, gamma)` | `Linear regression using stochastic gradient descent`  |
| `least_squares(y, tx)`     | `Least squares using normal equations` |
| `ridge_regression(y, tx, lambda_)`  | `ridge regression using normal equations` |
| `logistic_regression(y, x, initial_w, max_iters, gamma)`| `logistic regression using gradient descent or SGD` |
| `reg_logistic_regression(y, x, lambda_, initial_w, max_iters, gamma)` | `regularized logistic regression using gradient descent or SGDs` |

## Authors

* *Pedro Bedmar Lopez*
* *Kevin Qiu*