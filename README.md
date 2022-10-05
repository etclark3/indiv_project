# World Population Prediction
--------------
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a>
    <li><a href="#summary">Summary</a></li>
    <li><a href="#preliminary-questions">Questions</a></li>
    <li><a href="#planning">Planning</a></li>
    <li><a href="#data-dictionary">Data Dictionary</a></li>
    <li><a href="#Key-Findings-and-Takeaways">Key Findings and Takeaways</a></li>
    <li><a href="#recommendations">Recommendations</a></li>
    <li><a href="#additional-improvements">Additional Improvements</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#how-to-reproduce">How to Reproduce</a></li>
  </ol>
</details>
    
## About/Goals:
- (About) This project I wanted to veer into predicting the population by country. There are multiple datasets within https://www.kaggle.com/datasets/census/international-data?select=midyear_population_age_country_code.csv with varying features, but ended on a fairly straight-forward approach on world population based on the past 70 years of data. The data I'm working with predicts up until 2050 itself, so that is the data that is split.

- (Goals) I wanted to find the best model to determine future world population counts

## Summary:
- The best prediction model (Holt's Optimized) does well with validate, but less so with the test sample

## Preliminary Questions:
  1. Are the most populated countries in 1950 still so in 2022, 2050?
  1. Are country populations more equal in 1950 than 2022, 2050?

## Initial Hypothesis:
  - Population growth is somewhat steady over time

# Planning:
1. Acquire from .csv
    - Peer through datasets
    - Sift through files and find relevant data
2. Prepare
    - Find associated data and potentially merge files
3. Explore
    - Look for additional insights
4. Model
    - Find the best Model
5. Report Findings

## Data Dictionary:
<img width="713" alt="Screen Shot 2022-10-03 at 12 07 39 AM" src="https://user-images.githubusercontent.com/98612085/193505273-46f40821-2015-4d87-88a7-cbf5ef3ed32e.png">

## Key Findings and Takeaways:
- Holt's model doesn't work as well on test as it did on validate
- However, validates data consists of actual population counts (up until 2022), and it can be said the the leveling off that this data has following 2030, is not accurate. There are many factors that come to play, one of them being the lockdowns during the past two years that led to the United States' lowest population growth since the nation was founded.

#### Insights:
- In the last 70 years, world population has tripled 
- It is estimated to begin to slightly level off starting in 2030, but still almost quadruple by 2050
- Hypothesis is accurate

#### Model - Holt's Optimized (exponential=True) model had the best performance.

- Validate: 
  - '40,746,201' RMSE
  - '0.998' R-squared value
- Test: 
  - '379,245,054' RMSE
  - '0.76' R-squared value
  
## Recommendations:
  1. I would continue to use this model for predictions, but would potentially alter metrics to suit beyond 2022 population estimates
  2. Look at the data prior to 1950 going back 100 years

## Additional Improvements:
- Attempt a regression model based on country crude birth rate, mortality, and life expentancy to see if these results can be replicated using a different model type.

## Contact:
Everett Clark - everett.clark.t@gmail.com

## How to Reproduce:
#### Step 1 (Imports):  
- import numpy as np
- import pandas as pd
- import scipy.stats as stats
- from math import sqrt 
- import seaborn as sns
- import matplotlib.pyplot as plt
- from matplotlib.ticker import StrMethodFormatter, FuncFormatter
- import matplotlib.ticker as ticker
- from matplotlib.dates import DateFormatter
- import warnings
- warnings.filterwarnings('ignore')
- from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
- from statsmodels.tsa.api import Holt
- import statsmodels.api as sm
- from datetime import datetime
- from wrangle import prep_mp, pop_plot, total_pop, verify_split, evaluate, plot_eval, append_eval_df, predictions
- from wrangle import pred, val, known, exclude, plot_all, rmse, test_plot, growth, dist

#### Step 2 (Acquisition):  
- Acquire the database information from https://www.kaggle.com/datasets/census/international-data?select=midyear_population_age_country_code.csv

#### Step 3 (Preparation):  
- Prepare and split the dataset, use prep_mp. Then split is done 50% Train, 30% Validate, 20% Test

#### Step 4 (Exploration):
- Use seaborn or matplotlib.pyplot to create visualizations.

#### Step 5 (Modeling):
- Create models (Last Known, Simple Average, Holt's Optimized).
- Train each model and evaluate its accuracy on both the train and validate sets.
- Select the best performing model and use it on the test set.
- Document each step of the process and your findings.

[[Back to top](#top)]
