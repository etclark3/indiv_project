# Prediction
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
- The goal of this project is to find features or clusters of features to improve Zillow's log error for single family residences in three Southern California counties (Los Angeles, Orange, and Ventura) and to use these features to develop an improved machine learning model.

## Summary:
- After running four models on train and validate, the 2nd degree Polynomial Linear Regression model provided the lowest Root Mean Square Error (RMSE) compared to the baseline.
- We used the square footage of the home, ratio of bedrooms and bathrooms, lot size, age of the home, number of bathrooms, area cluster, and size cluster to predict logerror.. We selected a degree multiplier of 2. The RMSE of the selected model was 0.162 on train, 0.143 on validate, and 0.174 on test.
- The selected features had little impact in improving the overall prediction of log error when compared against baseline. The clusters did not significantly reduce the RMSE, but there was a very small improvement when using the absolute value for the log error. Overall, none of the models significantly outperformed Zillow's current model.

## Preliminary Questions:
  1. What is the relationship between square feet and log error? 
  1. Do area clusters have a large impact on the overall log error?
  1. Does the size of the home affect log error? Can that error be better determined by clustering by size?
  1. Does the location have an effect on log error? Where does the most log error occur?

## Initial Hypothesis:
  -

# Planning:
![image](https://user-images.githubusercontent.com/98612085/191094469-0c50c67a-d7e1-4711-9eb8-06261a8f10bb.png)

## Data Dictionary:

## Key Findings and Takeaways:

#### Insights:
- 

#### Best predictor features, Using visualization and statistical testing:


#### Model - The 2nd Degree Polynomial Regression model had the best performance.

- Train: 
  - '' RMSE
  - '' R-squared value
- Validate: 
  - '' RMSE
  - '' R-squared value
- Test: 
  - '' RMSE
  - '' R-squared value

#### (For Comparison)
    - Train RMSE (Mean): ''
    - Train RMSE (Median): ''
  
## Recommendations:
  1. 

## Additional Improvements:
-

## Contact:
Everett Clark - everett.clark.t@gmail.com

## How to Reproduce:
**In order to reproduce this project, you will need server credentials to the Codeup database or a .csv of the data**

#### Step 1 (Imports):  
- import pandas as pd
- import numpy as np
- import matplotlib.pyplot as plt

#### Step 2 (Acquisition):  
- 

#### Step 3 (Preparation):  
- 

#### Step 4 (Exploration):
- 


[[Back to top](#top)]
