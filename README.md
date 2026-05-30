# Machine-Learning-Project
Store Sales Forecasting Project:
<br>
This repository contains a Machine Learning project designed to predict future product sales for Corporación Favorita, a large grocery chain based in Ecuador. The goal is to build a predictive model that accurately forecasts demand across various store locations and product categories to optimize inventory management.
<br>
Overview:

Goal: Forecast product demand across multiple store locations and categories.

Data Focus: Uses the most recent two years of data (2016–2017) for efficiency.

Algorithm: LinearRegression via scikit-learn.

 The Pipeline:
 
Data Prep: Loads train/test data.

Feature Engineering: Extracts Month, Day, and Day of Week from dates to catch shopping trends.

Encoding: Converts text categories (e.g., "GROCERIES") into numeric codes.

Training: Learns patterns from store IDs, categories, promotions, and dates.

Output: Generates a structured submission.csv file.
