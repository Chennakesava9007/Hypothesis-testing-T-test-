# Hypothesis-testing-T-test-

1. Importing Required Libraries
The necessary Python libraries are imported:

pandas: Data manipulation and analysis.
seaborn, matplotlib: Visualization.
numpy: Numerical computations.
sklearn: Machine learning models (Logistic Regression and Naive Bayes).
scipy: Statistical tests (t-test).

2. Adding and Saving the Dataset
The football dataset is downloaded using the URL https://www.football-data.co.uk/mmz4281/2223/E0.csv.
It is read into a Pandas DataFrame and saved locally as FootballDataset.csv.
The relevant columns extracted from the dataset are:
HomeTeam: Name of the home team.
AwayTeam: Name of the away team.
FTHG: Full-time home goals.
FTAG: Full-time away goals.
FTR: Full-time result (Home win, Draw, Away win).

3. Preprocessing the Data
Dropping Missing Values: Removes rows with any missing values using dropna.
Label Encoding: Categorical columns (HomeTeam, AwayTeam, and FTR) are converted to numeric values using LabelEncoder.

5. Defining Features and Target
X: Features used for prediction (HomeTeam, AwayTeam, FTHG, FTAG).
y: Target variable (FTR), representing the match result.

6. Splitting Data for Training and Testing
The dataset is split into training (70%) and testing (30%) subsets using train_test_split with a random state of 101 for reproducibility.

7. Model Training
Two models are trained to predict the outcome (FTR):

Logistic Regression:
A probabilistic linear model for classification.
Predicts probabilities for the match outcomes.
Trained using logit.fit(Xtrain, ytrain).
Naive Bayes (GaussianNB):
Assumes features are independent and normally distributed.
Trained using nb.fit(Xtrain, ytrain).

8. Model Predictions
Predictions are made on the test dataset (Xtest) using both models:

pred1: Logistic Regression predictions.
pred2: Naive Bayes predictions.

9. Saving Predictions
The predictions of both models are stored in a CSV file (Performance11.csv) for further analysis.

10. Statistical Comparison (T-Test)
A t-test is performed to compare the means of the predictions from both models. This checks if there is a statistically significant difference in their performance.
Null Hypothesis (H₀): The two models have similar accuracy (no significant difference).
Alternative Hypothesis (H₁): The two models have different accuracies.
The t-test calculates:
t_statistic: Difference in the means scaled by variance.
p_value: Probability of observing the data if the null hypothesis is true.

11. Hypothesis Test Results
If p_value < alpha (α = 0.05), the null hypothesis is rejected, indicating a significant difference between the models.
If t_statistic > 0, Model 1 (Logistic Regression) is better.
Else, Model 2 (Naive Bayes) is better.
If p_value >= alpha, the null hypothesis is not rejected, indicating no significant difference.


Outputs:
T-Statistic: Quantifies the difference in means scaled by variance.
P-Value: The probability of observing the data assuming the null hypothesis is true.
Decision:
Reject or fail to reject the null hypothesis.
Determine which model performs better if the difference is significant.
