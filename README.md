# The-Best-Classifier-Loan-Predicton
This loan prediction project aims to assess the likelihood of loan repayment based on historical data. It involves data loading, preprocessing, and feature engineering to prepare the dataset. Four machine learning algorithms - K Nearest Neighbor, Decision Tree, Support Vector Machine, and Logistic Regression - are trained and evaluated for their accuracy in predicting loan outcomes. The KNN model emerges as the most effective, achieving the highest Jaccard index and F1-score among the evaluated models.

**Language and Libraries:** <br>
Python 3.8, Numpy, Pandas, Matplotlib, SkLearn.

**Introduction and Objective:** <br>
The project begins with an introduction to the task at hand, which is to predict whether a loan will be paid off or not. This is a common problem in the finance industry where lenders need to assess the risk associated with granting loans. <br> <br>
**Loading Libraries:** <br>
The necessary Python libraries are imported, including pandas for data manipulation, numpy for numerical operations, matplotlib for visualization, seaborn for enhanced plotting, and scikit-learn for machine learning algorithms. <br> <br>
**Dataset Overview:** <br>
A description of the dataset is provided, outlining the fields included in the dataset such as loan status, principal amount, terms, effective date, due date, age, education, and gender. <br> <br>
**Data Loading:** <br>
The dataset (loan_train.csv) is loaded into a pandas DataFrame (df) using the pd.read_csv() function. <br> <br>
**Data Preprocessing:** <br>
Date columns (due_date and effective_date) are converted to datetime objects for easier manipulation.
Visualizations are created to better understand the data, such as histograms showing the distribution of principal amounts and ages among genders and loan statuses.
Feature engineering is performed, such as extracting the day of the week from the effective_date column and creating a binary feature (weekend) indicating whether the loan was issued on a weekend.
Categorical variables (Gender and education) are converted to numerical values using one-hot encoding. <br><br>
**Feature Selection:** <br>
The feature set (X) is defined, including selected features for prediction such as principal amount, terms, age, gender, and weekend. <br><br>
**Data Normalization:** <br>
The feature set is standardized using preprocessing.StandardScaler() to ensure zero mean and unit variance. <br><br>
**Train-Test Split:** <br>
The dataset is split into training and testing sets using train_test_split() from scikit-learn, with 80% of the data used for training and 20% for testing. <br><br>
**Model Building:** <br>
**K Nearest Neighbor (KNN):** The KNN algorithm is applied with different values of k, and the best k value is determined using cross-validation. <br><br>
**Decision Tree:** <br>
A decision tree classifier is built with entropy as the criterion for information gain.
Support Vector Machine (SVM): Two SVM models are trained, one with the Radial Basis Function (RBF) kernel and the other with the Sigmoid kernel. <br><br>
**Logistic Regression:** <br>
A logistic regression model is built with regularization parameter C set to 0.01 and solver as 'liblinear'. <br><br>
Model Evaluation: <br>
**KNN:** The accuracy of the KNN model is evaluated using the Jaccard index and F1-score. <br>
**Decision Tree:** The accuracy of the decision tree model is evaluated using the Jaccard index and F1-score, and the tree is visualized. <br>
**SVM:** The accuracy of the SVM models is evaluated using the Jaccard index and F1-score. <br>
**Logistic Regression:** The accuracy of the logistic regression model is evaluated using the Jaccard index, F1-score, and LogLoss. <br><br>
**Report:** Finally, a summary table is created, showing the performance metrics (Jaccard index, F1-score, and LogLoss) for each algorithm. <br>

**User Guide:** <br>
**Requirements:** <br>
1. Ensure you have Jupyter Notebook and Python installed on your system. <br>
2. Open Jupyter Notebook: Launch Jupyter Notebook and navigate to the project folder. <br>
3. Execute Cells: Run each cell in the notebook sequentially to view code, visualizations, and model results. <br>
4. Interact and Explore: Feel free to modify code and parameters to interact with the project. The dataset link is provided within the notebook for reference. <br>

