---
title: Company Wise
description: A reference page in my new Starlight docs site.
---

# Google

## Role: Data Scientist

### 1. Why do you use feature selection?
Feature selection is used to **reduce overfitting**, **improve model performance**, and **decrease computational cost** by selecting only the most relevant features for the model.

### 2. What is the effect on the coefficients of logistic regression if two predictors are highly correlated?
When two predictors are highly correlated, it can lead to **multicollinearity**, which makes the coefficients **unstable** and their **interpretation unreliable**. This can affect the model's **predictive power**.

### 3. What are the confidence intervals of the coefficients?
Confidence intervals of the coefficients provide a range within which the true value of the coefficient is expected to lie with a certain level of confidence (usually **95%**). Narrow intervals indicate **precise estimates**, while wide intervals suggest **greater uncertainty**.

### 4. What’s the difference between Gaussian Mixture Model and K-Means?
The **Gaussian Mixture Model (GMM)** assumes that the data is generated from a mixture of several Gaussian distributions with unknown parameters. In contrast, **K-Means** clusters data by minimizing the variance within each cluster. GMM is **probabilistic**, while K-Means is **distance-based**.

### 5. How do you pick k for K-Means?
To pick the optimal **k** for K-Means, you can use methods like the **Elbow Method**, **Silhouette Score**, or **Gap Statistic**. These methods help determine the value of **k** that best fits the data.

### 6. How do you know when Gaussian Mixture Model is applicable?
A Gaussian Mixture Model is applicable when the data is thought to be generated from multiple **Gaussian distributions**. It's suitable when clusters have **different shapes, sizes, and densities**.

### 7. Assuming a clustering model’s labels are known, how do you evaluate the performance of the model?
When the labels are known, you can evaluate the performance using **metrics** such as **Adjusted Rand Index (ARI)**, **Mutual Information Score**, and **F1 Score**. These metrics compare the predicted labels to the true labels to assess the clustering accuracy.



# Company: Uber

## Role: Data Scientist

### 1. Pick any product or app that you really like and describe how you would improve it.
One product I really like is **Spotify**. To improve it, I would enhance the **personalized playlist algorithms** by incorporating **user mood detection** from their listening patterns and **feedback**. This would provide more **accurate recommendations** and improve user **engagement**.

### 2. How would you find an anomaly in a distribution?
To find an anomaly in a distribution, I would use **statistical methods** such as **z-scores** to detect outliers, or **machine learning algorithms** like **Isolation Forest** or **One-Class SVM**. These methods help identify data points that **deviate significantly** from the norm.

### 3. How would you go about investigating if a certain trend in a distribution is due to an anomaly?
To investigate if a trend is due to an anomaly, I would conduct a **time series analysis** and use techniques like **seasonal decomposition** to separate the trend, seasonality, and residuals. Additionally, performing a **root cause analysis** and checking for any **external factors** or **sudden changes** in the data would help confirm the anomaly.

### 4. How would you estimate the impact Uber has on traffic and driving conditions?
To estimate Uber's impact on traffic and driving conditions, I would collect and analyze **traffic data** before and after Uber's introduction in a region. Using **statistical models** and **regression analysis**, I could identify changes in **traffic congestion**, **average travel times**, and **accident rates**.

### 5. What metrics would you consider using to track if Uber’s paid advertising strategy to acquire new customers actually works? How would you then approach figuring out an ideal customer acquisition cost?
Metrics to track Uber's paid advertising strategy include **Customer Acquisition Cost (CAC)**, **Conversion Rate**, **Customer Lifetime Value (CLV)**, and **Return on Ad Spend (ROAS)**. To figure out an ideal CAC, I would compare it to the **CLV** to ensure the cost of acquiring a customer is **less than** the **revenue generated** over their lifetime, ensuring a **profitable strategy**.



# TCS
## Data Scientist
### 1. Explain about Time series models you have used?
I have used **ARIMA**, **SARIMA**, and **LSTM** for time series analysis. **ARIMA** is good for stationary data, **SARIMA** handles seasonality, and **LSTM** is powerful for capturing long-term dependencies in sequential data.

### 2. SQL Questions - Group by Top 2 Salaries for Employees - use Row num and Partition
sql code
SELECT * FROM (
  SELECT EmployeeID, Salary, ROW_NUMBER() OVER (PARTITION BY Department ORDER BY Salary DESC) as RowNum
  FROM Employees
) as Ranked
WHERE RowNum <= 2;

### 3. Pandas find Numeric and Categorical Columns. For Numeric columns in Data frame, find the mean of the entire column and add that mean value to each row of those numeric columns.
python code
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': ['a', 'b', 'c']})
numeric_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

for col in numeric_cols:
    mean_val = df[col].mean()
    df[col] = df[col] + mean_val

### 4. What is Gradient Descent? What is Learning Rate and Why we need to reduce or increase? Why Global minimum is reached and Why it doesn’t improve when increasing the LR after that point?
Gradient Descent is an optimization algorithm used to minimize the cost function. The learning rate determines the step size. A high learning rate can overshoot the minimum, while a low rate can slow convergence. The global minimum is the lowest point in the cost function, and increasing the learning rate beyond this point can lead to divergence.

### 5. What is Log-Loss and ROC-AUC?
Log-Loss measures the performance of a classification model whose output is a probability value between 0 and 1. ROC-AUC measures the ability of the model to distinguish between classes, with a higher value indicating better performance.

### 6. What is Multi-collinearity? How will you choose one feature if there are 2 highly correlated features? Give Examples with the techniques used.
Multi-collinearity occurs when two or more predictors are highly correlated, making it difficult to isolate their individual effects. To choose one feature, you can use VIF (Variance Inflation Factor) or correlation matrix. Techniques like PCA (Principal Component Analysis) can also be used.

### 7. VIF – Variance Inflation Factor – Explain.
VIF measures how much the variance of a regression coefficient is inflated due to multicollinearity. A VIF value greater than 10 indicates high multicollinearity, suggesting the need to remove the feature.

### 8. Do you know to use Amazon SageMaker for MLOPS?
Yes, I have experience using Amazon SageMaker for deploying and managing ML models. It helps with model training, deployment, and monitoring.

### 9. Explain your Projects end to end (15-20mins).
I have worked on several projects involving data collection, preprocessing, model selection, training, evaluation, and deployment. One example is a credit risk prediction model where I used logistic regression and random forest to classify borrowers. The project involved feature engineering, model tuning, and deploying the model using AWS SageMaker.

# Capital One
## Data Scientist

### 1. How would you build a model to predict credit card fraud?
To predict credit card fraud, I would use a supervised learning algorithm like Random Forest or XGBoost. The process includes data preprocessing, handling imbalanced data using techniques like SMOTE, and feature engineering. Evaluation would involve using metrics like Precision-Recall AUC.

### 2. How do you handle missing or bad data?
I handle missing data by imputation (mean, median, mode), deleting rows/columns, or using algorithms that handle missing data natively. Bad data is addressed by cleaning, standardization, and outlier detection.

### 3. How would you derive new features from features that already exist?
I would use feature engineering techniques such as polynomial features, log transformations, interaction terms, and domain-specific knowledge to create meaningful new features.

### 4. If you’re attempting to predict a customer’s gender, and you only have 100 data points, what problems could arise?
With only 100 data points, issues like overfitting, high variance, and low statistical power could arise. The model may not generalize well, and the predictions could be biased.

### 5. Suppose you were given two years of transaction history. What features would you use to predict credit risk?
I would use features such as transaction frequency, average transaction amount, payment history, credit utilization, and demographic information.

### 6. Design an AI program for Tic-tac-toe
To design an AI for Tic-tac-toe, I would use the Minimax algorithm with alpha-beta pruning. The AI would evaluate the game board and make optimal moves to either win or block the opponent.

### 7. Explain overfitting and what steps you can take to prevent it.
Overfitting occurs when a model performs well on training data but poorly on unseen data. To prevent it, use techniques like cross-validation, regularization (L1/L2), pruning (in decision trees), and ensuring sufficient training data.

### 8. Why does SVM need to maximize the margin between support vectors?
SVM maximizes the margin between support vectors to ensure better generalization of the model. A larger margin reduces the model's complexity and improves its ability to classify new data points accurately.

# Company: Latentview Analytics

## Role: Data Scientist

### 1. What is mean and median?
**Mean** is the average of a dataset, while **median** is the middle value when the data is sorted. Median is less affected by **outliers**.

### 2. Difference between normal and gaussian distribution
There is no difference; **normal distribution** is also known as the **Gaussian distribution**. It is a **bell-shaped** curve where most of the data points cluster around the mean.

### 3. What is central limit theorem?
The **Central Limit Theorem** states that the sampling distribution of the sample mean approaches a **normal distribution** as the sample size grows, regardless of the population's distribution.

### 4. What is null hypothesis?
The **null hypothesis** is a statement that there is **no effect** or **no difference**. It serves as the starting assumption for **statistical testing**.

### 5. What is confidence interval?
A **confidence interval** is a range of values that is likely to contain the **population parameter** with a certain level of confidence, typically **95%**.

### 6. What is covariance and correlation and how will you interpret it?
**Covariance** measures the direction of the relationship between two variables, while **correlation** measures the **strength and direction** of the relationship on a standardized scale from **-1 to 1**.

### 7. How will you find out the outliers in the dataset and is it always necessary to remove outliers?
Outliers can be found using methods like **z-scores**, **IQR** method, or **visualizations** like **box plots**. It is not always necessary to remove outliers; it depends on the context and the **impact on the analysis**.

### 8. Explain about Machine Learning
**Machine Learning** is a branch of AI that involves training algorithms to learn patterns from data and make predictions or decisions without being explicitly programmed.

### 9. Explain the algorithm of your choice
For example, **Decision Trees** are a non-parametric supervised learning method used for classification and regression. They split the data into branches to form a tree structure based on feature values.

### 10. Different methods of missing values imputation
Methods include **mean/median/mode imputation**, **interpolation**, **k-nearest neighbors**, and **using model-based imputation**.

### 11. Explain me your ML project
Provide a brief overview of your ML project, including the **problem statement**, **dataset**, **approach**, **algorithms used**, and **results**.

### 12. How did you handle imbalance dataset?
Handling imbalanced datasets can involve techniques like **resampling** (oversampling/undersampling), **SMOTE**, or using algorithms that are robust to imbalance.

### 13. What is stratified sampling?
**Stratified sampling** involves dividing the population into **strata** and sampling from each stratum proportionally to ensure all groups are adequately represented.

### 14. Difference between standard scaler and normal scaler
**StandardScaler** standardizes features by removing the mean and scaling to unit variance. **Normalizer** scales each sample individually to have unit norm.

# Company: Verizon

## Role: Data Scientist

### 1. How many cars are there in Chennai? How do you structurally approach coming up with that number?
Approach by using **estimation techniques**: estimate the population, average car ownership per household, and other factors like commercial vehicles. Combine these estimates to get a rough number.

### 2. Multiple Linear Regression?
**Multiple Linear Regression** models the relationship between two or more predictor variables and a response variable by fitting a linear equation.

### 3. OLS vs MLE?
**OLS (Ordinary Least Squares)** minimizes the sum of squared residuals, while **MLE (Maximum Likelihood Estimation)** maximizes the likelihood function based on the data.

### 4. R2 vs Adjusted R2? During Model Development which one do we consider?
**R²** measures the proportion of variance explained by the model. **Adjusted R²** adjusts for the number of predictors, making it more reliable for model selection.

### 5. Lift chart, drift chart
A **lift chart** shows the improvement of a model over random guessing. A **drift chart** monitors changes in model performance over time.

### 6. Sigmoid Function in Logistic regression
The **sigmoid function** maps any real-valued number to a value between 0 and 1, used in logistic regression to model the probability of the binary outcome.

### 7. ROC what is it? AUC and Differentiation?
**ROC (Receiver Operating Characteristic)** curve plots true positive rate vs. false positive rate. **AUC (Area Under the Curve)** measures the ability of the model to distinguish between classes.

### 8. Linear Regression from Multiple Linear Regression
**Linear Regression** involves one predictor and one response variable. **Multiple Linear Regression** involves multiple predictors.

### 9. P-Value what is it and its significance? What does P in P-Value stand for? What is Hypothesis Testing? Null hypothesis vs Alternate Hypothesis?
**P-Value** indicates the probability of obtaining the observed results if the null hypothesis is true. **P** stands for **probability**. **Hypothesis testing** involves testing an assumption (null hypothesis) against an alternative hypothesis.

### 10. Bias Variance Trade off?
**Bias-Variance Tradeoff** refers to the balance between a model's ability to minimize bias and variance to achieve optimal prediction accuracy.

### 11. Overfitting vs Underfitting in Machine Learning?
**Overfitting** occurs when a model learns noise in the training data. **Underfitting** occurs when a model is too simple to capture the underlying pattern.

### 12. Estimation of Multiple Linear Regression
Estimate the coefficients by minimizing the residual sum of squares using **Ordinary Least Squares (OLS)**.

### 13. Forecasting vs Prediction difference? Regression vs Time Series?
**Forecasting** typically involves time-series data to predict future values. **Prediction** can apply to any data to predict unknown outcomes. **Regression** focuses on relationships between variables, while **Time Series** deals with data indexed over time.

### 14. p, d, q values in ARIMA models
In **ARIMA** models: **p** is the number of lag observations, **d** is the number of times the data needs to be differenced to become stationary, and **q** is the size of the moving average window.



# Company: Fractal

## Role: Data Scientist

### 1. Difference between array and list
**Arrays** are fixed-size, homogeneous data structures, while **lists** are dynamic and can contain elements of different types.

### 2. Map function
The **map function** applies a given function to each item of an iterable (like a list) and returns a list of the results.

### 3. Scenario: If a coupon is distributed randomly to customers of Swiggy, how to check their buying behavior?
**Segmenting customers:** Group customers based on their behavior or characteristics.
**Compare customers who got a coupon and who did not:** Use statistical tests to analyze the differences in buying behavior between the two groups.

### 4. Which is faster for lookup: dictionary or list?
**Dictionaries** are faster for lookups compared to lists because they use a hash table for indexing.

### 5. How to merge two arrays
To merge two arrays, you can use the **concatenate function** in libraries like NumPy or use the **+ operator** for lists in Python.

### 6. How much time SVM takes to complete if 1 iteration takes 10 seconds for 1st class, and there are 4 classes.
If 1 iteration takes **10 seconds** for the 1st class and there are **4 classes**, the total time would be **40 seconds**.

### 7. Kernels in SVM, their difference
SVM kernels like **linear, polynomial, and RBF** transform the input data into a higher-dimensional space to make it easier to classify. Their differences lie in how they map the data and the shape of the decision boundaries they create.

# Company: Infosys

## Role: Data Scientist

### 1. Curse of dimensionality? How would you handle it?
The **curse of dimensionality** refers to various phenomena that arise when analyzing and organizing data in high-dimensional spaces. Handle it using **dimensionality reduction techniques** like PCA or **feature selection** methods.

### 2. How to find the multicollinearity in the dataset?
Use **Variance Inflation Factor (VIF)** or **correlation matrices** to detect multicollinearity in the dataset.

### 3. Explain the different ways to treat multicollinearity!
Treat multicollinearity by **removing highly correlated predictors**, using **principal component analysis (PCA)**, or **regularization techniques** like Ridge regression.

### 4. How do you decide which feature to keep and which feature to eliminate after performing a multicollinearity test?
Eliminate features with the **highest VIF values** or those contributing least to the model's predictive power after a multicollinearity test.

### 5. Explain logistic regression
**Logistic regression** is a statistical model that predicts the probability of a binary outcome using a logistic function.

### 6. We have a sigmoid function which gives us the probability between 0-1, then what is the need for log-loss in logistic regression?
**Log-loss** measures the accuracy of the predictions, penalizing false classifications more heavily, thus providing a more precise evaluation of the model.

### 7. P-value and its significance in statistical testing?
A **P-value** indicates the probability of obtaining the observed results, assuming the null hypothesis is true. It helps in determining the **statistical significance** of the test results.

### 8. How do you split the time series data and evaluation metrics for time series data?
Split time series data using **train-test split**, considering the temporal order. Evaluation metrics include **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **Root Mean Squared Error (RMSE)**.

### 9. How did you deploy your model in production? How often do you retrain it?
Deploy the model using **cloud services or containerization tools**. Retrain it periodically based on **new data availability** and **performance monitoring**.

# Company: Wipro

## Role: Data Scientist

### 1. Difference between WHERE and HAVING in SQL
**WHERE** filters records before any groupings are made, while **HAVING** filters records after groupings have been made.

### 2. Basics of Logistic Regression
**Logistic Regression** predicts the probability of a binary outcome using a logistic function.

### 3. How do you treat outliers?
Treat outliers by **removing them**, **transforming the data**, or using **robust statistical methods**.

### 4. Explain confusion matrix
A **confusion matrix** is a table used to evaluate the performance of a classification model, showing the true positives, false positives, true negatives, and false negatives.

### 5. Explain PCA
**PCA** reduces dimensionality by transforming the data into a set of linearly uncorrelated variables called **principal components**, based on the **covariance matrix and eigenvectors**.

### 6. How do you cut a cake into 8 equal parts using only 3 straight cuts?
To cut a cake into **8 equal parts** using only **3 straight cuts**, make two vertical cuts to divide the cake into 4 quadrants, then make one horizontal cut across all quadrants.

### 7. Explain K-means clustering
**K-means clustering** partitions data into **k clusters** by minimizing the variance within each cluster.

### 8. How is KNN different from K-means clustering?
**KNN** is a **supervised learning** algorithm for classification, while **K-means** is an **unsupervised learning** algorithm for clustering.

### 9. What would be your strategy to handle a situation indicating an imbalanced dataset?
Handle imbalanced datasets using **resampling techniques**, **synthetic data generation**, or using **evaluation metrics** that are robust to imbalance, like **F1 score**.

### 10. Stock market prediction: You would like to predict whether or not a certain company will declare bankruptcy within the next 7 days (by training on data of similar companies that had previously been at risk of bankruptcy). Would you treat this as a classification or a regression problem?
This would be treated as a **classification problem** since the outcome is a **binary decision** (bankruptcy or not).




# Company: Accenture

## Role: Data Scientist

### 1. What is the difference between K-NN and K-Means clustering?
**K-NN (K-Nearest Neighbors)** is a **supervised** learning algorithm used for classification and regression, where a data point is classified based on majority vote from its K nearest neighbors. **K-Means** is an **unsupervised** clustering algorithm that partitions data into K clusters by minimizing the variance within each cluster.

### 2. How to handle missing data? What imputation techniques can be used?
To handle missing data, you can use **imputation techniques** such as **mean, median, mode imputation**, or more advanced methods like **K-NN imputation**, **multiple imputation**, and **predictive modeling** to fill in missing values.

### 3. Explain topic modeling in NLP and various methods in performing topic modeling.
**Topic modeling** in NLP is a technique to discover the abstract topics within a collection of documents. Common methods include **Latent Dirichlet Allocation (LDA)** and **Non-Negative Matrix Factorization (NMF)**.

### 4. Explain how you would find and tackle an outlier in the dataset.
To find outliers, you can use methods like **box plots**, **Z-scores**, or **IQR (Interquartile Range)**. Tackling outliers involves either **removing them** if they are errors, **transforming** the data, or using **robust algorithms** that are not sensitive to outliers.

### Follow up: What about inliers?
Inliers are data points that fit well within the overall distribution but might still be problematic. Techniques like **robust statistical methods** or **re-sampling techniques** can be used to address inliers.

### 5. Explain back propagation in a few words and its variants.
**Backpropagation** is a method to compute the gradient of the loss function and update the weights in a neural network. Variants include **Stochastic Gradient Descent (SGD)**, **Mini-batch Gradient Descent**, and **Adam Optimizer**.

### 6. Is interpretability important for machine learning models? If so, ways to achieve interpretability for a machine learning model?
**Yes, interpretability is important** to ensure transparency, trust, and actionable insights. Methods include **model simplification**, **visualization techniques**, **SHAP values**, and **LIME** (Local Interpretable Model-agnostic Explanations).

### 7. How would you design a data science pipeline?
A data science pipeline involves **data collection**, **data preprocessing**, **feature engineering**, **model training**, **evaluation**, **hyperparameter tuning**, and **deployment**.

### 8. Explain bias-variance trade-off. How does this affect the model?
The **bias-variance trade-off** refers to the balance between **bias** (error due to oversimplification) and **variance** (error due to sensitivity to small fluctuations). High bias leads to **underfitting**, while high variance leads to **overfitting**. The goal is to find a model that minimizes both.

### 9. What does a statistical test do?
A **statistical test** determines whether there is enough evidence to reject a null hypothesis. It provides a **p-value** to measure the strength of the evidence against the null hypothesis.

### 10. How to determine if a coin is biased? Hint: Hypothesis testing
To determine if a coin is biased, use **hypothesis testing**: set up a null hypothesis (coin is fair) and an alternative hypothesis (coin is biased). Conduct a **chi-square test** or **binomial test** to see if the observed outcomes deviate significantly from the expected outcomes.

---

# Company: Tiger Analytics

## Role: Senior Analyst

### 1. What is deep learning, and how does it contrast with other machine learning algorithms?
**Deep learning** is a subset of machine learning involving neural networks with many layers (deep networks). It excels in handling large volumes of unstructured data. Unlike traditional machine learning algorithms, which rely heavily on feature engineering, deep learning models can **automatically learn features** from raw data.

### 2. When should you use classification over regression?
Use **classification** when the output is a **categorical variable** and **regression** when the output is a **continuous variable**.

### 3. Using Python how do you find Rank, linear and tensor equations for a given array of elements? Explain your approach.
You can use **NumPy** or **SciPy** libraries to find rank using `numpy.linalg.matrix_rank()`, solve linear equations using `numpy.linalg.solve()`, and work with tensors using the **TensorFlow** or **PyTorch** libraries.

### 4. What exactly do you know about Bias-Variance decomposition?
**Bias-Variance decomposition** separates the prediction error into three components: **bias**, **variance**, and **irreducible error**. It helps understand the trade-off and how to optimize the model performance.

### 5. What is the best recommendation technique you have learned and what type of recommendation technique helps to predict ratings?
The **best recommendation technique** could be **Collaborative Filtering** (user-based or item-based). For predicting ratings, **Matrix Factorization techniques** like **SVD (Singular Value Decomposition)** are effective.

### 6. How can you assess a good logistic model?
A good logistic model can be assessed using metrics such as **accuracy**, **precision**, **recall**, **F1 score**, **ROC-AUC**, and by checking **calibration curves**.

### 7. How do you read the text from an image? Explain.
To read text from an image, use **Optical Character Recognition (OCR)**. Libraries like **Tesseract** or **Google Vision API** can extract and recognize text from images.

### 8. What are all the options to convert speech to text? Explain and name few available tools to implement the same.
Options to convert speech to text include **automatic speech recognition (ASR)** systems. Tools like **Google Speech-to-Text**, **IBM Watson Speech to Text**, **Microsoft Azure Speech**, and **Amazon Transcribe** are available for implementation.

---

## My checklist before going for an SQL round of interview:

- WHERE, AND, OR, NOT, IN
- ORDER BY, ASC, DESC
- IS NULL
- LIMIT
- MIN, MAX, COUNT, AVG, SUM
- LIKE, WILDCARDS
- IN BETWEEN
- INNER JOIN
- LEFT JOIN
- Subqueries (most important)
- UNION
- GROUP BY
- HAVING
- LEFT, RIGHT, MID, CONCAT
- PARTITION BY, OVER
- LEAD, LAG
- RANK, DENSE_RANK, PERCENT_RANK
- ROW_NUMBER, CUME_DIST
- FIRST_VALUE, LAST_VALUE
- AS

---

# Company Name: Tata IQ

## Role: Data Analyst

### 1. Why data science as a career?
Data science is a career that combines **analytical skills**, **statistical knowledge**, and **computational expertise** to extract valuable insights from data, making it **highly impactful and rewarding**.

### 2. What is p-value?
A **p-value** indicates the probability of obtaining test results at least as extreme as the observed results, assuming that the null hypothesis is true. A **low p-value** (< 0.05) suggests that the null hypothesis can be rejected.

### 3. What is a histogram?
A **histogram** is a graphical representation of the distribution of numerical data, showing the frequency of data points within specified ranges (bins).

### 4. What is a confidence interval?
A **confidence interval** is a range of values, derived from sample data, that is likely to contain the true value of an unknown population parameter with a certain level of confidence (e.g., 95%).

### 5. You are a Sr data analyst at a new Online Cab booking Startups. How will you do data collection and leverage the data to give useful insights to the company?
Collect data from **booking transactions**, **customer feedback**, **GPS logs**, and **usage patterns**. Analyze this data to identify **peak usage times**, **popular routes**, **customer preferences**, and **areas for operational improvement**.

### 6. Guestimate: No of cabs booking per day in Ranchi.
Estimate the number of cabs booked per day in Ranchi by considering factors like **population size**, **average number of trips per person**, and **market penetration rate**.

### 7. You are the product head manager at an NBFC that gives secured loans. What factors will you consider for giving a loan?
Consider factors such as **credit score**, **income stability**, **employment history**, **debt-to-income ratio**, and **collateral value**.

### 8. Inventory Database based on that have to do basic pandas/sql query? Joins/merge to get avg sales, its chart?
Use **pandas** or **SQL** to perform joins/merge operations on the inventory database to calculate average sales and create visualizations using libraries like **Matplotlib** or **Seaborn**.

### 9. You have a list of 3 numbers. Return the minimum difference. Can use any Python/SQL.
Write a function in Python or SQL to find and return the minimum difference between any two numbers in the list.

### 10. What is Big Data?
**Big Data** refers to large and complex data sets that are difficult to process using traditional data processing techniques. It involves **volume, velocity, variety**, and **veracity**.

---

## Role: Junior Data Scientist

### 1. Explain the architecture of CNN.
A **Convolutional Neural Network (CNN)** consists of **convolutional layers**, **pooling layers**, **fully connected layers**, and **activation functions**. It is designed to automatically and adaptively learn spatial hierarchies of features from input images.

### 2. If we put a 3×3 filter over a 6×6 image, what will be the size of the output image?
Assuming no padding and a stride of 1, the size of the output image will be **(6-3+1) x (6-3+1) = 4x4**.

### 3. What will you do to reduce overfitting in deep learning models?
To reduce overfitting, you can use **techniques** such as **dropout**, **early stopping**, **data augmentation**, **regularization (L2/L1)**, and **cross-validation**.

### 4. Can you write a program for inverted star pattern in Python?
```python
def inverted_star_pattern(n):
    for i in range(n, 0, -1):
        print('*' * i)

inverted_star_pattern(5)
```


### 5. Write a program to create a dataframe and remove elements from it.
```python
import pandas as pd

# Create DataFrame
data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
df = pd.DataFrame(data)

# Remove elements
df.drop(['A', 'C'], axis=1, inplace=True)
print(df)
```

### 6. I have 2 guns with 6 holes in each, and I load a single bullet in each gun. What is the probability that if I fire the guns simultaneously at least 1 gun will fire (at least means one or more than one)?
The probability that at least one gun will fire is 11/36.



# Company: Mindtree

## Role: Data Scientist

### 1. What is central tendency?
Central tendency measures the center of a data distribution. **Mean**, **median**, and **mode** are common measures.

### 2. Which central tendency method is used if there exists any outliers?
The **median** is preferred when there are outliers because it is **not affected by extreme values**.

### 3. Central limit theorem
The **central limit theorem** states that the sampling distribution of the sample mean approaches a **normal distribution** as the sample size grows, regardless of the population's distribution.

### 4. Chi-Square test
The **Chi-Square test** evaluates whether there is a **significant association** between two categorical variables.

### 5. A/B testing
**A/B testing** compares two versions of a variable to determine which one performs better in terms of a specific metric.

### 6. Difference between Z and t distribution (Linked to A/B testing)
**Z-distribution** is used when the sample size is large and the population variance is known, whereas the **t-distribution** is used for smaller sample sizes and unknown population variance.

### 7. Outlier treatment method
Outliers can be treated by **removing** them, **transforming** the data, or using **robust statistical methods**.

### 8. ANOVA test
The **ANOVA test** checks for significant differences between the means of **three or more groups**.

### 9. Cross validation
**Cross-validation** is a technique to evaluate the model's performance by partitioning the data into **training and testing** sets multiple times.

### 10. How will you work in a machine learning project if there is a huge imbalance in the data?
Use techniques like **resampling**, **SMOTE**, or **adjusting class weights** to address data imbalance.

### 11. Formula of sigmoid function
The formula for the sigmoid function is **1 / (1 + e^(-x))**.

### 12. Can we use sigmoid function in case of multiple classification?
No, for multiple classification, **softmax function** is used instead of the sigmoid function.

### 13. What is Area under the curve?
**Area Under the Curve (AUC)** measures the ability of a model to distinguish between classes. Higher AUC indicates a better model.

### 14. Which metric is used to split a node in Decision Tree?
Metrics like **Gini impurity** and **Information Gain** are used to split nodes in a decision tree.

### 15. What is ensemble learning?
**Ensemble learning** combines multiple models to produce a more **robust and accurate** prediction.

### 16. 3 situation based questions
1. **How would you handle missing data in a dataset?**
   Use methods like **imputation**, **deletion**, or **predictive modeling**.

2. **How would you approach a new dataset?**
   Start with **exploratory data analysis (EDA)** to understand the data distribution, patterns, and anomalies.

3. **How do you ensure your model is not overfitting?**
   Use techniques like **cross-validation**, **regularization**, and **pruning**.

# Company: Genpact

## Role: Data Scientist

### 1. Why do we select validation data other than test data?
Validation data is used to **tune the model** parameters, while test data is used to **evaluate the model's performance**.

### 2. Difference between linear logistic regression?
**Linear regression** predicts continuous outcomes, while **logistic regression** predicts binary outcomes using a logistic function.

### 3. Why do we take such a complex cost function for logistic?
The logistic cost function is complex to handle the **non-linear nature** of the logistic model and ensure **convex optimization**.

### 4. Difference between random forest and decision tree?
A **decision tree** is a single model, while a **random forest** is an ensemble of multiple decision trees to improve accuracy and reduce overfitting.

### 5. How would you decide when to stop splitting the tree?
Stop splitting based on **predefined criteria** such as maximum depth, minimum samples per leaf, or when further splitting does not improve performance.

### 6. Measures of central tendency
**Mean**, **median**, and **mode** are the primary measures of central tendency.

### 7. What is the requirement of k means algorithm?
K-Means requires the number of clusters **k** to be specified before running the algorithm.

### 8. Which clustering technique uses combining of clusters?
**Hierarchical clustering** uses a combination of clusters to create a tree-like structure.

### 9. Which is the oldest probability distribution?
The **Uniform distribution** is considered one of the oldest probability distributions.

### 10. What all values does a random variable can take?
A random variable can take **discrete or continuous** values depending on the type of distribution it follows.

### 11. Types of random variables
Random variables can be **discrete** or **continuous**.

### 12. Normality of residuals
For many statistical models, the residuals should be **normally distributed** to validate the model assumptions.




# Company: Ford

## Role: Data Scientist

### 1. How would you check if the model is suffering from multicollinearity?
To check for multicollinearity, you can use the **Variance Inflation Factor (VIF)**. A VIF value above **10** indicates high multicollinearity, suggesting the need for further investigation.

### 2. What is transfer learning? Steps you would take to perform transfer learning.
Transfer learning is a technique where a **pre-trained model** is used as a starting point for a new task. Steps include: **1)** Select a pre-trained model, **2)** Fine-tune the model with new data, **3)** Evaluate and adjust as needed.

### 3. Why is CNN architecture suitable for image classification? Not an RNN?
**CNNs** are suitable for image classification due to their ability to capture **spatial hierarchies** in images using **convolutional layers**. **RNNs** are designed for **sequential data** and are less effective for capturing spatial features.

### 4. What are the approaches for solving class imbalance problem?
Approaches include **oversampling** the minority class, **undersampling** the majority class, using **synthetic data generation** (e.g., **SMOTE**), and applying **cost-sensitive learning**.

### 5. When sampling, what types of biases can be inflected? How to control the biases?
Biases include **selection bias**, **confirmation bias**, and **survivorship bias**. Control biases by using **random sampling**, ensuring a **representative sample**, and applying **blind analysis**.

### 6. Explain concepts of epoch, batch, iteration in machine learning.
- **Epoch**: One complete pass through the entire training dataset.
- **Batch**: A subset of the training data used to update model weights.
- **Iteration**: One update of the model weights using a single batch.

### 7. What type of performance metrics would you choose to evaluate the different classification models and why?
Metrics include **accuracy**, **precision**, **recall**, **F1 score**, and **AUC-ROC**. Choose based on the problem context, such as **precision** and **recall** for imbalanced datasets.

### 8. What are some of the types of activation functions and specifically when to use them?
- **ReLU**: Commonly used in hidden layers for its computational efficiency.
- **Sigmoid**: Used for binary classification problems.
- **Softmax**: Used in the output layer for multi-class classification.

### 9. What are the conditions that should be satisfied for a time series to be stationary?
A time series is stationary if its **mean**, **variance**, and **autocorrelation** are **constant over time**.

### 10. What is the difference between Batch and Stochastic Gradient Descent?
- **Batch Gradient Descent**: Uses the entire dataset for each update, which is computationally expensive but stable.
- **Stochastic Gradient Descent**: Uses one data point per update, making it faster but noisier.

### 11. What is the difference between K-NN and K-Means clustering?
- **K-NN**: A supervised algorithm used for **classification** based on the nearest neighbors.
- **K-Means**: An unsupervised algorithm used for **clustering** by minimizing within-cluster variance.

---

# Company: Quantiphi

## Role: Machine Learning Engineer

### 1. What happens when neural nets are too small? What happens when they are large enough?
- **Too small**: May lead to **underfitting**, failing to capture the data's complexity.
- **Large enough**: Can lead to **overfitting** if not regularized properly.

### 2. Why do we need pooling layer in CNN? Common pooling methods?
Pooling layers reduce **dimensionality** and **computational cost**. Common methods include **max pooling** and **average pooling**.

### 3. Are ensemble models better than individual models? Why/why not?
Ensemble models are generally better as they **combine multiple models** to reduce **variance** and **bias**, leading to improved **generalization**.

### 4. Use Case: Consider you are working for a pen manufacturing company. How would you help the sales team with leads using data analysis?
I would analyze **sales data**, identify **trends** and **patterns**, segment the customer base, and use **predictive analytics** to generate leads and optimize **targeted marketing** campaigns.

### 5. Assume you were given access to a website's Google Analytics data. In order to increase conversions, how do you perform A/B testing to identify the best page design?
Set up **A/B testing** by creating **variations** of the page design, divide traffic between them, and analyze the results using **conversion metrics** to determine the best performing design.

### 6. How is random forest different from Gradient boosting algorithm, given both are tree-based algorithms?
- **Random Forest**: Uses multiple **independent decision trees** and averages their predictions.
- **Gradient Boosting**: Builds trees **sequentially** where each tree corrects the errors of the previous ones.

### 7. Describe steps involved in creating a neural network?
1. **Define** the network architecture.
2. **Initialize** weights and biases.
3. **Forward propagate** the inputs.
4. **Compute** the loss.
5. **Backward propagate** to update weights.
6. **Repeat** until convergence.

### 8. In brief, how would you perform the task of sentiment analysis?
1. **Preprocess** the text data.
2. **Tokenize** and vectorize the text.
3. **Train** a model (e.g., LSTM, BERT) on labeled sentiment data.
4. **Evaluate** and fine-tune the model.
5. **Deploy** the model for sentiment prediction.

---

# Company: TheMathCompany

## Role: Analyst (Data Science)

### 1. Central limit theorem
The central limit theorem states that the **sampling distribution** of the sample mean approaches a **normal distribution** as the sample size grows, regardless of the population's distribution.

### 2. Hypotheses testing
A statistical method used to decide whether there is enough evidence to reject a **null hypothesis** in favor of an **alternative hypothesis**.

### 3. P value
The p-value measures the **probability** of obtaining results at least as extreme as the observed results, assuming the null hypothesis is true. A **low p-value** (< 0.05) indicates **strong evidence** against the null hypothesis.

### 4. T-test
A t-test is used to determine if there is a **significant difference** between the means of two groups. It can be **one-sample**, **independent two-sample**, or **paired-sample**.

### 5. Assumptions of linear regression
- **Linearity**: The relationship between predictors and the target is linear.
- **Independence**: Observations are independent.
- **Homoscedasticity**: Constant variance of errors.
- **Normality**: Errors are normally distributed.

### 6. Correlation and covariance
- **Correlation**: Measures the **strength and direction** of a linear relationship between two variables.
- **Covariance**: Measures the **extent to which two variables change together**.

### 7. How to identify & treat outliers and missing values?
- **Outliers**: Use **visualization** (e.g., box plots), and treat using **trimming**, **transformation**, or **imputation**.
- **Missing values**: Identify using **descriptive statistics** and treat using **deletion**, **mean/mode imputation**, or **predictive modeling**.

### 8. Explain Box and whisker plot.
A box and whisker plot displays the **distribution** of data through its quartiles. The **box** shows the interquartile range (IQR), and the **whiskers** extend to the minimum and maximum values within 1.5 * IQR from the quartiles.

### 9. Explain any unsupervised learning algorithm.
**K-Means Clustering**: An unsupervised algorithm that partitions data into **K clusters** by minimizing the variance within each cluster. It iteratively assigns data points to the nearest cluster centroid and recalculates centroids.

### 10. Explain Random forest.
A random forest is an **ensemble learning method** that constructs multiple **decision trees** during training and outputs the **mode of the classes** (classification) or **mean prediction** (regression) of individual trees.

### 11. Business and technical questions related to your project.
Be prepared to discuss the **objective**, **methodology**, **results**, and **impact** of your project, along with any **technical challenges** and solutions.

### 12. Explain any scope of improvement in your project.
Identify areas such as **algorithm enhancements**, **feature engineering**, **data quality improvements**, or **performance optimization** that could further improve your project's results.

### 13. Questions based on case studies.
Be ready to analyze and provide solutions to **real-world problems**, focusing on **data analysis**, **modeling approaches**, and **business impact**.

### 14. Write SQL query to find employee with highest salary in each department.
```sql
SELECT department, employee, MAX(salary) 
FROM employees 
GROUP BY department;
```
### 15. Write SQL query to find unique email domain name & their respective count.
```sql
SELECT SUBSTRING_INDEX(email, '@', -1) AS domain, COUNT(*) 
FROM employees 
GROUP BY domain;
```
### 16. Solve question (15) using Python.
```python
import pandas as pd

# Assuming df is the DataFrame containing employee data
df['domain'] = df['email'].apply(lambda x: x.split('@')[1])
domain_counts = df['domain'].value_counts().reset_index()
domain_counts.columns = ['domain', 'count']
print(domain_counts)
```

### Rounds

1. Technical Test (Python, SQL, Statistics) (Coding+MCQ) (90 min).
2. Telephonic interview (10 min).
3. Technical interview (45 min).
4. Fitment interview (25 min).
5. HR interview (30 min).


# Company: Cognizant

## Role: Data Scientist

### 1. SQL question on inner join and cross join
**Inner Join** returns only the rows where there is a match in both tables. **Cross Join** returns the Cartesian product of the two tables, combining all rows.

### 2. SQL question on group-by
**GROUP BY** is used to aggregate data across multiple records by one or more columns. It is often used with aggregate functions like **SUM**, **AVG**, **COUNT**, etc.

### 3. Case study question on customer optimization of records for different marketing promotional offers
Customer optimization involves analyzing customer data to tailor marketing promotions for maximum effectiveness, utilizing segmentation, and predictive modeling techniques.

### 4. Tuple and list
**Tuple**: Immutable, ordered collection of elements. **List**: Mutable, ordered collection of elements. Use tuples for fixed data and lists for dynamic data.

### 5. Linear regression
**Linear Regression** models the relationship between a dependent variable and one or more independent variables using a linear equation.

### 6. Logistic regression steps and process
**Logistic Regression** involves: 1) Data Preprocessing, 2) Splitting the data, 3) Fitting the model, 4) Making predictions, and 5) Evaluating performance using metrics like **AUC-ROC**.

### 7. Tell me about your passion for data science? Or What brought you to this field?
I have a passion for **data-driven decision-making** and **solving complex problems** using analytical skills. The ability to derive insights and predictive models from data fascinates me.

### 8. What is the most common problems you face whilst working on data science projects?
Common problems include **data quality issues**, **handling missing values**, and **overfitting** models. Ensuring data integrity and selecting the right algorithms are crucial steps.

### 9. Describe the steps to take to forecast quarterly sales trends. What specific models are most appropriate in this case?
Steps: 1) Collect data, 2) Data preprocessing, 3) Exploratory Data Analysis, 4) Model selection (e.g., **ARIMA**, **Exponential Smoothing**, **LSTM**), 5) Model training and validation, 6) Forecasting and evaluation.

### 10. What is the difference between gradient and slope, differentiation and integration?
**Gradient** measures the rate of change in multiple dimensions, while **slope** is in one dimension. **Differentiation** finds the rate of change; **Integration** calculates the area under the curve.

### 11. When to use deep learning instead of machine learning. Advantages, Disadvantages of using deep learning?
Use **deep learning** for large datasets with complex patterns (e.g., image and speech recognition). Advantages: **High accuracy**, **automatic feature extraction**. Disadvantages: **Requires large datasets**, **computationally expensive**.

### 12. What are vanishing and exploding gradients in neural networks?
**Vanishing gradients** occur when gradients become too small, hindering learning. **Exploding gradients** occur when gradients become too large, causing instability. Both affect training efficiency.

# Company: Husqvarna Group

## Role: Data Scientist

### 1. Telecom Customer Churn Prediction. Explain the project end to end?
The project involves: 1) Data collection, 2) Data preprocessing (cleaning, encoding), 3) Feature selection, 4) Model training (e.g., **Logistic Regression**, **Random Forest**), 5) Evaluation (using metrics like **accuracy**, **F1 score**), and 6) Deployment.

### 2. Data Pre-Processing Steps used.
Steps include: 1) Handling missing values, 2) Encoding categorical variables, 3) Normalizing/standardizing features, and 4) Splitting data into training and testing sets.

### 3. Sales forecasting how is it done using Statistical vs DL models - Efficiency.
**Statistical models** (e.g., ARIMA) are simpler and faster for short-term trends. **Deep Learning models** (e.g., LSTM) handle complex, non-linear patterns and long-term dependencies but require more data and computational power.

### 4. Logistic Regression - How much percent of Customer has churned and how much have not churned?
The model predicts the **probability of churn**. For evaluation, compare the predicted probabilities to the actual churn rates and calculate metrics like **precision**, **recall**, and **accuracy**.

### 5. What are the Evaluation Metric parameters for testing Logistic Regression?
Common metrics include **Accuracy**, **Precision**, **Recall**, **F1 Score**, and **AUC-ROC**.

### 6. What packages in Python can be used for ML? Why do we prefer one over another?
Packages include **scikit-learn** (easy-to-use), **TensorFlow** (deep learning), **Keras** (user-friendly DL), **Pandas** (data manipulation), **NumPy** (numerical operations). Preference depends on the task complexity and ease of use.

### 7. Numpy vs Pandas basic difference.
**NumPy** is for numerical operations and array handling. **Pandas** is built on NumPy, providing data structures and functions for data manipulation and analysis.

### 8. Feature on which this Imputation was done, and which method did we use there?
Imputation is done on features with missing values. Methods include **mean/median/mode imputation**, **KNN imputation**, and **iterative imputation** based on the data context.

### 9. Tuple vs Dictionary. Where do we use them?
**Tuple**: Ordered, immutable collection. Use for fixed data sequences. **Dictionary**: Unordered, mutable collection of key-value pairs. Use for fast lookups and data mapping.

### 10. What is NER - Named Entity Recognition?
**NER** is a process in NLP to identify and classify named entities (e.g., **person names**, **locations**, **organizations**) in text into predefined categories.

# Company: Deloitte

## Role: Data Scientist

### 1. Conditional Probability
Conditional probability is the likelihood of an event occurring given that another event has already occurred, denoted as **P(A|B)**.

### 2. Can Linear Regression be used for Classification? If Yes, why if No why?
**No**, linear regression is for predicting continuous values. For classification, **logistic regression** is used as it predicts probabilities for discrete classes.

### 3. Hypothesis Testing. Null and Alternate hypothesis
**Null hypothesis (H0)** assumes no effect or relationship. **Alternate hypothesis (H1)** assumes an effect or relationship exists. Hypothesis testing evaluates the evidence against H0.

### 4. Derivation of Formula for Linear and logistic Regression
Linear Regression: **y = β0 + β1x + ε**. Logistic Regression: **log(p/(1-p)) = β0 + β1x**, where p is the probability of the outcome.

### 5. Why use Decision Trees?
**Decision Trees** are easy to interpret, handle both numerical and categorical data, and capture non-linear relationships. However, they can overfit the data.

### 6. PCA Advantages and Disadvantages?
**PCA Advantages**: Reduces dimensionality, removes multicollinearity, improves model performance. **Disadvantages**: Can lose interpretability, sensitive to outliers.

### 7. What is Naive Bayes Theorem? Multinomial, Bernoulli, Gaussian Naive Bayes.
**Naive Bayes** applies Bayes' theorem with the assumption of feature independence. **Multinomial**: For discrete data. **Bernoulli**: For binary data. **Gaussian**: For continuous data.

### 8. Central Limit Theorem?
The **Central Limit Theorem** states that the distribution of the sample mean approximates a normal distribution as the sample size becomes large, regardless of the population's distribution.

### 9. Scenario based question on when to use which ML model?
Choice depends on data characteristics, problem type (classification/regression), and model interpretability. For example, use **Random Forest** for flexibility and **SVM** for high-dimensional data.

### 10. Over Sampling and Under Sampling
**Over Sampling** increases the number of minority class samples, while **Under Sampling** reduces the number of majority class samples to address class imbalance.

### 11. Over Fitting and Under Fitting
**Overfitting** occurs when the model learns noise and details in the training data, failing to generalize. **Underfitting** happens when the model is too simple to capture the underlying pattern.

### 12. Core Concepts behind Each ML model mentioned in my Resume.
Understand the theoretical foundations, assumptions, and applications of each model, including their strengths and weaknesses.

### 13. Genie Index Vs Entropy
**Gini Index** measures impurity used in decision trees. **Entropy** measures the randomness in the information content. Both are used for feature selection in decision trees.

### 14. How to deal with imbalance data in classification modelling?
Techniques include **resampling** (over-sampling, under-sampling), **SMOTE**, **cost-sensitive learning**, and using performance metrics like **Precision-Recall** to evaluate models.




# Company: Wipro

## Role: Data Scientist

### 1. What is a Python Package, and have you created your own Python Package?
A Python package is a **collection of modules** that allows for code reuse and organization. Yes, I have created my own Python package to **simplify complex tasks** and **promote reusability** in my projects.

### 2. Explain about Time series models you have used?
I have used **ARIMA**, **SARIMA**, and **Prophet** for time series forecasting. These models help in capturing **seasonal trends**, **irregular patterns**, and **forecasting future values**.

### 3. SQL Questions - Group by Top 2 Salaries for Employees - use Row num and Partition
```sql
SELECT *
FROM (
    SELECT employee_id, salary,
           ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) as rn
    FROM employees
) 
WHERE rn <= 2;
```

### 4. Pandas find Numeric and Categorical Columns. For Numeric columns in Data frame, find the mean of the entire column and add that mean value to each row of those numeric columns.
```python
import pandas as pd

# Assuming df is the DataFrame
numeric_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(exclude=['number']).columns

for col in numeric_cols:
    mean_val = df[col].mean()
    df[col] = df[col] + mean_val
```

### 5. What is Gradient Descent? What is Learning Rate and Why we need to reduce or increase? Why Global minimum is reached and Why it doesn’t improve when increasing the LR after that point?
Gradient Descent is an **optimization algorithm** used to minimize the loss function. The **learning rate** controls the **step size** during the optimization. A smaller learning rate leads to **more precise convergence**, while a larger one may cause overshooting. The global minimum is reached when the loss function **cannot be reduced further**, and increasing the learning rate may cause it to **diverge**.

### 6. Two Logistic Regression Models - Which one will you choose - One is trained on 70% and other on 80% data. Accuracy is almost same.
Choose the model **trained on 80%** data as it is likely to have **better generalization** and **more robust performance**.

### 7. What is Log-Loss and ROC-AUC?
Log-Loss measures the **performance of a classification model** where the output is a probability value. ROC-AUC is the area under the ROC curve, which plots the true positive rate against the false positive rate. Both metrics are **used to evaluate classification models**.

### 8. Do you know to use Amazon SageMaker for MLOPS?
Yes, I am familiar with using Amazon SageMaker for MLOps to build, train, and deploy machine learning models at scale.

### 9. Explain your Projects end to end (15-20mins).
In my projects, I start with problem identification, data collection, and exploratory data analysis. I then perform feature engineering, model selection, and training. Finally, I evaluate the model, deploy it, and monitor its performance for continuous improvement.



# Company: Infosys

## Role: Data Scientist

### 1. Measures of central tendency
Measures of central tendency include the **mean**, **median**, and **mode**, which summarize a dataset with a single value representing the center of its distribution.

### 2. What is the requirement of k means algorithm
The **K-Means algorithm** requires the user to specify the number of clusters **k** and is based on the **Euclidean distance** between data points.

### 3. Which clustering technique uses combining of clusters
**Hierarchical clustering** uses the combining (agglomerative) or dividing (divisive) of clusters to form a hierarchy of clusters.

### 4. Which is the oldest probability distribution
The **Binomial distribution** is one of the oldest probability distributions, describing the number of successes in a fixed number of independent Bernoulli trials.

### 5. What all values does a random variable can take
A random variable can take **discrete** values (specific, countable outcomes) or **continuous** values (any value within a range).

### 6. Types of random variables
There are two types of random variables: **Discrete random variables** and **Continuous random variables**.

### 7. Normality of residuals
The normality of residuals is important for **linear regression** assumptions, indicating that residuals should follow a **normal distribution**.

### 8. Probability questions
Probability questions involve calculating the likelihood of events using **probability rules** and **theorems**.

### 9. Sensitivity and specificity etc.
**Sensitivity** measures the true positive rate, while **specificity** measures the true negative rate, both important for evaluating classification models.

### 10. Explain bias - variance trade off. How does this affect the model?
The **bias-variance trade-off** refers to the balance between **underfitting** (high bias) and **overfitting** (high variance). A good model minimizes both to improve **predictive accuracy**.

### 11. What is multicollinearity? How to identify and remove it.
Multicollinearity occurs when **predictor variables** are highly correlated, causing issues in regression models. It can be identified using **Variance Inflation Factor (VIF)** and removed by **dropping variables** or using **principal component analysis**.

# Company: Tiger Analytics

## Role: Data Scientist

### 1. What are the projects done by you.
Describe the **key projects** you have worked on, highlighting your **role**, **technologies used**, and **impact**.

### 2. Suppose there is a client who wants to know if giving discounts is beneficial or not. How would you approach this problem?
To determine if discounts are beneficial, analyze **sales data** before and after discounts, consider **customer segmentation**, and perform **A/B testing**.

### 3. The same client wants to know how much discount he should give in the next month for maximum profits.
To find the optimal discount, use **predictive modeling** and **optimization techniques** to balance **sales volume** and **profit margins**.

### 4. Can you have a modeling approach to say in the last year what mistakes the client did in giving discounts?
Analyze past discount strategies using **regression analysis** and **time-series analysis** to identify **patterns** and **improvement areas**.

### 5. What feature engineering techniques you used in past projects.
Discuss techniques like **handling missing values**, **encoding categorical variables**, **scaling features**, and **creating interaction terms**.

### 6. What models you used and selected the final model.
Mention the models you have used, such as **linear regression**, **decision trees**, **random forests**, and the criteria for **selecting the final model**, like **accuracy**, **precision**, **recall**, and **cross-validation scores**.

# Company: Genpact

## Role: Data Scientist

### 1. What makes you feel that you would be suitable for this role, since you come from a different background?
Highlight **transferable skills**, **relevant experiences**, and your **passion for data science** that make you suitable for the role.

### 2. What is an imbalanced dataset?
An imbalanced dataset has a **disproportionate ratio** of classes, often requiring techniques like **resampling** or **adjusted metrics** for proper handling.

### 3. What are the factors you will consider in order to predict the population of a city in the future?
Consider factors such as **birth rates**, **death rates**, **migration patterns**, **economic conditions**, and **historical trends**.

### 4. Basic statistics questions?
Be prepared to answer questions on **mean**, **median**, **mode**, **variance**, **standard deviation**, and **probability distributions**.

### 5. What are the approaches for treating the missing values?
Treat missing values using techniques like **mean/mode imputation**, **regression imputation**, **k-nearest neighbors imputation**, or **removing missing data**.

### 6. Evaluation metrics for Classification?
Common metrics include **accuracy**, **precision**, **recall**, **F1 score**, **ROC-AUC**, and **confusion matrix**.

### 7. Bagging vs Boosting with examples
**Bagging** involves building multiple models in parallel (e.g., Random Forest), while **Boosting** builds models sequentially to correct errors (e.g., Gradient Boosting).

### 8. Handling of imbalanced datasets
Handle imbalanced datasets using techniques like **SMOTE**, **undersampling**, **oversampling**, and using **balanced accuracy** as a metric.

### 9. What are your career aspirations?
Discuss your **long-term career goals**, how you plan to achieve them, and how the role aligns with your **aspirations**.

### 10. What's the graph of y = |x|-2
The graph of \( y = |x| - 2 \) is a **V-shaped graph** shifted **downwards by 2 units**.

### 11. Estimate on number of petrol cars in Delhi
Use **historical data**, **registration records**, and **statistical models** to estimate the number of petrol cars in Delhi.

### 12. Case study on opening a retail store
Consider **location analysis**, **target market**, **competition**, **cost analysis**, and **potential revenue** when conducting a case study on opening a retail store.

### 13. Order of execution of SQL
The order of execution in SQL is: **FROM**, **WHERE**, **GROUP BY**, **HAVING**, **SELECT**, **ORDER BY**, **LIMIT**.




# Company: Ericsson

## Role: Data Scientist

### Round No: 1st Round

### 1. How to reverse a linked list
To reverse a linked list, you can **iterate through the list** and reverse the pointers of each node to point to the previous node until all pointers are reversed.

### 2. Give a logistic regression model in production, how would you find out the coefficients of different input features.
In production, you can **retrieve the coefficients** of a logistic regression model by accessing the model's **attributes** (e.g., `.coef_` in scikit-learn) after the model is trained.

### 3. What is the p-value in OLS regression
The **p-value** in OLS regression indicates the **probability** that the observed data would occur by chance if the null hypothesis is true. **Low p-values (< 0.05)** suggest that the corresponding feature has a **significant** impact on the target variable.

### 4. What's the reason for high bias or variance
**High bias** is often caused by **underfitting**, where the model is too simple. **High variance** is caused by **overfitting**, where the model is too complex and captures noise in the data.

### 5. Which models are generally high biased or high variance
**Linear models** (e.g., linear regression) tend to have **high bias**, while **complex models** (e.g., decision trees, neural networks) tend to have **high variance**.

### 6. Write code to find the 8 highest value in the DataFrame
```python
import pandas as pd

# Assuming df is your DataFrame
top_8_values = df.unstack().nlargest(8)
print(top_8_values)
```

### 7. What's the difference between array and list
**Arrays are collections of elements of the same data type**, providing faster computations. **Lists are collections of elements of different data types** and offer more flexibility.

### 8. What's the difference between Gradient boosting and Xgboost
Gradient Boosting is a general boosting method that builds models sequentially. XGBoost is an optimized and high-performance implementation of gradient boosting with additional features like regularization, parallel processing, and handling missing values.

### 9. Is XOR data linearly separable
No, **XOR data is not linearly separable** because it cannot be divided by a single linear boundary.

### 10. How do we classify XOR data using logistic regression
To classify XOR data using logistic regression, you need to add polynomial features (e.g., interaction terms) to make the data linearly separable.

### 11. Some questions from my previous projects
Be prepared to discuss specific details about your previous projects, such as challenges faced, technologies used, and outcomes achieved.

### 12. Given a sand timer of 4 and 7 mins how would you calculate 10 mins duration.
Start both timers. When the 4-minute timer runs out, turn it over (4 minutes elapsed). When the 7-minute timer runs out, turn it over (7 minutes elapsed, 3 minutes left on the 4-minute timer). When the 4-minute timer runs out again, turn it over (8 minutes elapsed). When the 3 minutes left on the 7-minute timer runs out, 10 minutes have elapsed.

### 13. What's the angle between hour and minute hand in clock at 3:15
At 3:15, the minute hand is at the 3 (90 degrees) and the hour hand is a quarter of the way between 3 and 4. Each hour is 30 degrees, so the hour hand is at 3 * 30 + 7.5 = 97.5 degrees. The angle between them is 97.5 - 90 = 7.5 degrees.



# Company: FISERVE

## Role: Data Scientist

### 1. How would you check if the model is suffering from multi Collinearity?
To check for multicollinearity, you can use the **Variance Inflation Factor (VIF)**. A VIF value greater than 10 indicates high multicollinearity.

### 2. What is transfer learning? Steps you would take to perform transfer learning.
Transfer learning involves using a **pre-trained model** on a new, related task. Steps include:
- **Select a pre-trained model** on a large dataset.
- **Remove the final layer(s)**.
- **Add new layers** specific to the new task.
- **Train the model** on the new dataset.

### 3. Why is CNN architecture suitable for image classification? Not an RNN?
**CNNs** are suitable for image classification because they can **capture spatial hierarchies** in images through convolutional layers. **RNNs** are designed for sequential data and do not efficiently handle spatial information in images.

### 4. What are the approaches for solving class imbalance problem?
Approaches include **resampling techniques** (oversampling, undersampling), using **class weights**, and implementing **synthetic data generation methods** like SMOTE.

### 5. When sampling what types of biases can be inflected? How to control the biases?
Types of biases include **selection bias**, **response bias**, and **measurement bias**. Control them by ensuring **random sampling**, **blinding**, and using **validated measurement tools**.

### 6. Explain concepts of epoch, batch, iteration in machine learning.
- **Epoch**: One full pass through the entire dataset.
- **Batch**: A subset of the dataset used to train the model.
- **Iteration**: One update of the model's parameters, typically equal to the number of batches per epoch.

### 7. What type of performance metrics would you choose to evaluate the different classification models and why?
Metrics like **accuracy**, **precision**, **recall**, **F1-score**, and **ROC-AUC** are used to evaluate classification models to understand different aspects of their performance, especially in cases of imbalanced datasets.

### 8. What are some of the types of activation functions and specifically when to use them?
- **ReLU**: For most hidden layers due to efficient gradient propagation.
- **Sigmoid**: For binary classification outputs.
- **Softmax**: For multi-class classification outputs.
- **Tanh**: When negative values are needed.

### 9. What is the difference between Batch and Stochastic Gradient Descent?
**Batch Gradient Descent** updates weights after computing the gradient of the entire dataset, while **Stochastic Gradient Descent (SGD)** updates weights after computing the gradient of a single example, making SGD faster but noisier.

### 10. What is difference between K-NN and K-Means clustering?
**K-NN** is a **supervised learning** algorithm used for classification, while **K-Means** is an **unsupervised learning** algorithm used for clustering data points into groups.

### 11. How to handle missing data? What imputation techniques can be used?
Missing data can be handled by **deleting rows/columns**, **mean/mode/median imputation**, **KNN imputation**, or using advanced methods like **multiple imputation**.

---

# Company: Landmark Group

## Role: Data Scientist

### 1. Use Case - Consider you are working for pen manufacturing company. How would you help sales team with leads using Data analysis?
Using data analysis, identify **patterns and trends** in sales data, **segment customers**, and provide **targeted marketing strategies** to generate leads and improve sales.

### 2. Assume you were given access to a website google analytics data. In order to increase conversions, how do you perform A/B testing to identify best page design?
Perform A/B testing by creating two versions of a page (A and B), **randomly directing visitors** to each version, and comparing **conversion rates** to determine the better design.

### 3. How is random forest different from Gradient boosting algorithm, given both are tree-based algorithm?
**Random Forest** uses **bagging** to create multiple decision trees and averages their results, while **Gradient Boosting** builds trees **sequentially** to correct errors made by previous trees, focusing on **reducing bias**.

### 4. Describe steps involved in creating a neural network?
- **Define the architecture** (number of layers, nodes per layer).
- **Initialize weights and biases**.
- **Forward propagate** input data through the network.
- **Compute loss** using a loss function.
- **Backpropagate** to adjust weights.
- **Iterate** until the model converges.

### 5. LSTM solves the vanishing gradient problem, that RNN primarily have. How?
LSTM networks use **gates** (input, output, forget) to control information flow, **preserving gradients** and effectively **learning long-term dependencies**.

### 6. In brief, how would you perform the task of sentiment analysis?
Perform sentiment analysis by:
- **Preprocessing text** (tokenization, stemming, stop word removal).
- **Feature extraction** (TF-IDF, word embeddings).
- **Train a classifier** (e.g., SVM, LSTM).
- **Evaluate model** performance using metrics like accuracy and F1-score.

---

# Company: Axtria

### 1. RNN, NN and CNN difference.
**RNN**: Processes sequential data and maintains memory.  
**NN**: Basic neural network, suitable for tabular data.  
**CNN**: Specialized for image data with convolutional layers to capture spatial features.

### 2. Supervised, unsupervised and reinforcement learning with their algo example.
- **Supervised**: Labeled data (e.g., Linear Regression, SVM).
- **Unsupervised**: Unlabeled data (e.g., K-Means, PCA).
- **Reinforcement**: Learning via rewards (e.g., Q-Learning).

### 3. Difference between AI, ML and DL
- **AI**: Broad field aiming to create intelligent machines.
- **ML**: Subfield of AI focused on learning from data.
- **DL**: Subfield of ML using deep neural networks.

### 4. How do you do dimensionality reduction?
Using techniques like **PCA**, **t-SNE**, or **Autoencoders** to reduce the number of features while preserving important information.

### 5. What is Multicollinearity?
Multicollinearity occurs when **independent variables** in a regression model are **highly correlated**, making coefficient estimates unstable.

### 6. Parameters of random forest
Important parameters include **number of trees**, **maximum depth**, **minimum samples split**, and **maximum features**.

### 7. Parameters of deep learning algos
Include **learning rate**, **number of epochs**, **batch size**, **number of layers**, and **number of neurons per layer**.

### 8. Different feature selection methods
Methods include **filter methods** (e.g., Chi-Square), **wrapper methods** (e.g., RFE), and **embedded methods** (e.g., LASSO).

### 9. Confusion matrix
A table used to evaluate the performance of a classification model, showing **true positives, false positives, true negatives, and false negatives**.

---

# Company: Latentview Analytics

## Role: Data Scientist

### 1. What is mean and median?
- **Mean**: Average of a dataset.
- **Median**: Middle value in a dataset.

### 2. Difference between normal and Gaussian distribution
**Normal distribution** is also known as **Gaussian distribution**; both are synonymous.

### 3. What is central limit theorem?
The **central limit theorem** states that the distribution of the sample mean approaches a normal distribution as the sample size increases.

### 4. What is null hypothesis?
The **null hypothesis** is a statement that there is **no effect or no difference**, used as a starting point for statistical testing.

### 5. What is confidence interval?
A **confidence interval** is a range of values that is likely to contain the true population parameter with a certain level of confidence (e.g., 95%).

### 6. What is covariance and correlation and how will you interpret it?
- **Covariance**: Measures the joint variability of two variables.
- **Correlation**: Standardized measure of the relationship between two variables, ranging from -1 to 1.

### 7. How will you find out the outliers in the dataset and is it always necessary to remove outliers?
Outliers can be detected using **box plots, z-scores, or IQR method**. It's not always necessary to remove them; it depends on the context.

### 8. Explain about Machine Learning
**Machine Learning** involves creating algorithms that allow computers to learn from and make predictions based on data.

### 9. Explain the algorithm of your choice
Explain the working and application of an algorithm such as **Random Forest**, **SVM**, or **Neural Networks**.

### 10. Different methods of missing values imputation
Methods include **mean/mode/median imputation**, **KNN imputation**, and **multiple imputation**.

### 11. Explain me your ml project
Describe a machine learning project, including **problem statement**, **data preprocessing**, **model selection**, **training**, and **evaluation**.

### 12. How did you handle imbalance dataset?
Handled using **resampling techniques**, **class weights**, or **synthetic data generation methods** like SMOTE.

### 13. What is stratified sampling?
**Stratified sampling** involves dividing the population into subgroups (strata) and sampling from each subgroup proportionally.

### 14. Difference between standard scalar and normal scalar
**Standard scaler** standardizes features by removing the mean and scaling to unit variance, while **normal scaler** typically refers to min-max scaling.

### 15. Different type of visualization in DL project
Visualizations include **loss/accuracy plots**, **confusion matrices**, **feature maps**, and **activation histograms**.

### 16. What architecture have you used
Describe the architecture used in a DL project, such as **CNN**, **RNN**, or **Transformer**.

### 17. Why have you not used RNN in your NLP project
RNNs may not be used due to issues like **vanishing gradients** or the preference for models like **Transformers** for their better performance.

### 18. Why we don't prefer CNN in NLP based project
CNNs are not preferred for NLP because they are better suited for spatial data, whereas **RNNs** or **Transformers** handle sequential data more effectively.

### 19. What is exploding gradient and vanishing gradient and how to rectify it
- **Exploding Gradient**: Gradients grow too large, use **gradient clipping** to rectify.
- **Vanishing Gradient**: Gradients become too small, use **ReLU activation** or **LSTM/GRU**.

### 20. Difference between LSTM and GRU
Both are RNN variants, but **GRU** has a simpler architecture with fewer gates, making it computationally efficient compared to **LSTM**.

### 21. What is precision and recall
- **Precision**: The ratio of true positives to the sum of true and false positives.
- **Recall**: The ratio of true positives to the sum of true positives and false negatives.

### 22. What is AUC metric
**AUC (Area Under the Curve)** measures the ability of a classifier to distinguish between classes. It is used with **ROC curves**.

### 23. What if your precision and recall are same
If precision and recall are the same, it indicates a balanced performance in terms of false positives and false negatives.

### 24. What is Bias Variance Trade Off?
The **bias-variance trade-off** refers to the balance between the error due to bias (error from assumptions) and variance (error from sensitivity to data). **Minimizing both** is crucial for good model performance.

---

# Company: Bridgei2i

## Role: Senior Analytics Consultant

### 1. What is the difference between Cluster and Systematic Sampling?
**Cluster Sampling** involves dividing the population into clusters and randomly sampling clusters. **Systematic Sampling** involves selecting every k-th element from a list.

### 2. Differentiate between a multi-label classification problem and a multi-class classification problem.
- **Multi-label**: Each instance can belong to **multiple classes**.
- **Multi-class**: Each instance belongs to **one of many classes**.

### 3. How can you iterate over a list and also retrieve element indices at the same time?
Use the **enumerate()** function in Python to iterate over a list and retrieve indices.

### 4. What is Regularization and what kind of problems does regularization solve?
**Regularization** adds a penalty term to the loss function to **prevent overfitting** by discouraging complex models.

### 5. If the training loss of your model is high and almost equal to the validation loss, what does it mean? What should you do?
This indicates **underfitting**. Increase model complexity, add more features, or train for more epochs.

### 6. Explain evaluation protocols for testing your models? Compare hold-out vs k-fold cross-validation vs iterated k-fold cross-validation methods of testing.
- **Hold-out**: Splitting data into training and test sets.
- **K-fold cross-validation**: Splitting data into k subsets and training/testing k times.
- **Iterated k-fold cross-validation**: Multiple rounds of k-fold cross-validation to reduce variance in the estimate.

### 7. Can you cite some examples where a false positive is more important than a false negative?
In **spam detection**, a false positive (legitimate email marked as spam) is more critical. In **medical tests** (e.g., cancer screening), missing a true case (false negative) can be more serious.

### 8. What is the advantage of performing dimensionality reduction before fitting an SVM?
Dimensionality reduction can **improve SVM performance** by reducing noise and computational complexity, making the model more efficient.

### 9. How will you find the correlation between a categorical variable and a continuous variable?
Use statistical tests like the **point biserial correlation coefficient** or **ANOVA**.

### 10. How will you calculate the accuracy of a model using a confusion matrix?
Accuracy = (True Positives + True Negatives) / (Total number of observations).

### 11. You are given a dataset with 1500 observations and 15 features. How many observations will you select in each decision tree in a random forest?
Typically, **sqrt(n_features)** for classification or **log2(n_features)**, adjusted based on performance.

### 12. Given that you let the models run long enough, will all gradient descent algorithms lead to the same model when working with Logistic or Linear regression problems?
Yes, they should converge to the same model if they run long enough and the learning rate is appropriate.

### 13. What do you understand by statistical power of sensitivity and how do you calculate it?
Statistical power (sensitivity) is the probability of correctly rejecting a false null hypothesis. It can be calculated as **1 - β** (where β is the Type II error rate).

### 14. What is pruning, entropy and information gain in decision tree algorithm?
- **Pruning**: Removing parts of the tree to prevent overfitting.
- **Entropy**: Measure of impurity or disorder.
- **Information Gain**: Reduction in entropy, used to decide splits.

### 15. What are the types of biases that can occur during sampling?
Biases include **selection bias**, **response bias**, **sampling bias**, and **measurement bias**.




# Company: Prodapt Solutions

## Role: Data Scientist

### 1. Telecom Customer Churn Prediction. Explain the project end to end?
In a **Telecom Customer Churn Prediction** project, we start with **data collection** from various sources such as call logs, billing information, and customer demographics. Next, we perform **data pre-processing** to clean and transform the data. We then **select features** and **train models** (e.g., logistic regression, decision trees) to predict churn. Finally, we **evaluate the model** using metrics like accuracy, precision, and recall, and deploy it to predict future churn.

### 2. Data Pre-Processing Steps used.
Data pre-processing involves steps like **data cleaning** (handling missing values, removing duplicates), **data transformation** (normalization, encoding categorical variables), and **feature selection** to improve model performance.

### 3. Sales forecasting how is it done using Statistical vs DL models - Efficiency.
**Statistical models** like ARIMA are simpler and faster for small datasets, while **Deep Learning models** (e.g., LSTM) handle complex patterns and larger datasets more efficiently but require more computational resources.

### 4. Logistic Regression - How much percent of Customer has churned and how much have not churned?
Logistic regression provides the **probability** of churn for each customer. By setting a threshold (e.g., 0.5), we can classify customers and calculate the **percentage of churn** vs. **non-churn**.

### 5. What are the Evaluation Metric parameters for testing Logistic Regression?
Evaluation metrics for logistic regression include **accuracy**, **precision**, **recall**, **F1 score**, and **ROC-AUC**.

### 6. What packages in Python can be used for ML? Why do we prefer one over another?
Common Python packages for ML include **scikit-learn**, **TensorFlow**, and **PyTorch**. Scikit-learn is preferred for classical ML models due to its simplicity, while TensorFlow and PyTorch are preferred for deep learning due to their flexibility and performance.

### 7. Numpy vs Pandas basic difference.
**NumPy** is used for numerical operations on large arrays and matrices, while **Pandas** provides data structures (Series and DataFrame) for data manipulation and analysis.

### 8. Feature on which this Imputation was done, and which method did we use there?
Imputation can be done on features with missing values using methods like **mean**, **median**, or **mode** imputation, or more advanced methods like **K-Nearest Neighbors (KNN)** imputation.

### 9. Tuple vs Dictionary. Where do we use them?
**Tuples** are immutable and used for storing ordered collections of items, while **dictionaries** are mutable and used for storing key-value pairs for fast lookups.

### 10. What is NER - Named Entity Recognition?
**Named Entity Recognition (NER)** is an NLP technique used to identify and classify entities (e.g., names, dates, locations) in text into predefined categories.

# Company: Landmark Group

## Role: Data Scientist

### 1. SQL question on inner join and cross join
**Inner Join** retrieves records with matching values in both tables, while **Cross Join** returns the Cartesian product of the two tables.

### 2. SQL question on group-by
The **GROUP BY** clause groups rows sharing a property so that an aggregate function can be applied to each group.

### 3. Case study question on customer optimization of records for different marketing promotional offers
Optimization involves analyzing customer data to segment customers and tailor promotional offers using techniques like **RFM analysis** and **predictive modeling**.

### 4. Tuple and list
**Tuples** are immutable, ordered collections, while **lists** are mutable, ordered collections. Use tuples for fixed data and lists for dynamic data.

### 5. Linear regression
**Linear regression** models the relationship between a dependent variable and one or more independent variables using a linear equation.

### 6. Logistic regression steps and process
Steps include **data pre-processing**, **feature selection**, **model training**, **evaluation**, and **deployment**.

### 7. Tell me about your passion for data science? Or What brought you to this field?
My passion for data science stems from a desire to **solve complex problems** and **uncover insights** from data, driving decision-making and innovation.

### 8. What is the most common problems you face whilst working on data science projects?
Common problems include **data quality issues**, **feature selection**, and **model overfitting**.

### 9. Describe the steps to take to forecast quarterly sales trends. What specific models are most appropriate in this case?
Steps include **data collection**, **pre-processing**, **model selection** (e.g., ARIMA, LSTM), **model training**, and **evaluation**. ARIMA is good for time series, while LSTM is suitable for complex patterns.

### 10. What is the difference between gradient and slope, differentiation and integration?
**Gradient** is the multi-variable generalization of slope. **Differentiation** finds the rate of change, while **integration** finds the total accumulated change.

### 11. When to use deep learning instead of machine learning. Advantages, Disadvantages of using deep learning?
Use **deep learning** for large datasets and complex patterns. Advantages include high accuracy and feature learning; disadvantages include high computational cost and data requirements.

### 12. What are vanishing and exploding gradients in neural networks?
**Vanishing gradients** occur when gradients become too small, hindering training. **Exploding gradients** happen when gradients become too large, causing instability.

### 13. What happens when neural nets are too small? What happens when they are large enough?
**Small neural nets** may underfit, missing patterns. **Large neural nets** may overfit, capturing noise instead of the signal.

### 14. Why do we need pooling layer in CNN? Common pooling methods?
**Pooling layers** reduce spatial dimensions and computational load. Common methods include **max pooling** and **average pooling**.

### 15. Are ensemble models better than individual models? Why/why not?
**Ensemble models** are often better due to **combining multiple models** to reduce variance and improve robustness. However, they can be more complex and computationally intensive.

# Company: Mindtree

## Role: Data Scientist

### 1. What is central tendency?
**Central tendency** measures the center of a data distribution, including **mean**, **median**, and **mode**.

### 2. Which central tendency method is used If there exists any outliers?
The **median** is preferred because it is less affected by outliers.

### 3. Central limit theorem
The **central limit theorem** states that the distribution of the sample mean approaches a normal distribution as the sample size grows, regardless of the population's distribution.

### 4. Chi-Square test
The **Chi-Square test** assesses the association between categorical variables.

### 5. A/B testing
**A/B testing** compares two versions (A and B) to determine which performs better statistically.

### 6. Difference between Z and t distribution (Linked to A/B testing)
The **Z distribution** is used when the population variance is known and the sample size is large, while the **t distribution** is used when the sample size is small and the population variance is unknown.

### 7. Outlier treatment method
Outlier treatment methods include **removal**, **transformation**, or **imputation**.

### 8. ANOVA test
The **ANOVA test** compares means among three or more groups to see if at least one differs significantly.

### 9. Cross validation
**Cross-validation** evaluates model performance by dividing data into training and testing sets multiple times.

### 10. How will you work in a machine learning project if there is a huge imbalance in the data?
Handle imbalanced data using techniques like **resampling**, **SMOTE**, or **adjusting class weights**.

### 11. Formula of sigmoid function
The **sigmoid function** is \( \sigma(x) = \frac{1}{1 + e^{-x}} \).

### 12. Can we use sigmoid function in case of multiple classification (I said no)
No, for multiple classification, use the **softmax function** instead.

### 13. Then which function is used
The **softmax function** is used for multiple classification problems.

### 14. What is Area under the curve
**Area Under the Curve (AUC)** measures the ability of a classifier to distinguish between classes.

### 15. Which metric is used to split a node in Decision Tree
**Gini impurity** and **information gain** (entropy) are used to split nodes in a decision tree.

### 16. What is ensemble learning
**Ensemble learning** combines multiple models to improve performance and robustness.

### 17. 3 situation based questions
These questions typically involve applying data science concepts to hypothetical real-world scenarios to assess problem-solving skills.

# Company: CodeBase Solutions

## Role: Data Scientist

### 1. What are the ML techniques you've used in projects?
I've used techniques like **regression**, **classification**, **clustering**, and **dimensionality reduction**.

### 2. Very first question was PCA? Why use PCA?
**Principal Component Analysis (PCA)** is used to **reduce dimensionality** and **remove noise** while retaining the most important information.

### 3. Types of Clustering techniques (Not algorithms)? Which Clustering techniques will you use in which Scenario - example with a Program?
Clustering techniques include **partitioning**, **hierarchical**, and **density-based**. Use **K-Means** for partitioning, **Agglomerative** for hierarchical, and **DBSCAN** for density-based clustering.

### 4. OCR - What type of OCR did you use in your project - Graphical or Non - Graphical?
I used **graphical OCR** for recognizing text in images with complex backgrounds and **non-graphical OCR** for plain text documents.

### 5. OCR - What is a Noise? What types of noise will you face when performing OCR? Handwritten can give more than 70% accuracy when I wrote in 2012 but you're saying 40%.
**Noise** in OCR includes **background artifacts**, **distorted text**, and **blurriness**. Handwritten text introduces additional challenges due to **variability in writing styles**.

### 6. Logistic Regression vs Linear Regression with a real-life example - explain?
**Logistic regression** predicts probabilities and is used for classification (e.g., spam detection). **Linear regression** predicts continuous outcomes (e.g., housing prices).

### 7. Is Decision tree Binary or multiple why use them?
Decision trees can be **binary** or **multi-way**. They are used for their **simplicity** and **interpretability**.

### 8. Do you know Map Reduce and ETL concepts?
Yes, **MapReduce** is a programming model for processing large datasets, and **ETL (Extract, Transform, Load)** is a process in data warehousing.

### 9. What is a Dictionary or Corpus in NLP and how do you build it?
A **dictionary** in NLP contains words and their meanings. A **corpus** is a large collection of texts. They are built by **tokenizing**, **lemmatizing**, and **annotating** text data.

### 10. How do you basically build a Dictionary, Semantic Engine, Processing Engine in a NLP project, where does all the Synonyms (Thesaurus words go).
To build these, use **text processing** techniques to create a dictionary, implement a **semantic engine** for understanding context, and develop a **processing engine** to analyze text. Synonyms are stored in a **thesaurus** within the dictionary.

### 11. What are the Types of Forecasting? What are the ML and DL models for forecasting (He said Fast-forwarding models as example) other than Statistical (ARIMA) models you've used in your projects?
Types of forecasting include **time series** and **causal forecasting**. ML models like **Random Forest**, **Gradient Boosting**, and DL models like **LSTM** are used.

### 12. What is a Neural Network? Types of Neural Networks you know?
A **neural network** is a series of algorithms that attempt to recognize underlying relationships in data. Types include **Feedforward**, **Convolutional (CNN)**, **Recurrent (RNN)**, and **Generative Adversarial Networks (GANs)**.

### 13. Write a Decision Tree model with a Python Program.
```python
from sklearn.tree import DecisionTreeClassifier
# Sample data
X = [[0, 0], [1, 1]]
y = [0, 1]
# Model
clf = DecisionTreeClassifier()
clf = clf.fit(X, y)
# Prediction
print(clf.predict([[2., 2.]]))
```


### 14. How do you build an AZURE ML model? What are all the Azure products you've used? I said Azure ML Studio.
Build an Azure ML model by creating a workspace, uploading data, choosing an algorithm, training the model, and deploying it. Products used include Azure ML Studio, Azure Data Factory, and Azure Databricks.

### 15. Cibil score is an example for Fuzzy model and not a Classification model.
A Cibil score uses fuzzy logic to assess creditworthiness by **considering multiple factors and their degrees of truth**.

### 16. What is an outlier give a real life example? how do you find them and eliminate them? I gave an example of calculating Average salary of an IT employee.
An outlier is a data point **significantly different from others**. Example: In salary data, an outlier might be a very high executive salary. Detect using **IQR** or **Z-score**, and treat by **removal** or **transformation**.

# Company: Deloitte
## Role: Data Scientist
### 1. G values, P values, T values
G values are from G tests for goodness of fit. P values indicate the probability of observing results under the null hypothesis. T values come from t-tests comparing sample means.

### 2. Conditional Probability
Conditional probability is the **likelihood of an event occurring** given that **another event has already occurred**, calculated as P(A|B) = P(A ∩ B) / P(B).

### 3. Central Values of Tendency
Measures include mean, median, and mode.

### 4. Can Linear Regression be used for Classification? If Yes, why if No why?
No, linear regression is for **continuous outcomes**. For classification, use logistic regression due to its **probabilistic interpretation**.

### 5. Hypothesis Testing. Null and Alternate hypothesis
Hypothesis testing involves a null hypothesis (H0) (no effect) and an alternative hypothesis (H1) (an effect exists), to determine if there is enough evidence to reject H0.

### 6. Derivation of Formula for Linear and logistic Regression
Linear: Y = β0 + β1X + ε
Logistic: log(p/(1-p)) = β0 + β1X

### 7. Where to start a Decision Tree. Why use Decision Trees?
Start at the root node. Use decision trees for their **interpretability** and ease of use.

### 8. PCA Advantages and Disadvantages?
Advantages: Dimensionality reduction, noise reduction.
Disadvantages: Loss of interpretability, sensitive to scaling.

### 9. Why Bayes theorem? DB Bayes and Naïve Bayes Theorem?
Bayes theorem calculates conditional probabilities. Naïve Bayes assumes feature independence, simplifying computation.

### 10. Central Limit Theorem?
States the distribution of sample means approximates a normal distribution as sample size increases.

### 11. R packages in and out? For us it's Python Packages in and out.
Python packages include scikit-learn, Pandas, and TensorFlow for ML tasks.

### 12. Scenario based question on when to use which ML model?
Depends on data type, problem nature, and performance requirements.

### 13. Over Sampling and Under Sampling
Over Sampling increases minority class samples. Under Sampling reduces majority class samples.

### 14. Over Fitting and Under Fitting
Overfitting captures noise. Underfitting misses patterns.

### 15. Core Concepts behind Each ML model.
Concepts include model selection, training, evaluation, and optimization.

### 16. Genie Index Vs Entropy
Both measure impurity in decision trees. Gini focuses on binary splits, entropy on information gain.

### 17. how to deal with imbalance data in classification modelling? SMOTHE techniques
Handle imbalanced data with techniques like SMOTE (Synthetic Minority Over-sampling Technique).



# Verizon Data Science Interview Questions

### 1. How many cars are there in Chennai? How do you structurally approach coming up with that number?
To estimate the number of cars in Chennai, you can use a **top-down approach**. Start with the population of Chennai, estimate the percentage of households, average number of cars per household, and adjust for commercial vehicles and public transportation usage.

### 2. Multiple Linear Regression?
Multiple Linear Regression is a statistical technique that models the relationship between a dependent variable and **multiple independent variables**.

### 3. OLS vs MLE?
**Ordinary Least Squares (OLS)** minimizes the sum of squared residuals, while **Maximum Likelihood Estimation (MLE)** maximizes the likelihood that the observed data was generated by the model.

### 4. R2 vs Adjusted R2? During Model Development which one do we consider?
**R²** measures the proportion of variance explained by the model, while **Adjusted R²** adjusts for the number of predictors in the model. During model development, consider **Adjusted R²** to account for model complexity.

### 5. Lift chart, drift chart
A **Lift Chart** shows the performance of a model by comparing the predicted results to actual results. A **Drift Chart** monitors changes in model performance over time due to data drift.

### 6. Sigmoid Function in Logistic Regression
The **Sigmoid Function** transforms the linear output of a logistic regression model into a probability between 0 and 1.

### 7. ROC, what is it? AUC and Differentiation?
The **Receiver Operating Characteristic (ROC)** curve plots the true positive rate against the false positive rate. **AUC (Area Under the Curve)** measures the model's ability to distinguish between classes.

### 8. Linear Regression from Multiple Linear Regression
**Linear Regression** involves one predictor, while **Multiple Linear Regression** involves multiple predictors to model the dependent variable.

### 9. P-Value, what is it and its significance? What does P in P-Value stand for? What is Hypothesis Testing? Null Hypothesis vs Alternate Hypothesis?
The **P-Value** indicates the probability of observing the data given that the null hypothesis is true. **P** stands for **probability**. **Hypothesis Testing** involves testing an assumption (null hypothesis) against an alternative hypothesis.

### 10. Bias-Variance Trade-off?
The **Bias-Variance Trade-off** is the balance between the error due to bias (error from erroneous assumptions) and variance (error from sensitivity to fluctuations in the training set).

### 11. Overfitting vs Underfitting in Machine Learning?
**Overfitting** occurs when a model learns the noise in the training data, while **Underfitting** occurs when a model is too simple to capture the underlying patterns in the data.

### 12. Estimation of Multiple Linear Regression
Estimation involves finding the coefficients that minimize the sum of squared residuals using techniques like OLS.

### 13. Forecasting vs Prediction difference? Regression vs Time Series?
**Forecasting** predicts future values based on past data, while **Prediction** can apply to any unknown value. **Regression** models relationships between variables, while **Time Series** models data points indexed in time order.

### 14. p,d,q values in ARIMA models
**p**: Number of lag observations included. **d**: Number of times the data needs to be differenced. **q**: Size of the moving average window.

### 15. What will happen if d=0?
If **d=0**, the ARIMA model does not perform differencing, treating the series as stationary.

### 16. Is your data for Forecasting Uni or multi-dimensional?
This depends on the dataset. **Unidimensional data** has a single time series, while **Multidimensional data** includes multiple time series.

### 17. How to find the node to start with in a Decision tree?
Start with the feature that provides the highest **information gain** or **reduction in impurity**.

### 18. TYPES of Decision trees - CART vs C4.5 vs ID3
**CART** (Classification and Regression Trees) uses Gini impurity. **C4.5** uses entropy and can handle both categorical and continuous data. **ID3** is an earlier version of C4.5, primarily for categorical data.

### 19. Gini index vs entropy
Both are measures of impurity in decision trees. **Gini Index** calculates the probability of misclassification, while **Entropy** measures the impurity using information theory.

### 20. Linear vs Logistic Regression
**Linear Regression** predicts a continuous output, while **Logistic Regression** predicts a probability for a binary outcome.

### 21. Decision Trees vs Random Forests
**Decision Trees** are prone to overfitting. **Random Forests** use multiple decision trees to reduce overfitting and improve generalization.

### 22. Questions on linear regression, how it works and all
Linear regression models the relationship between a dependent variable and one or more independent variables by fitting a linear equation.

### 23. Asked to write some SQL queries
Be prepared to demonstrate your ability to write and optimize SQL queries for data manipulation and retrieval.

### 24. Asked about past work experience
Discuss your relevant work experience, focusing on projects, tools, and techniques you have used.

### 25. Some questions on inferential statistics (hypothesis testing, sampling techniques)
Understand concepts like hypothesis testing, sampling techniques, and their applications in making inferences about a population.

### 26. Some questions on table (how to filter, how to add calculated fields etc)
Be familiar with SQL operations to filter data, add calculated fields, and manipulate tables.

### 27. Why do you use Licensed Platform when other Open source packages are available?
Licensed platforms often offer **better support, reliability, and additional features** that might not be available in open-source packages.

### 28. What certification have you done?
Highlight any relevant **certifications** in data science, machine learning, or related fields.

### 29. What is a Confidence Interval?
A **Confidence Interval** provides a range of values within which the true parameter value is expected to lie with a certain level of confidence (e.g., 95%).

### 30. What are Outliers? How to Detect Outliers?
**Outliers** are data points that deviate significantly from other observations. They can be detected using **statistical methods** (e.g., Z-scores, IQR).

### 31. How to Handle Outliers?
Outliers can be handled by **removing**, **transforming**, or **treating** them using techniques like **capping** or **imputation**.

# Company: L&T Financial Services

## Role: Data Scientist

### 1. Explain your Projects
Provide a brief overview of your **key projects**, focusing on objectives, methodologies, tools used, and results.

### 2. Assumptions in Multiple Linear Regression
Assumptions include **linearity**, **independence**, **homoscedasticity**, **normality of errors**, and **no multicollinearity**.

### 3. Decision tree algorithm
A **Decision Tree** algorithm splits the data into subsets based on feature values, aiming to maximize information gain or reduce impurity.

### 4. Gini index
The **Gini Index** measures the impurity of a dataset; a lower Gini Index indicates a purer node.

### 5. Entropy
**Entropy** measures the randomness in the information being processed. Higher entropy indicates more disorder and less purity.

### 6. Formulas of Gini and Entropy
- **Gini Index**: \( 1 - \sum (p_i)^2 \)
- **Entropy**: \( -\sum p_i \log(p_i) \)

### 7. Random forest algorithm
**Random Forest** builds multiple decision trees and merges them to get a more accurate and stable prediction.

### 8. XGBoost Algorithm
**XGBoost** is an optimized gradient boosting algorithm that enhances performance and speed.

### 9. Central Limit Theorem
The **Central Limit Theorem** states that the distribution of the sample mean approaches a normal distribution as the sample size increases, regardless of the population's distribution.

### 10. R²
**R²** measures the proportion of variance in the dependent variable explained by the independent variables.

### 11. Adjusted R²
**Adjusted R²** adjusts the R² value for the number of predictors in the model, providing a more accurate measure.

### 12. VIF
**Variance Inflation Factor (VIF)** measures the degree of multicollinearity among independent variables. A high VIF indicates high multicollinearity.

### 13. Different Methods to measure Accuracy
Methods include **confusion matrix**, **accuracy**, **precision**, **recall**, **F1 score**, and **AUC-ROC**.

### 14. Explain Bagging and Boosting
**Bagging** involves creating multiple subsets of data, training models, and averaging their predictions. **Boosting** sequentially adjusts weights for misclassified data points to improve model accuracy.

### 15. Difference Between Bagging and Boosting
**Bagging** reduces variance and helps avoid overfitting, while **Boosting** reduces bias and increases model performance.

### 16. Various Ensemble techniques
Techniques include **Bagging**, **Boosting**, **Stacking**, and **Blending**.

### 17. P value and its significance
The **P-Value** indicates the probability that the observed data would occur by chance under the null hypothesis. A lower p-value (< 0.05) typically suggests rejecting the null hypothesis.

### 18. F1 Score
The **F1 Score** is the harmonic mean of precision and recall, providing a balance between the two metrics.

### 19. Type 1 and Type II error
**Type I Error** (false positive) occurs when the null hypothesis is incorrectly rejected. **Type II Error** (false negative) occurs when the null hypothesis is incorrectly accepted.

### 20. Logical questions for Type 1 and Type II error
Be prepared to explain scenarios and consequences of Type I and Type II errors in real-world contexts.

### 21. Logical questions for Null and Alternate Hypothesis
Explain the **null hypothesis** as a statement of no effect or difference, and the **alternate hypothesis** as a statement indicating the presence of an effect or difference.




# Role: Data Scientist

### 1. Decorators in Python
Decorators in Python are used to **modify the behavior** of a function or method. They are **higher-order functions** that take another function as an argument and extend its behavior without explicitly modifying it.

#### Live Example:
```python
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```

### 2. Generators in Python
Generators are used for creating iterators in a memory-efficient way. They use the yield keyword instead of return to produce a series of values lazily, on-demand.

Live Example:
```python
def my_generator():
    yield 1
    yield 2
    yield 3

gen = my_generator()
for value in gen:
    print(value)
```

### SQL Questions
### 3.1 Group by Top 2 Salaries for Employees
To find the top 2 salaries for employees grouped by department:
```sql
SELECT department, employee, salary
FROM (
    SELECT department, employee, salary,
           ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rank
    FROM employees
) as ranked_salaries
WHERE rank <= 2;
```

### 4. Pandas find Numeric and Categorical Columns
```python
import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': ['a', 'b', 'c'],
    'C': [1.5, 2.5, 3.5]
})

numeric_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(include=['object']).columns
```

### 4.1 For Numeric columns, find the mean of the entire column and add that value to each row of the column.
```python
for col in numeric_cols:
    mean_value = df[col].mean()
    df[col] = df[col] + mean_value
```

### 5. What is Gradient Descent?
Gradient Descent is an optimization algorithm used to **minimize the cost function** in machine learning models by iteratively adjusting the model parameters in the direction of the **negative gradient**.

### 5.1 What is Learning Rate and Why is it reduced sometimes?
The learning rate determines the size of the steps taken towards the **minimum of the cost function**. It is sometimes reduced to **ensure convergence** and **avoid overshooting** the minimum during the optimization process.

### 6. Two Logistic Regression Models - Which one will you choose - One is trained on 70% and other on 80% data. Accuracy is almost the same.
Choosing between two logistic regression models trained on 70% and 80% of the data, with similar accuracy, generally involves considering other factors like variance and overfitting. The model trained on 80% might be preferred for better generalization due to more training data.

### 7. What is LogLoss?
LogLoss (Logarithmic Loss) measures the performance of a classification model where the prediction is a probability value between 0 and 1. Lower LogLoss indicates a better-performing model.

### 8. Explain your Projects end to end. (15-20 mins)
In this part, provide a detailed walkthrough of your projects, covering the problem statement, data collection and preprocessing, model building and evaluation, results, and deployments. Highlight the key challenges faced and the solutions implemented.


# Role: Data Science Intern

### 1. Tell me about your journey as a Data Science aspirant
I started as a data science enthusiast by taking online courses and working on personal projects. Over time, I gained **hands-on experience** through internships and collaborating on real-world problems, which solidified my understanding and passion for data science.

### 2. What was the one challenging project or task that you did in this domain and why was it challenging?
One challenging project was predicting customer churn for a telecom company. It was challenging due to the **imbalanced dataset** and the need for feature engineering to improve model performance.

### 3. What model did you use for that? I replied Random Forests
For this project, I used **Random Forests** due to their ability to handle **non-linear relationships** and provide **feature importance** insights.

### 4. What is Random Forest and how is it used?
**Random Forest** is an ensemble learning method that combines multiple decision trees to improve **accuracy** and **reduce overfitting**. It is used for both **classification** and **regression** tasks.

### 5. How are Random Forest different from Decision Trees and what problems do they solve that decision trees can't?
Random Forests reduce the **overfitting** problem commonly seen with Decision Trees by averaging the results of multiple trees, thus providing **better generalization**.

### 6. Multi class Classification and which metric is preferred for it
In multi-class classification, metrics like **F1 Score** or **Accuracy** are often preferred, but **F1 Score** is particularly useful when dealing with imbalanced classes.

### 7. Given a banking scenario to predict Loan Defaulters, which metric will you use?
For predicting loan defaulters, **ROC-AUC** or **F1 Score** are preferred metrics as they handle **class imbalance** and focus on both precision and recall.

### 8. How will you handle the class imbalance in this case?
To handle class imbalance, techniques like **SMOTE (Synthetic Minority Over-sampling Technique)**, **class weighting**, or **undersampling** the majority class can be used.

# Company: Latentview Analytics

### 1. What is mean and median
**Mean** is the average of a dataset, while **median** is the middle value that separates the dataset into two halves. Median is less affected by **outliers**.

### 2. Difference between normal and gaussian distribution
There is no difference; **normal distribution** is another name for **Gaussian distribution**. Both describe a symmetric, bell-shaped distribution.

### 3. What is central limit theorem
The **Central Limit Theorem** states that the sampling distribution of the sample mean approaches a **normal distribution** as the sample size increases, regardless of the population's distribution.

### 4. What is null hypothesis
The **null hypothesis** is a statement that there is **no effect** or **no difference**, and it serves as the basis for testing statistical significance.

### 5. What is confidence interval
A **confidence interval** is a range of values that is likely to contain the **true parameter** of the population with a certain level of confidence (e.g., 95%).

### 6. What is covariance and correlation and how will you interpret it.
**Covariance** indicates the direction of the linear relationship between variables, while **correlation** measures both the strength and direction of the linear relationship. A correlation close to **+1** or **-1** indicates a strong relationship, while close to **0** indicates no relationship.

### 7. How will you find out the outliers in the dataset and is it always necessary to remove outliers?
Outliers can be detected using **statistical methods** (e.g., z-score, IQR). It is not always necessary to remove outliers; it depends on whether they represent **valid data points** or errors.

### 8. Explain about Machine Learning
**Machine Learning** is a field of AI that enables computers to learn from data and make decisions without explicit programming. It involves **supervised**, **unsupervised**, and **reinforcement learning**.

### 9. Explain the algorithm of your choice
**Random Forest** is an ensemble learning algorithm that constructs multiple decision trees during training and outputs the mode of the classes for classification or mean prediction for regression.

### 10. Different methods of missing values imputation
Methods include **mean/mode/median imputation**, **KNN imputation**, and using **algorithms** like **Random Forest** to predict missing values.

### 11. Explain me your ml project
In my ML project, I built a predictive model to classify customer reviews as positive or negative. I used **NLP techniques** for text preprocessing and a **Random Forest classifier** for prediction.

### 12. How did you handle imbalance dataset
I handled the imbalanced dataset using **SMOTE**, adjusting **class weights**, and **undersampling** the majority class to ensure balanced training.

### 13. What is stratified sampling
**Stratified sampling** involves dividing the population into **subgroups (strata)** and sampling from each subgroup to ensure representation of all groups.

### 14. Difference between standard scalar and normal scalar
**StandardScaler** standardizes features by removing the mean and scaling to unit variance, while **MinMaxScaler** scales features to a given range, usually [0, 1].

### 15. Different type of visualization in DL project
Visualizations include **loss and accuracy plots**, **confusion matrices**, and **activation maps** to understand model performance and feature learning.

### 16. What architecture have you used
I used a **Convolutional Neural Network (CNN)** architecture for image classification due to its ability to capture spatial hierarchies in images.

### 17. Why have you not used RNN in your nlp project
I used **Transformers** instead of RNNs in my NLP project because Transformers are better at capturing **long-range dependencies** and are more **parallelizable**.

### 18. Why we don't prefer CNN in nlp based project
CNNs are less preferred for NLP because they are not as effective in capturing the **sequential nature** of text as RNNs or Transformers.

### 19. What is exploding gradient and vanishing gradient and how to rectify it
**Exploding gradients** occur when gradients grow too large, while **vanishing gradients** occur when they become too small. Solutions include **gradient clipping** and using **activation functions** like **ReLU**.

### 20. Difference between LSTM and GRU
**LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)** are both RNN variants. GRUs are simpler and **faster** but may be less effective on complex tasks compared to LSTMs.

### 21. What is precision and recall
**Precision** is the ratio of true positives to all predicted positives, while **recall** is the ratio of true positives to all actual positives. Both are important for evaluating model performance.

### 22. What is AUC metric
**AUC (Area Under the Curve)** measures the ability of a classifier to distinguish between classes and is used to evaluate **binary classification** models.

### 23. What if your precision and recall are same
If precision and recall are the same, it indicates a **balance** between the model's ability to identify true positives and avoid false positives, resulting in an **F1 Score** equal to the precision and recall value.

# Data Science Interview Questions

### 1. Naive bayes assumptions
Naive Bayes assumes that features are **independent** given the class label and that each feature contributes equally and independently to the probability of the outcome.

### 2. What are the approaches for solving class imbalance problem?
Approaches include **resampling** methods (oversampling and undersampling), using **synthetic data** (e.g., SMOTE), and **adjusting class weights** during training.

### 3. When sampling what types of biases can be inflicted? How to control the biases?
Types of biases include **selection bias**, **sampling bias**, and **response bias**. They can be controlled by ensuring **random sampling**, using **stratified sampling**, and **blinding** during data collection.

### 4. GRU is faster compared to LSTM. Why?
GRU is faster because it has a **simpler architecture** with fewer gates than LSTM, making it less computationally intensive.

### 5. What is difference between K-NN and K-Means clustering?
**K-NN** is a supervised learning algorithm used for classification and regression, while **K-Means** is an unsupervised learning algorithm used for clustering data into k groups.

### 6. How to determine if a coin is biased? Hint: Hypothesis testing
To determine if a coin is biased, perform a **hypothesis test** such as a **binomial test** to compare the observed number of heads/tails to the expected distribution under the null hypothesis of a fair coin.

### 7. How will you present the statistical inference of a particular numerical column?
Present the statistical inference using **summary statistics** (mean, median, variance), **visualizations** (histograms, box plots), and **hypothesis tests** to draw conclusions.

### 8. How would you design a data science pipeline?
A data science pipeline includes **data collection**, **data cleaning**, **feature engineering**, **model training**, **model evaluation**, and **deployment**. Each step ensures data quality and model performance.

### 9. Explain back propagation in few words and its variants?
**Backpropagation** is an algorithm used to update the weights of a neural network by calculating the gradient of the loss function. Variants include **Stochastic Gradient Descent (SGD)**, **Mini-batch Gradient Descent**, and **Adam**.

### 10. Explain topic modeling in NLP and various methods in performing topic modeling.
**Topic modeling** is a technique to discover abstract topics within a collection of documents. Methods include **Latent Dirichlet Allocation (LDA)** and **Non-negative Matrix Factorization (NMF)**.



# Company: Myntra

## Role: Data Analyst

### Round type: Use case Round

#### Problem Statement:

Given 2 teams of Myntra namely:

1. **Finance Team**: They focus on making decisions that are **money-driven**.
2. **Customer Experience Team**: They aim to improve the **customer experience** with Myntra.

Whenever a customer places a refund request, Myntra can process it in 2 different ways:

1. **Directly accept the return request.**
2. **Put the request on hold and verify the product for damages or manhandling by the customer.** Only if the products are found to be in proper state, accept the return.

Now, there is a conflict of opinion between these two teams:

- **Finance Team** likes the 2nd option as it **minimizes the chances of loss**.
- **Customer Experience Team** likes the 1st option as their main aim is to **improve customer experience**.

#### Questions:

### 1. Suppose you are part of the Customer Experience team. How would you convince the Finance team to follow the 1st step?
To convince the Finance team to follow the 1st step, highlight the **long-term benefits** of **customer satisfaction** and **loyalty**. Explain that a positive return experience can lead to **repeat purchases**, **positive reviews**, and **word-of-mouth referrals**, which ultimately increases revenue.

### 2. What kind of data would you be looking for solving this task?
I would look for data on:
- **Customer return rates** and **reasons for returns**.
- **Customer satisfaction scores** related to the return process.
- **Repeat purchase rates** and **customer lifetime value**.
- **Cost analysis** of processing returns directly versus holding and verifying.
- **Historical data** on **fraudulent returns** and **losses** due to unchecked returns.

### 3. Is there any need for model building for this use case?
Model building can be useful to predict the **impact of return policies** on customer behavior and financial outcomes. A predictive model can help:
- Estimate the **probability of repeat purchases** based on the return experience.
- Assess the **risk of fraudulent returns**.
- Quantify the **long-term revenue impact** of different return policies.

Using these insights, the decision can be data-driven, balancing both customer experience and financial prudence.




# Company: Ericsson

## Role: Data Scientist

### Round No: 1st Round

### 1. How to reverse a linked list
To reverse a linked list, **iterate** through the list and **change the next pointers** of each node to point to the previous node. Continue until all nodes are reversed.

### 2. Given a logistic regression model in production, how would you find out the coefficients of different input features.
To find the coefficients, **access the trained model** object and use the **attribute** that stores the coefficients, typically `model.coef_` in libraries like **scikit-learn**.

### 3. What is the p-value in OLS regression
The p-value in OLS regression indicates the **probability** that the observed relationship between the independent and dependent variables occurred **by chance**. A **low p-value** (< 0.05) suggests that the coefficient is **statistically significant**.

### 4. What's the reason for high bias or variance
High bias is often due to an **underfitting** model that is too simple, while high variance results from an **overfitting** model that is too complex and sensitive to the training data.

### 5. Which models are generally high biased or high variance
**Linear models** and **simple algorithms** like **Linear Regression** are typically high bias, while **complex models** like **Decision Trees** and **neural networks** tend to be high variance.

### 6. Write code to find the 8 highest value in the DataFrame
```python
import pandas as pd

# Assuming df is your DataFrame and you want to find the 8 highest values in a column named 'values'
top_8_values = df['values'].nlargest(8)
print(top_8_values)
```

### 7. What's the difference between array and list
Arrays are homogeneous data structures with fixed size, used for numerical operations (efficient). Lists are heterogeneous, dynamic in size, and more flexible for general purposes.

### 8. What's the difference between Gradient Boosting and XGBoost
Gradient Boosting is a general boosting technique, while XGBoost is an optimized implementation that is faster, more efficient, and offers additional features like regularization.

### 9. Is XOR data linearly separable
No, XOR data is not linearly separable because a single linear boundary cannot separate the classes. It requires a non-linear classifier.

### 10. How do we classify XOR data using logistic regression
To classify XOR data using logistic regression, you can use polynomial features or kernel trick to transform the data into a higher-dimensional space where it becomes linearly separable.

### 11. Some questions from my previous projects
Be prepared to discuss your projects in detail, focusing on your role, the technologies used, the challenges faced, and the outcomes.

### 12. Given a sand timer of 4 and 7 mins how would you calculate 10 mins duration.
Start both timers simultaneously. When the 4-minute timer runs out, flip it (4 minutes). When the 7-minute timer runs out, flip it (7 minutes). When the 4-minute timer runs out again (8 minutes), flip it immediately. When it runs out for the last time, 10 minutes will have passed.

### 13. What's the angle between hour and minute hand in clock as 3:15
At 3:15, the minute hand is at 90 degrees (15 minutes), and the hour hand is at 97.5 degrees (3 hours and 15 minutes). The angle between them is 7.5 degrees.



# Company: Legato Health Technologies

## Role: MLOps Engineer

### 1. Complete ML technical stack used in project?
A complete ML technical stack typically includes **data collection** (APIs, databases), **data processing** (Pandas, NumPy), **model training** (scikit-learn, TensorFlow, PyTorch), **model deployment** (Docker, Kubernetes), and **monitoring** (Prometheus, Grafana).

### 2. Different activation function?
Common activation functions include **ReLU** (Rectified Linear Unit), **Sigmoid**, **Tanh**, and **Leaky ReLU**. Each function has its own characteristics and is used based on the specific needs of the neural network.

### 3. How do you handle imbalance data?
Imbalance data can be handled using techniques like **resampling** (oversampling/undersampling), **SMOTE** (Synthetic Minority Over-sampling Technique), and using algorithms that are robust to imbalance like **XGBoost** or **class weight adjustment**.

### 4. Difference between sigmoid and softmax?
The **sigmoid** function outputs a probability between 0 and 1 for binary classification, while the **softmax** function outputs a probability distribution across multiple classes, ensuring that the sum of probabilities is 1.

### 5. Explain about optimizers?
Optimizers like **SGD** (Stochastic Gradient Descent), **Adam**, and **RMSprop** are algorithms used to update the weights of a neural network to minimize the loss function. They differ in their approach to adjusting learning rates and handling gradients.

### 6. Precision-Recall Trade off?
The **precision-recall trade-off** involves balancing between precision (the accuracy of positive predictions) and recall (the ability to find all positive instances). **High precision** reduces false positives, while **high recall** reduces false negatives.

### 7. How do you handle False Positives?
False positives can be handled by **adjusting the decision threshold**, using more **robust features**, and applying techniques like **cross-validation** and **ensemble methods** to improve model accuracy.

### 8. Explain LSTM architecture by taking example of 2 sentences and how it will be processed?
An **LSTM** (Long Short-Term Memory) processes sequences of data. For example, two sentences "I love machine learning" and "LSTM models are powerful" are tokenized, embedded, and passed through the LSTM layers, which capture the dependencies and context between words over time.

### 9. Decision Tree Parameters?
Key parameters for decision trees include **max_depth**, **min_samples_split**, **min_samples_leaf**, **criterion** (e.g., Gini impurity, entropy), and **max_features**.

### 10. Bagging and boosting?
**Bagging** (Bootstrap Aggregating) involves training multiple models on different subsets of the data and averaging their predictions. **Boosting** sequentially trains models, each correcting errors of the previous one, to improve performance.

### 11. Explain bagging internals?
In **bagging**, multiple datasets are generated by randomly sampling with replacement from the original dataset. Models are trained on these datasets, and their predictions are combined (e.g., by averaging or voting) to produce the final output.

### 12. Write a program by taking an url and give a rough code approach how you will pass payload and make a post request?
```python
import requests

url = 'http://example.com/api'
payload = {'key1': 'value1', 'key2': 'value2'}
headers = {'Content-Type': 'application/json'}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

### 13. Different modules used in python?
Common Python modules include NumPy (numerical operations), Pandas (data manipulation), scikit-learn (machine learning), requests (HTTP requests), and matplotlib (visualization).

### 14. Another coding problem of checking balanced parentheses?
```python
def is_balanced(s):
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in pairs.values():
            stack.append(char)
        elif char in pairs.keys():
            if stack == [] or pairs[char] != stack.pop():
                return False
    return stack == []

# Example usage
print(is_balanced("(){}[]"))  # True
print(is_balanced("([{}])"))  # True
print(is_balanced("(]"))      # False
```



# Company: Legato Health Technologies

## Role: MLOps Engineer

### 1. Complete ML technical stack used in project?
The **ML technical stack** includes **data ingestion** (e.g., Kafka, Flume), **data storage** (e.g., HDFS, S3), **data processing** (e.g., Spark, Hadoop), **model training** (e.g., TensorFlow, PyTorch), **model deployment** (e.g., Kubernetes, Docker), and **monitoring** (e.g., Prometheus, Grafana).

### 2. Different activation function?
Common activation functions include **ReLU** (Rectified Linear Unit), **Sigmoid**, **Tanh**, and **Softmax**. Each has unique properties suitable for different types of neural networks.

### 3. How do you handle imbalance data?
Imbalanced data can be handled using techniques like **resampling** (oversampling the minority class or undersampling the majority class), using **SMOTE** (Synthetic Minority Over-sampling Technique), and **adjusting class weights** in the algorithm.

### 4. Difference between sigmoid and softmax?
**Sigmoid** activation function outputs a value between **0 and 1** and is used for **binary classification**. **Softmax** outputs a probability distribution over multiple classes, making it suitable for **multi-class classification**.

### 5. Explain about optimizers?
Optimizers like **SGD (Stochastic Gradient Descent)**, **Adam (Adaptive Moment Estimation)**, and **RMSprop** are used to minimize the loss function by updating the model's parameters efficiently during training.

### 6. Precision-Recall Trade off?
The **Precision-Recall Trade-off** involves balancing **precision** (correct positive predictions out of all positive predictions) and **recall** (correct positive predictions out of all actual positives). Improving one can often lead to a decrease in the other.

### 7. How do you handle False Positives?
False positives can be handled by **adjusting the decision threshold**, using **precision-recall trade-offs**, and employing techniques like **cost-sensitive learning** to minimize the impact.

### 8. Explain LSTM architecture by taking example of 2 sentences and how it will be processed?
LSTM (Long Short-Term Memory) processes sequences of data. For example, given sentences "I love machine learning" and "Machine learning is fascinating", LSTM processes each word in sequence, maintaining and updating its **cell state** and **hidden state** to capture context and dependencies over time.

### 9. Decision Tree Parameters?
Key parameters of a decision tree include **max_depth** (maximum depth of the tree), **min_samples_split** (minimum samples required to split an internal node), and **criterion** (function to measure the quality of a split, like Gini or entropy).

### 10. Bagging and boosting?
**Bagging** (Bootstrap Aggregating) involves training multiple models on different subsets of data and averaging their predictions. **Boosting** sequentially trains models, each trying to correct errors made by the previous one, to improve overall performance.

### 11. Explain bagging internals
Bagging involves creating **multiple subsets** of the original dataset using **bootstrapping** (random sampling with replacement), training a model on each subset, and then **aggregating** their predictions (e.g., through voting or averaging).

### 12. Write a program by taking an URL and give a rough code approach on how you will pass payload and make a POST request?
```python
import requests

url = 'http://example.com/api'
payload = {'key1': 'value1', 'key2': 'value2'}

response = requests.post(url, json=payload)

print(response.status_code)
print(response.json())
```


### 13. Different modules used in Python?
Common Python modules include numpy (numerical operations), pandas (data manipulation), matplotlib (data visualization), scikit-learn (machine learning), and requests (HTTP requests).

### 14. Another coding problem of checking balanced parentheses?
```python
def is_balanced(s):
    stack = []
    matching_bracket = {')': '(', ']': '[', '}': '{'}
    for char in s:
        if char in matching_bracket.values():
            stack.append(char)
        elif char in matching_bracket.keys():
            if stack == [] or matching_bracket[char] != stack.pop():
                return False
    return stack == []

# Example usage
print(is_balanced("([])"))  # True
print(is_balanced("([)]"))  # False
```


# Data Science Interview Questions:

### 1. How do check the Normality of a dataset?
To check the normality of a dataset, you can use **visual methods** like Q-Q plots and **statistical tests** such as the Shapiro-Wilk test or Kolmogorov-Smirnov test.

### 2. Difference Between Sigmoid and Softmax functions?
The **Sigmoid function** is used for binary classification, squashing input values to a range between **0 and 1**. The **Softmax function** is used for multi-class classification, converting input values into **probabilities** that sum to **1**.

### 3. Can logistic regression be used for more than 2 classes?
Yes, logistic regression can be extended for more than 2 classes using techniques like **One-vs-Rest (OvR)** or **Multinomial Logistic Regression**.

### 4. What are Loss Function and Cost Functions? Explain the key Difference Between them?
A **Loss Function** measures the error for a single training example, while a **Cost Function** is the **average loss** over the entire training dataset. Essentially, the cost function is an **aggregate** of the loss function.

### 5. What is F1 score? How would you use it?
The **F1 score** is the harmonic mean of precision and recall, used to balance the trade-off between **false positives and false negatives**. It's especially useful when the class distribution is **imbalanced**.

### 6. In a neural network, what if all the weights are initialized with the same value?
If all weights are initialized with the same value, the model will lack **symmetry breaking**, leading to the same updates during training and **failing to learn** effectively.

### 7. Why should we use Batch Normalization?
**Batch Normalization** stabilizes and accelerates training by normalizing the input of each layer, which helps in reducing the impact of **internal covariate shift**.

### 8. In a CNN, if the input size is 5x5 and the filter size is 7x7, then what would be the size of the output?
If the filter size is larger than the input size, the output size would be **undefined** or result in a **dimension error**. Typically, the filter size should be less than or equal to the input size.

### 9. What do you mean by exploding and vanishing gradients?
**Exploding gradients** occur when gradients become excessively large, causing unstable training. **Vanishing gradients** happen when gradients become too small, slowing down or preventing the training process.

### 10. What are the applications of transfer learning in Deep Learning?
**Transfer learning** allows a model trained on one task to be used on another, leveraging **pre-trained models** for tasks like **image recognition**, **NLP**, and **medical diagnosis**.

### 11. Why does a Convolutional Neural Network (CNN) work better with image data?
A **Convolutional Neural Network (CNN)** works better with image data because it captures **spatial hierarchies** through convolutional layers, making it effective for **feature extraction and pattern recognition**.

# Data Science Interview Questions:

### 1. What is the Central Limit Theorem and why is it important?
The **Central Limit Theorem (CLT)** states that the distribution of the sample mean approximates a **normal distribution** as the sample size becomes large. It's important because it allows us to make **inferences** about population parameters.

### 2. What is the difference between type I vs type II error?
A **Type I error** (false positive) occurs when a true null hypothesis is rejected. A **Type II error** (false negative) occurs when a false null hypothesis is not rejected. Balancing these errors is crucial for hypothesis testing.

### 3. Tell me the difference between an inner join, left join/right join, and union.
An **inner join** returns rows with matching values in both tables. A **left join** returns all rows from the left table and matched rows from the right table, while a **right join** does the opposite. A **union** combines the results of two queries into a single dataset.

### 4. Explain the 80/20 rule, and tell me about its importance in model validation.
The **80/20 rule**, or **Pareto Principle**, states that 80% of effects come from 20% of causes. In model validation, it emphasizes the importance of **allocating resources efficiently** to the most impactful features or data.

### 5. What is one way that you would handle an imbalanced data set that’s being used for prediction (i.e., vastly more negative classes than positive classes)?
One way to handle an imbalanced dataset is by using **resampling techniques** like **oversampling the minority class** or **undersampling the majority class** to achieve a more balanced distribution.

### 6. Is it better to spend five days developing a 90-percent accurate solution or 10 days for 100-percent accuracy?
It depends on the **context and cost** of errors. Often, a **90% accurate solution** is sufficient and more cost-effective. Striving for **100% accuracy** can lead to overfitting and diminishing returns.

### 7. Most common characteristics used in descriptive statistics?
Common characteristics in descriptive statistics include **mean**, **median**, **mode**, **variance**, **standard deviation**, **range**, and **percentiles**.

### 8. What do you mean by degree of freedom?
The **degree of freedom** refers to the number of values in a calculation that are **free to vary**. It's crucial in statistical testing to determine the **shape of distributions**.

### 9. Why is the t-value the same for a 90% two-tail and 95% one-tail test?
The **t-value** is the same because the **total area** under the curve for both tests represents the **same probability** (i.e., 10% in the tails for a 90% two-tail test and 5% in one tail for a 95% one-tail test).

### 10. What does it mean if a model is heteroscedastic? What about homoscedastic?
A model is **heteroscedastic** if the **variance** of the errors varies across observations. It is **homoscedastic** if the **variance** is constant across observations.

### 11. You roll a biased coin (p(head)=0.8) five times. What’s the probability of getting three or more heads?
The probability can be calculated using the **binomial distribution** formula. For **p(head)=0.8**, the probability of getting **three or more heads** out of five rolls is approximately **0.9421**.

### 12. What does interpolation and extrapolation mean? Which is generally more accurate?
**Interpolation** involves estimating values within the range of the data, while **extrapolation** involves estimating values outside the data range. **Interpolation** is generally more accurate because it relies on known data points.

# Data Science Interview Questions:

### 1. What is the aim of conducting A/B Testing?
The aim of **A/B Testing** is to compare two versions of a variable to determine which one performs better in terms of a specified metric, allowing for **data-driven decisions**.

### 2. Explain p-value.
The **p-value** measures the probability of obtaining results at least as extreme as the observed results, assuming the null hypothesis is true. A **low p-value** (< 0.05) indicates **strong evidence** against the null hypothesis.

### 3. Explain how a ROC curve works?
A **ROC curve** plots the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)** at various threshold settings. It helps to evaluate the **discriminative ability** of a classifier.

### 4. What is pruning in Decision Tree?
**Pruning** is the process of removing **nodes** from a decision tree to reduce its size and **improve generalization** by avoiding overfitting.

### 5. How will you define the number of clusters in a clustering algorithm?
The number of clusters can be defined using methods like the **Elbow Method**, **Silhouette Score**, and **Gap Statistic**, which help determine the optimal number of clusters.

### 6. When to use Precision and when to use Recall?
Use **Precision** when the cost of false positives is high, and **Recall** when the cost of false negatives is high. The choice depends on the specific context of the problem.

### 7. What are the assumptions required for linear regression? What if some of these assumptions are violated?
Linear regression assumes **linearity**, **independence**, **homoscedasticity**, **normality**, and **no multicollinearity**. If these assumptions are violated, it can lead to **biased estimates** and **invalid inferences**.

### 8. How are covariance and correlation different from one another?
**Covariance** measures the **direction** of the linear relationship between two variables, while **correlation** measures both the **strength and direction** of the relationship and is **scaled** between -1 and 1.

### 9. How can we relate standard deviation and variance?
**Standard deviation** is the **square root** of the variance. Both measure the **spread** of data points, with variance providing a squared measure and standard deviation being in the same units as the data.

### 10. Explain the phrase "Curse of Dimensionality".
The **"Curse of Dimensionality"** refers to the various issues that arise when analyzing data in high-dimensional spaces, where the volume of the space increases exponentially, making **data sparse** and models prone to overfitting.

### 11. What does the term Variance Inflation Factor mean?
The **Variance Inflation Factor (VIF)** quantifies how much the **variance** of a regression coefficient is inflated due to multicollinearity. High VIF values indicate high correlation among predictors.

### 12. What is the significance of Gamma and Regularization in SVM?
**Gamma** defines the influence of a single training example in **Support Vector Machines (SVM)**, while **regularization** helps control the **trade-off** between achieving a low error on training data and minimizing model complexity.



# Data Science Interview Questions

### 1. How will you calculate the Sensitivity of machine learning models?
Sensitivity, or **Recall**, is calculated as **TP / (TP + FN)**, where TP is True Positives and FN is False Negatives. It measures the model's ability to correctly identify positive instances.

### 2. What do you mean by cluster sampling and systematic sampling?
**Cluster Sampling** involves dividing the population into clusters, then randomly selecting entire clusters. **Systematic Sampling** selects every k-th individual from a list after a random start.

### 3. Explain Eigenvectors and Eigenvalues.
**Eigenvectors** are vectors that remain in the same direction after a linear transformation. **Eigenvalues** are scalars indicating how much the eigenvector is scaled during the transformation.

### 4. Explain Gradient Descent.
**Gradient Descent** is an optimization algorithm used to minimize the loss function by iteratively moving in the direction of the steepest descent, calculated by the gradient.

### 5. How does Backpropagation work? Also, state its various variants.
**Backpropagation** is a method to update the weights in a neural network by propagating the error backward from the output layer to the input layer. Variants include **Stochastic**, **Mini-batch**, and **Batch** gradient descent.

### 6. What do you know about Autoencoders?
**Autoencoders** are neural networks used for unsupervised learning that aim to compress input data into a lower-dimensional code and then reconstruct the output to be as close to the original input as possible.

### 7. What is Dropout in Neural Networks?
**Dropout** is a regularization technique where randomly selected neurons are ignored during training to prevent overfitting.

### 8. What is the difference between Batch and Stochastic Gradient Descent?
**Batch Gradient Descent** uses the entire dataset to compute the gradient, while **Stochastic Gradient Descent (SGD)** updates the model parameters using one training example at a time.

### 9. What are the different kinds of Ensemble learning?
Ensemble learning techniques include **Bagging** (e.g., Random Forest), **Boosting** (e.g., AdaBoost, Gradient Boosting), and **Stacking**.

### 10. What is entropy, information gain, and Gini index in decision tree classifier and regression?
- **Entropy** measures the impurity in a dataset.
- **Information Gain** is the reduction in entropy after a dataset is split.
- **Gini Index** measures the impurity based on the probability of a randomly chosen element being misclassified.

# Role: Data Scientist

### 1. What is central tendency?
**Central Tendency** refers to the measure that represents the center or typical value of a dataset, including mean, median, and mode.

### 2. Which central tendency method is used if there exist any outliers?
The **Median** is used as it is less affected by outliers compared to the mean.

### 3. Central Limit Theorem
The **Central Limit Theorem** states that the distribution of the sample mean approximates a normal distribution as the sample size becomes large, regardless of the population's distribution.

### 4. Chi-Square test
The **Chi-Square test** is a statistical test used to determine if there is a significant association between categorical variables.

### 5. A/B testing
**A/B Testing** is a method of comparing two versions of a variable to determine which one performs better in a controlled environment.

### 6. Difference between Z and t distribution (Linked to A/B testing)
The **Z-distribution** is used when the sample size is large and population variance is known. The **t-distribution** is used when the sample size is small and population variance is unknown.

### 7. Outlier treatment method
Outlier treatment methods include **removal**, **transformation**, or **capping and flooring** based on the percentile.

### 8. ANOVA test
**ANOVA (Analysis of Variance)** is used to compare the means of three or more samples to understand if at least one sample mean is significantly different.

### 9. Cross validation
**Cross-validation** is a technique to evaluate the model's performance by partitioning the data into subsets, training the model on some subsets, and validating it on the remaining subsets.

### 10. How will you work in a machine learning project if there is a huge imbalance in the data?
Techniques include **resampling** (over-sampling and under-sampling), **SMOTE (Synthetic Minority Over-sampling Technique)**, and using **class-weighted algorithms**.

### 11. Formula of sigmoid function
The sigmoid function is given by **1 / (1 + e^(-x))**.

### 12. Can we use sigmoid function in case of multiple classification (I said no)
No, the **sigmoid function** is used for binary classification.

### 13. Then which function is used?
For multi-class classification, the **Softmax function** is used.

### 14. What is Area under the curve?
**Area Under the Curve (AUC)** refers to the area under the ROC curve, indicating the model's ability to distinguish between classes.

### 15. Which metric is used to split a node in Decision Tree?
**Gini Index** or **Information Gain** is used to split nodes in a decision tree.

### 16. What is ensemble learning?
**Ensemble Learning** combines multiple models to produce a better overall model. Techniques include **Bagging**, **Boosting**, and **Stacking**.

### 17. 3 situation-based questions
**Example questions may include:**
- How would you handle missing data in a dataset?
- Describe a time when you improved a model’s performance.
- How do you prioritize tasks in a machine learning project?

## Statistics Checklist before going for a Data Science Interview:

#### 1. Inferential and descriptive Statistics
#### 2. Sample
#### 3. Population
#### 4. Random variables
#### 5. Probability Distribution Function
#### 6. Probability Mass Function
#### 7. Cumulative Distribution Function
#### 8. Expectation and Variance
#### 9. Binomial Distribution
#### 10. Bernoulli Distribution
#### 11. Normal Distribution
#### 12. Z-score
#### 13. Central Limit Theorem
#### 14. Hypothesis Testing
#### 15. Confidence Interval
#### 16. Chi Square Test
#### 17. Anova Test
#### 18. F-Stats

## Role: Data Scientist

### 1. What is central tendency?
**Central Tendency** refers to the measure that represents the center or typical value of a dataset, including mean, median, and mode.

### 2. Which central tendency method is used if there exist any outliers?
The **Median** is used as it is less affected by outliers compared to the mean.

### 3. Central limit theorem
The **Central Limit Theorem** states that the distribution of the sample mean approximates a normal distribution as the sample size becomes large, regardless of the population's distribution.

### 4. Chi-Square test
The **Chi-Square test** is a statistical test used to determine if there is a significant association between categorical variables.

### 5. A/B testing
**A/B Testing** is a method of comparing two versions of a variable to determine which one performs better in a controlled environment.

### 6. Difference between Z and t distribution (Linked to A/B testing)
The **Z-distribution** is used when the sample size is large and population variance is known. The **t-distribution** is used when the sample size is small and population variance is unknown.

### 7. Outlier treatment method
Outlier treatment methods include **removal**, **transformation**, or **capping and flooring** based on the percentile.

### 8. ANOVA test
**ANOVA (Analysis of Variance)** is used to compare the means of three or more samples to understand if at least one sample mean is significantly different.

### 9. Cross validation
**Cross-validation** is a technique to evaluate the model's performance by partitioning the data into subsets, training the model on some subsets, and validating it on the remaining subsets.

### 10. How will you work in a machine learning project if there is a huge imbalance in the data?
Techniques include **resampling** (over-sampling and under-sampling), **SMOTE (Synthetic Minority Over-sampling Technique)**, and using **class-weighted algorithms**.

### 11. Formula of sigmoid function
The sigmoid function is given by **1 / (1 + e^(-x))**.

### 12. Can we use sigmoid function in case of multiple classification (I said no)
No, the **sigmoid function** is used for binary classification.

### 13. Then which function is used?
For multi-class classification, the **Softmax function** is used.

### 14. What is Area under the curve?
**Area Under the Curve (AUC)** refers to the area under the ROC curve, indicating the model's ability to distinguish between classes.

### 15. Which metric is used to split a node in Decision Tree?
**Gini Index** or **Information Gain** is used to split nodes in a decision tree.

### 16. What is ensemble learning?
**Ensemble Learning** combines multiple models to produce a better overall model. Techniques include **Bagging**, **Boosting**, and **Stacking**.

### 17. 3 situation-based questions
**Example questions may include:**
- How would you handle missing data in a dataset?
- Describe a time when you improved a model’s performance.
- How do you prioritize tasks in a machine learning project?



# Company: Legato Health Technologies

## Role: MLOps Engineer

### 1. Complete ML technical stack used in project?
The ML technical stack typically includes **data preprocessing**, **model training**, **hyperparameter tuning**, **model evaluation**, **deployment**, and **monitoring**. Common tools are **Python, TensorFlow, PyTorch, Scikit-learn, Docker, Kubernetes, and CI/CD pipelines**.

### 2. Different activation function?
Common activation functions include **ReLU**, **Sigmoid**, **Tanh**, and **Leaky ReLU**. These functions introduce non-linearity into the network, helping it to learn complex patterns.

### 3. How do you handle imbalance data?
Imbalance data can be handled using techniques like **resampling (oversampling/undersampling)**, **SMOTE (Synthetic Minority Over-sampling Technique)**, using **different evaluation metrics** (like Precision-Recall), and **algorithmic approaches** such as **cost-sensitive learning**.

### 4. Difference between sigmoid and softmax?
**Sigmoid** outputs a probability between 0 and 1 for binary classification. **Softmax** outputs a probability distribution over multiple classes, ensuring that the sum of probabilities equals 1, making it suitable for multi-class classification.

### 5. Explain about optimisers?
Optimizers like **SGD (Stochastic Gradient Descent)**, **Adam**, **RMSprop**, and **Adagrad** are used to minimize the loss function by updating the model weights iteratively. **Adam** is popular for its adaptive learning rate and faster convergence.

### 6. Precision-Recall Trade off?
Precision-Recall Trade off is a balance between **precision** (the accuracy of positive predictions) and **recall** (the ability to find all positive instances). High precision can lead to lower recall and vice versa. The trade-off is managed based on the problem requirements.

### 7. How do you handle False Positives?
False Positives can be handled by **adjusting the decision threshold**, using **ensemble methods**, and focusing on metrics like **precision** to minimize their occurrence. In some cases, **post-processing** steps are used to validate positive predictions.

### 8. Explain LSTM architecture by taking example of 2 sentences and how it will be processed?
LSTM (Long Short-Term Memory) processes sequences using **memory cells** and **gates** (input, forget, and output gates). For example, two sentences "The cat sat on the mat" and "The dog lay on the rug" will be processed word by word, maintaining context over time to capture dependencies.

### 9. Decision Tree Parameters?
Key parameters of a Decision Tree include **max depth**, **min samples split**, **min samples leaf**, **max features**, and **criterion** (like Gini or Entropy) which control the tree's growth and decision rules.

### 10. Bagging and boosting?
**Bagging** (Bootstrap Aggregating) reduces variance by training multiple models on different subsets of data and averaging their predictions. **Boosting** reduces bias by training models sequentially, each focusing on correcting errors of the previous model.

### 11. Explain bagging internals
Bagging involves creating multiple subsets of the original data through **bootstrapping** (random sampling with replacement), training individual models on these subsets, and then **aggregating their predictions** (e.g., by voting for classification or averaging for regression).

### 12. Write a program by taking an URL and give a rough code approach how you will pass payload and make a post request?
```python
import requests

url = 'https://example.com/api'
payload = {'key1': 'value1', 'key2': 'value2'}

response = requests.post(url, json=payload)

print(response.status_code)
print(response.json())
```

### 13. Different modules used in Python?
Common Python modules include NumPy (numerical operations), Pandas (data manipulation), Matplotlib and Seaborn (data visualization), Scikit-learn (machine learning), and TensorFlow/PyTorch (deep learning).

### 14. Another coding problem of checking balanced parentheses?
```python
def is_balanced(s):
    stack = []
    brackets = {'(': ')', '{': '}', '[': ']'}
    for char in s:
        if char in brackets:
            stack.append(char)
        elif char in brackets.values():
            if not stack or brackets[stack.pop()] != char:
                return False
    return not stack

# Example usage
print(is_balanced("({[]})"))  # Output: True
print(is_balanced("({[})"))   # Output: False
```



# Company: Legato Health Technologies

## Role: MLOps Engineer

### 1. Complete ML technical stack used in project?
A complete ML technical stack typically includes **data collection**, **data processing**, **model training**, **model deployment**, and **monitoring**. Tools like **TensorFlow**, **PyTorch**, **Docker**, **Kubernetes**, and **CI/CD pipelines** are commonly used.

### 2. Different activation function?
Different activation functions include **ReLU**, **Sigmoid**, **Tanh**, and **Leaky ReLU**. Each has specific properties that affect how neural networks learn and make predictions.

### 3. How do you handle imbalance data?
Imbalanced data can be handled using techniques like **resampling (over-sampling and under-sampling)**, **SMOTE (Synthetic Minority Over-sampling Technique)**, and **using different evaluation metrics like Precision-Recall**.

### 4. Difference between sigmoid and softmax?
The **sigmoid** function outputs a probability between **0 and 1** and is used for **binary classification**. The **softmax** function outputs a probability distribution over multiple classes and is used for **multi-class classification**.

### 5. Explain about optimisers?
Optimizers like **SGD (Stochastic Gradient Descent)**, **Adam**, and **RMSprop** are algorithms used to **minimize the loss function** and **update the model's weights** during training.

### 6. Precision-Recall Trade off?
The Precision-Recall trade-off involves balancing between **precision (minimizing false positives)** and **recall (minimizing false negatives)**. **High precision** indicates fewer false positives, while **high recall** indicates fewer false negatives.

### 7. How do you handle False Positives?
False positives can be handled by **adjusting the decision threshold**, using **more representative training data**, and employing techniques like **cost-sensitive learning**.

### 8. Explain LSTM architecture by taking example of 2 sentences and how it will be processed?
LSTM (Long Short-Term Memory) processes sequences by maintaining a **memory cell**. For two sentences, it will process each word sequentially, updating its **hidden state** and **cell state** to capture the context and dependencies between words.

### 9. Decision Tree Parameters?
Decision Tree parameters include **max_depth**, **min_samples_split**, **min_samples_leaf**, and **criterion** (e.g., Gini impurity or entropy) which control the tree's structure and splitting criteria.

### 10. Bagging and boosting?
**Bagging** (Bootstrap Aggregating) and **boosting** are ensemble techniques. Bagging reduces variance by combining multiple models trained on different subsets of data. Boosting reduces bias by sequentially training models, each focusing on the errors of the previous one.

### 11. Explain bagging internals?
Bagging involves creating multiple subsets of the original dataset through **random sampling with replacement**. Each subset is used to train a model, and the final prediction is made by **aggregating** (e.g., majority voting or averaging) the predictions of all models.

### 12. Write a program by taking an url and give a rough code approach how you will pass payload and make a post request?
```python
import requests

url = "http://example.com/api"
payload = {"key1": "value1", "key2": "value2"}

response = requests.post(url, json=payload)

print(response.status_code)
print(response.json())
```


### 13. Different modules used in python?
Common Python modules include NumPy, Pandas, scikit-learn, matplotlib, TensorFlow, PyTorch, requests, and Flask.

### 14. Another coding problem of checking balanced parentheses?
```python
def is_balanced(s):
    stack = []
    mapping = {")": "(", "]": "[", "}": "{"}
    
    for char in s:
        if char in mapping:
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:
            stack.append(char)
    
    return not stack

# Example usage
print(is_balanced("()[]{}"))  # True
print(is_balanced("(]"))      # False
```



# Company: Cerence

## Role: NLU Developer

### 1. Write a function that takes two strings as inputs and returns true if they are anagrams of each other and false otherwise.
**Example**:
```python
def are_anagrams(str1, str2):
    return sorted(str1) == sorted(str2)

# Examples:
print(are_anagrams("hello", "hlleo"))  # Output: True
print(are_anagrams("hello", "helo"))   # Output: False
```

### 2. Write a function that takes an array of strings "A" and an integer "n", and returns the list of all strings of length "n" from the array "A" that can be constructed as the concatenation of two strings from the same array "A".
Example:
```python
def find_concatenated_strings(A, n):
    result = []
    set_A = set(A)
    for s in A:
        if len(s) == n:
            for i in range(1, n):
                if s[:i] in set_A and s[i:] in set_A:
                    result.append(s)
                    break
    return result

# Examples:
A = ["dog", "tail", "sky", "or", "hotdog", "tailor", "hot"]
n = 6
print(find_concatenated_strings(A, n))  # Output: ["hotdog", "tailor"]
```

### 3. Given an array "arr" of numbers and a starting number "x", find "x" such that the running sums of "x" and the elements of the array "arr" are never lower than 1.
Example:
```python
def find_starting_x(arr):
    min_sum = 0
    current_sum = 0
    for num in arr:
        current_sum += num
        min_sum = min(min_sum, current_sum)
    return 1 - min_sum

# Examples:
arr = [-2, 3, 1, -5]
print(find_starting_x(arr))  # Output: 4
```

# Company: GEOTAB

## Role: Python

### 1. Is Python a language that follows pass by value, or pass by reference or pass by object reference?
Python uses **pass-by-object-reference**. This means that **mutable objects** (like lists) can be changed in place, while **immutable objects** (like integers) cannot.

### 2. What are lambda functions and how to use them?
**Lambda functions** are **anonymous** functions defined using the `lambda` keyword. They are useful for **short, throwaway functions** that are not reused. Example: `lambda x: x + 1` defines a function that increments its input by 1.

### 3. Difference between mutable and immutable objects with example.
**Mutable objects** can be changed after creation (e.g., **lists**). Example: `list1 = [1, 2, 3]; list1[0] = 10` changes the first element to 10. **Immutable objects** cannot be changed (e.g., **tuples**). Example: `tuple1 = (1, 2, 3); tuple1[0] = 10` raises an error.

### 4. What are Python decorators? Why do we use them?
**Decorators** are functions that **modify the behavior of other functions** or methods. They are used for **logging, access control, memoization**, and more. Example: `@decorator_function` above a function definition modifies its behavior.

## Role: SQL

### 1. What is the difference between Inner join and left inner join?
An **Inner Join** returns only the **matching rows** from both tables. A **Left Inner Join** (or **Left Join**) returns all rows from the **left table** and the matching rows from the right table. If no match is found, NULLs are returned for columns from the right table.

### 2. What are window functions?
**Window functions** perform calculations across a set of table rows related to the current row. They allow for **running totals, moving averages**, and **ranking** without collapsing rows into a single output row.

### 3. What is the use of groupby?
The **GROUP BY** clause groups rows sharing a property so that an **aggregate function** (like SUM, COUNT, AVG) can be applied to each group. It is used to perform **summarized calculations** on data subsets.



# Role: MLOps Engineer

## 1st Round

### 1. Introduction
Provide a brief introduction about yourself, your background, and your experience relevant to the role of an MLOps Engineer.

### 2. Current NLP architecture used in my project
Explain the **NLP architecture** you are currently using, including the **models**, **libraries**, and **tools** involved, and how they integrate to solve your project's requirements.

### 3. How will you identify Data Drift? Once identified how would you automate the handling of Data Drift
Identify data drift by monitoring **statistical properties** of input data and model predictions. Automate handling by implementing **data validation checks**, **alert systems**, and **retraining pipelines**.

### 4. Data Pipeline used
Describe the data pipeline, including **data ingestion**, **processing**, **storage**, **feature extraction**, and **model training/serving** stages.

### 5. Fasttext word embedding vs word2vec
**FastText** considers subword information, making it better for handling **out-of-vocabulary words**. **Word2Vec** learns vector representations for **individual words**.

### 6. When should we use Tf-IDF and when predictive based word embedding will be advantageous over Tf-IDF
Use **TF-IDF** for simpler tasks like **text classification**. Predictive-based embeddings like **Word2Vec** or **FastText** are better for tasks requiring **semantic understanding**, such as **text similarity**.

### 7. Metrics used to validate our model
Use metrics like **accuracy**, **precision**, **recall**, **F1-score**, and **AUC-ROC** to validate the model, depending on the problem type (classification, regression).

### 8. In MongoDB write a query to find employee names from a collection
```mongodb
db.collection.find({}, { name: 1, _id: 0 })
```

### 9. In Python write a program to separate 0s and 1s from an array- (0,1,0,1,1,0,1,0)
```python 
def separate_zeros_ones(arr):
    zeros = [x for x in arr if x == 0]
    ones = [x for x in arr if x == 1]
    return zeros + ones

arr = [0, 1, 0, 1, 1, 0, 1, 0]
print(separate_zeros_ones(arr))
```




# Company: Latentview
Initial they had asked for the explaining the project which I had done. I explained the Customer prediction case. Then I was asked with python questions by sharing my screen.

### 1. How do you handle the correlated variables without removing them
Use techniques like Principal Component Analysis (PCA) or Factor Analysis to reduce dimensionality and handle correlated variables without removing them.

### 2. Explain the SMOTE, ADAYSN technique
SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic samples for the minority class to balance class distribution. ADASYN (Adaptive Synthetic Sampling) adjusts the number of synthetic samples based on data density.

### 3. What is stratified sampling technique
Stratified sampling divides the data into strata or groups based on certain characteristics and samples from each group to ensure representative distribution.

### 4. Explain the working of random forest and xgboost
Random Forest is an ensemble method using multiple decision trees to improve performance and reduce overfitting. XGBoost is a gradient boosting framework that builds additive models in a stage-wise fashion, optimizing for speed and accuracy.

### 5. How do you optimise the Recall of your output
To optimize Recall, adjust the decision threshold, use class weighting, and implement resampling techniques like oversampling or undersampling.

### 6. What are chi-square and ANOVA test
Chi-square test checks for independence between categorical variables. ANOVA (Analysis of Variance) tests for differences between means of three or more groups.

### 7. In python they asked for LOC,ILOC, how do you remove duplicate, how to get unique values in column
LOC: Access a group of rows and columns by labels or a boolean array.
ILOC: Access a group of rows and columns by integer positions.
Remove duplicates: df.drop_duplicates()
Get unique values: df['column'].unique()

### 8. In SQL they asked for the query for having matches between different teams
```python
SELECT team1, team2, COUNT(*)
FROM matches
GROUP BY team1, team2
HAVING COUNT(*) > 1;
```

# Company: Enquero Global

## Role: Data Scientist

### 1. Previous job role and responsibilities
I previously worked as a **Data Analyst**, where I was responsible for **data collection**, **preprocessing**, **EDA**, **feature engineering**, and **model building**. My primary role involved **analyzing data** to derive actionable insights and support decision-making processes.

### 2. Problem statement of your project and How do you overcome challenges
In my recent project, I tackled the challenge of **predicting customer churn**. The primary challenge was handling **imbalanced data**. I used techniques like **SMOTE** for oversampling the minority class and **precision-recall curves** to select the optimal threshold for classification.

### 3. How do you handle feature which had many categories
For features with many categories, I use techniques like **one-hot encoding**, **target encoding**, or **embedding** methods depending on the dataset size and model requirements. This helps in effectively representing categorical variables.

### 4. When to use precision and recall
**Precision** is preferred when the cost of **false positives** is high, while **recall** is crucial when the cost of **false negatives** is high. In an imbalanced dataset, recall might be more critical to capture the minority class effectively.

### 5. What are outliers & how do you handle them
Outliers are data points that significantly differ from other observations. They can be handled using methods like **z-score**, **IQR method**, or **transformations**. Sometimes, domain knowledge is used to decide whether to **remove or cap** them.

### 6. Joins, self joins, said me to write SQL queries on self joins
**Self joins** are used to join a table with itself to compare rows within the same table. Example SQL query:
```sql
SELECT a.id, a.name, b.name AS manager_name
FROM employees a
JOIN employees b ON a.manager_id = b.id;
```

### 7. How good your with python
I have strong proficiency in Python, with extensive experience in libraries like pandas, numpy, scikit-learn, and tensorflow. I use Python for data analysis, machine learning, and building end-to-end solutions.

### 8. Logic for reverse string
To reverse a string in Python:
```python
def reverse_string(s):
    return s[::-1]
```



# Company: Deloitte

## Role: Data Scientist

### Candidate Name: Wanted to remain anonymous

## ROUND 1:

### 1. Data collection - How do you collect data and data preprocessing?
Data collection is done through **surveys, APIs, web scraping**, and **databases**. Data preprocessing involves **cleaning**, **normalization**, **handling missing values**, and **transformations** to ensure the data is ready for analysis.

### 2. Focused on EDA part
Exploratory Data Analysis (EDA) involves **summarizing** the main characteristics of the data, often using **visualization** methods. This includes **detecting outliers**, **identifying patterns**, and **understanding relationships** between variables.

### 3. Have you deployed any project in the cloud? If yes, which cloud did you use and how did you do that?
Yes, I have deployed projects on **AWS**. The process includes **setting up EC2 instances**, **configuring S3 buckets for storage**, and using **services like Lambda and RDS** for compute and database needs.

### 4. How do you interact with domain experts and business analytics people?
I interact with domain experts and business analysts through **regular meetings**, **workshops**, and **collaborative tools** to gather requirements, understand business needs, and validate the results.

### 5. How do you replace missing values for continuous variables?
Missing values for continuous variables can be replaced using **mean, median, or mode** depending on the distribution. For skewed distributions, **median** is preferred.

### 6. What is the largest dataset you have handled till now and what was the size of the dataset?
The largest dataset I have handled was around **500GB**, comprising **millions of records** from various sources.

### 7. A continuous variable is having missing values, so how will you decide that the missing values should be imputed by mean or median?
If the data is **normally distributed**, I will use the **mean**. For **skewed distributions**, I will use the **median** to avoid bias.

### 8. What is PCA and what does each component mean? Also, what is the maximum value for the number of components?
**Principal Component Analysis (PCA)** is a dimensionality reduction technique. Each component represents a **linear combination** of the original variables that maximize the variance. The maximum number of components is equal to the **number of original features**.

### 9. What is a test of independence? How do you calculate the Chi-square value?
A test of independence, like the **Chi-square test**, evaluates if there is a significant association between two categorical variables. The **Chi-square value** is calculated by comparing the observed and expected frequencies.

### 10. When is precision preferred over recall or vice-versa?
**Precision** is preferred when the cost of **false positives** is high. **Recall** is preferred when the cost of **false negatives** is high.

### 11. Advantages and disadvantages of Random Forest over Decision Tree?
**Advantages** of Random Forest include **higher accuracy**, **resistance to overfitting**, and **robustness**. **Disadvantages** include **higher computational cost** and **lack of interpretability**.

### 12. What is the C hyperparameter in the SVM algorithm and how does it affect the bias-variance tradeoff?
The **C hyperparameter** in SVM controls the **trade-off between achieving a low error on the training data and minimizing the margin**. A **high C** can lead to **low bias** but **high variance**, while a **low C** can lead to **high bias** but **low variance**.

### 13. What are the assumptions of linear regression?
The assumptions of linear regression are **linearity**, **independence**, **homoscedasticity**, **normality of errors**, and **no multicollinearity**.

### 14. Difference between Stemming and Lemmatization?
**Stemming** reduces words to their **base or root form**, often resulting in non-real words. **Lemmatization** reduces words to their **base or root form** considering the **context** and producing real words.

### 15. Difference between Correlation and Regression?
**Correlation** measures the **strength and direction** of a linear relationship between two variables. **Regression** predicts the **value of a dependent variable** based on the value of one or more independent variables.

### 16. What is p-value and confidence interval?
The **p-value** measures the **probability** that the observed results occurred by chance. A **confidence interval** provides a **range of values** that is likely to contain the true parameter value.

### 17. What is multicollinearity and how do you deal with it? What is VIF?
**Multicollinearity** occurs when independent variables are **highly correlated**. It can be dealt with by **removing one of the correlated variables** or using **techniques like PCA**. **Variance Inflation Factor (VIF)** measures the **degree of multicollinearity**.

### 18. What is the difference between apply, applymap, and map function in Python?
**apply()** is used on **DataFrames** to apply a function along an axis. **applymap()** is used on **DataFrames** to apply a function element-wise. **map()** is used on **Series** to map values using a **function or dictionary**.

