---
title: Example Reference
description: A reference page in my new Starlight docs site.
---

# Interview Questions Set 1

## Statistics Questions

### 1. Where have you used Hypothesis Testing in your Machine Learning Solution?
**Answer:** **Hypothesis testing** is often used to determine if a particular feature or set of features significantly impacts the target variable. For example, in a **predictive model for customer churn**, hypothesis testing can be used to evaluate if certain customer behaviors or demographics are statistically significant predictors of churn.

### 2. What kind of statistical tests have you performed in your ML Application?
**Answer:** Common statistical tests include **t-tests, chi-square tests, ANOVA, and correlation tests** to assess relationships between variables, ensure data normality, and validate model assumptions.

### 3. What do you understand by P Value? And what is its use in ML?
**Answer:** The **p-value** measures the probability that an observed difference could have occurred just by random chance. In ML, it helps to **determine the significance of features** and to test hypotheses about model parameters.

### 4. Which type of error is more severe, Type 1 or Type 2? And why, with example.
**Answer:** **Type 1 error (false positive)** is often considered more severe because it can lead to **incorrect actions** being taken. For example, falsely detecting a fraudulent transaction can cause **unnecessary customer inconvenience**.

### 5. Where can we use chi-square, and have you used this test anywhere in your application?
**Answer:** **Chi-square tests** are used for testing relationships between **categorical variables**. I have used it to test independence between categorical features in a **customer segmentation model**.

### 6. Can we use Chi-square with a Numerical dataset? If yes, give an example. If no, give a reason?
**Answer:** **No, chi-square tests** are designed for categorical data. For numerical data, other tests like **t-tests or ANOVA** are more appropriate.

### 7. What do you understand by ANOVA Testing?
**Answer:** **ANOVA (Analysis of Variance)** tests whether there are statistically significant differences between the means of three or more independent groups. It’s useful for **comparing multiple treatment effects** in an experiment.

### 8. Give me a scenario where you can use Z test and T test.
**Answer:** **Z-tests** are used when the sample size is **large (n > 30)**, for example, comparing the means of large datasets. **T-tests** are used for **smaller sample sizes (n ≤ 30)**, such as comparing test scores between two small groups.

### 9. What do you understand by inferential Statistics?
**Answer:** **Inferential statistics** involves making inferences about populations based on samples. It includes **estimating population parameters, testing hypotheses, and making predictions**.

### 10. When you are trying to calculate Std Deviation or Variance, why do you use N-1 in the Denominator? (Hint: Basel Connection)
**Answer:** Using **N-1 (Bessel’s correction)** provides an **unbiased estimate** of the population variance from a sample.

### 11. What do you understand by right skewness, Give example?
**Answer:** **Right skewness (positive skew)** indicates that the tail on the right side of the distribution is longer or fatter than the left. An example is **income distribution**, where a few people have very high incomes.

### 12. What is the difference between Normal distribution and Standard Normal Distribution and Uniform Distribution?
**Answer:** **Normal distribution** is a bell-shaped curve defined by its mean and standard deviation. **Standard Normal Distribution** is a normal distribution with a **mean of 0 and a standard deviation of 1**. **Uniform distribution** has constant probability over an interval.

### 13. What are the different kinds of Probabilistic distributions you heard of?
**Answer:** Common distributions include **normal, binomial, Poisson, uniform, exponential, and beta distributions**.

### 14. What do you understand by symmetric dataset?
**Answer:** A **symmetric dataset** has a distribution where the **left and right sides are mirror images**. The **mean, median, and mode are equal**.

### 15. In your last project, were you using symmetric data or Asymmetric Data? If it’s asymmetric, what kind of EDA have you performed?
**Answer:** I used **asymmetric data** and performed EDA techniques such as **transformation (e.g., log transformation)** and **visualization (e.g., histograms)** to understand and handle skewness.

### 16. Can you please tell me the formula for skewness?
**Answer:** **Skewness = (N/(N-1)(N-2)) * Σ((Xi - X̄)/σ)^3**

### 17. Have you applied Student T distribution Anywhere?
**Answer:** Yes, in **small sample hypothesis testing** to determine if there are significant differences between two groups.

### 18. What do you understand by statistical analysis of data, Give me a scenario where you have used statistical analysis in the last projects?
**Answer:** **Statistical analysis** involves collecting, exploring, and presenting large amounts of data to discover underlying patterns and trends. For example, I used it in **A/B testing** to compare user engagement metrics between two versions of a web page.

### 19. Can you please tell me the criterion to apply binomial distribution, with an example?
**Answer:** **Binomial distribution** is applied when there are **fixed number of trials**, each trial has **two possible outcomes**, and the **probability of success is constant**. For example, **flipping a coin 10 times** and counting the number of heads.

### 20. There are 100 people, who are taking this particular 30-day Data science interview preparation course, what is the probability that 10 people will be able to make a transition in 1 week? If 50 people were able to make a transition in 3 weeks? (Hint: Poisson Distribution)
**Answer:** This can be modeled using the **Poisson distribution**. Given λ (average rate) is **50/3 weeks = ~16.67 people/week**, the probability can be calculated using the Poisson formula **P(X=k) = (e^(-λ) * λ^k) / k!**.

### 21. Let's suppose I have appeared in 3 interviews, what is the probability that I am able to crack at least 1 interview?
**Answer:** If p is the probability of cracking an interview, the probability of **not cracking any** is **(1-p)^3**. Therefore, the probability of **cracking at least one** is **1 - (1-p)^3**.

### 22. Explain Gaussian Distribution in your own way.
**Answer:** **Gaussian distribution**, or **normal distribution**, is a **symmetric, bell-shaped curve** where most of the observations cluster around the central peak, and probabilities for values taper off equally on both sides.

### 23. What do you understand by 1st, 2nd, and 3rd Standard Deviation from Mean?
**Answer:** These represent intervals within the normal distribution where approximately **68%, 95%, and 99.7%** of the data points lie, respectively.

### 24. What do you understand by variance in data in simple words?
**Answer:** **Variance** measures how much the **data points in a dataset differ from the mean**. High variance means data points are spread out; low variance means they are clustered close to the mean.

### 25. If variance of dataset is too high, how will you handle it or decrease it?
**Answer:** High variance can be handled by **data normalization, reducing dimensionality,** or using **regularization techniques** in models.

### 26. Explain the relationship between Variance and Bias.
**Answer:** **Variance** measures the model's sensitivity to changes in the training data, while **bias** measures the error introduced by approximating real-world problems. There is often a **trade-off between bias and variance** in model performance.

### 27. Tell me what kind of graph-based approach I will be able to apply to find out standardization of Dataset?
**Answer:** You can use **box plots** to visualize the distribution and identify outliers or **z-score plots** to standardize and compare features.

### 28. What do you understand by Z Value given in Z Table?
**Answer:** **Z-value** represents the **number of standard deviations a data point is from the mean**. It’s used in hypothesis testing and confidence intervals.

### 29. Do you know a Standard Normal Distribution Formula?
**Answer:** The formula is: **Z = (X - μ) / σ**, where X is the value, μ is the mean, and σ is the standard deviation.

### 30. Can you please explain the critical region in your way?
**Answer:** The **critical region** is the range of values for the test statistic that leads to the **rejection of the null hypothesis** in hypothesis testing.

### 31. Have you used AB testing in your project so far? If yes, Explain. If no, Tell me about AB testing.
**Answer:** Yes, **A/B testing** compares two versions of a web page or app feature to determine which one performs better based on predefined metrics like **click-through rates or conversion rates**.

### 32. Can we use Alternate hypothesis as a null Hypothesis?
**Answer:** No, the **null hypothesis** typically represents the default or no effect state, while the **alternative hypothesis** represents the effect or difference we aim to detect.

### 33. Can you please explain the confusion matrix for more than 2 variables?
**Answer:** A **confusion matrix** for multi-class classification includes rows for **actual classes** and columns for **predicted classes**, with each cell representing the count of instances for corresponding actual-predicted class pairs.

### 34. Give me an example of False Negative From this interview.
**Answer:** A **false negative** in this context would be not recognizing a correct answer as correct, mistakenly identifying it as incorrect.

### 35. What do you understand by Precision, Recall and F1 Score with example?
**Answer:** **Precision** is the ratio of true positives to the total predicted positives, **recall** is the ratio of true positives to the actual positives, and **F1 score** is the harmonic mean of precision and recall. For example, in a **spam detection model**, precision is the proportion of correctly identified spam emails among all identified spam emails, recall is the proportion of correctly identified spam emails among all actual spam emails, and F1 score balances the two.

### 36. What kind of questions do you ask your client if they give you a dataset?
**Answer:** Questions about the **data source, data quality, missing values, feature definitions,** and the **business objective or problem** to be solved.

### 37. Have you ever done F test on your dataset, if yes, give example. If No, then explain F distribution?
**Answer:** Yes, I have used the **F-test** to compare model variances. **F-distribution** is used in **analysis of variance (ANOVA)** to compare statistical models.

### 38. What is AUC & ROC Curve? Explain with uses.
**Answer:** **AUC (Area Under the Curve)** measures the overall performance of a classification model. **ROC (Receiver Operating Characteristic) curve** plots true positive rate against false positive rate at various threshold settings. It’s used to **evaluate model performance**.

### 39. Who decided in your last project, what will be the accuracy of your model & what was the criterion to make the decision.
**Answer:** The project **stakeholders and data science team** collectively decided the accuracy criterion based on **business requirements and acceptable error rates**.

### 40. What do you understand by 1 tail test & 2 tail test? Give example.
**Answer:** **One-tailed tests** predict the **direction of the effect**, while **two-tailed tests** only predict that there will be an effect, regardless of direction. For example, testing if a new drug has a positive effect (one-tailed) versus testing if the drug has any effect (two-tailed).

### 41. What do you understand by power of a test?
**Answer:** The **power of a test** is the probability that it correctly **rejects a false null hypothesis** (detects an effect when there is one). High power means lower **Type II error** rate.

### 42. How do you set level of significance for your dataset?
**Answer:** The **significance level** is typically set at **0.05**, representing a 5% risk of rejecting the null hypothesis when it is true.

### 43. Have you ever used T table in any of your project so far? If No, then why is statistic important for a data scientist? If yes, explain the scenario.
**Answer:** Yes, I have used **T tables** for determining critical values in **small sample hypothesis testing** scenarios. Statistics are crucial for data scientists to **validate models and derive insights from data**.

### 44. Can we productionize a statistical model?
**Answer:** Yes, **statistical models** can be productionized by deploying them as **APIs** or integrating them into applications to make real-time predictions.

### 45. How frequently do you build the model and test it?
**Answer:** The frequency depends on the **project requirements**. In dynamic environments, models may be retrained and tested **weekly or monthly** to incorporate new data.

### 46. What are the testing techniques that you use for model testing, name some of those?
**Answer:** Techniques include **cross-validation, A/B testing, confusion matrix, ROC-AUC, precision-recall**, and various statistical tests.

### 47. What do you understand by sensitivity in dataset? Give example.
**Answer:** **Sensitivity (recall)** is the ability of a model to correctly identify **true positives**. For example, in a **medical test**, sensitivity is the proportion of correctly identified positive cases out of all actual positive cases.

### 48. Let’s suppose you are trying to solve a classification problem; how do you decide which algorithm to use? Give scenarios.
**Answer:** The choice depends on the **dataset size, feature types,** and **problem requirements**. For instance, **logistic regression** for binary classification, **decision trees** for interpretability, and **neural networks** for complex, high-dimensional data.

### 49. Can we use Logistic regression for classification if my number of classes are 5?
**Answer:** Yes, using **multinomial logistic regression**, which extends logistic regression to handle **multiple classes**.

### 50. Let’s suppose there is a company like OLA or UBER that provides service to many customers, then how will they make sure that car availability in a particular region and what kind of dataset is required?
**Answer:** The company would need **real-time data** on **demand patterns, car locations, driver availability,** and **traffic conditions**. **Predictive models** can then optimize car distribution based on this data.

### 51. AI Solution for architecture -- Let’s suppose there is an agricultural field in different areas in India, and we know soil & weather conditions are different over India, so I am trying to build a system that helps me understand what kind of treatments I will be able to apply to my crops, which crop I can grow in a particular month so I can maximize the benefit from the soil. Then what kind of algorithm will you use whether it's ML, DL, Vision? What will be your approach, and what kind of solution design will you provide?
**Answer:** I would use a combination of **machine learning** for **predictive analysis**, **deep learning** for **image recognition** (to identify crop health), and **IoT sensors** for real-time soil and weather data collection. The solution would involve a centralized system that collects data from various sources, applies predictive models, and provides recommendations on crop treatments and planting schedules.

### 52. I have a client, they are facing a problem in terms of maintaining the pipeline for water. So what kind of AI solution will you design to identify the leakage and maintenance?
**Answer:** An AI solution would involve deploying **sensors along the pipeline** to monitor flow rates and detect anomalies. **Machine learning models** could analyze sensor data to **predict and identify potential leakages**, while maintenance schedules can be optimized using **predictive maintenance algorithms**.

### 53. Let’s suppose I am building a solution for blind people. What kind of AI solution will you provide to help them interact with the system, an Affordable solution?
**Answer:** I would develop a **voice-activated system** with **natural language processing (NLP) capabilities** to allow blind users to interact with devices using **voice commands**. The system could include **text-to-speech and speech-to-text functionalities**, and integrate with affordable hardware like smartphones or smart speakers.

### 54. What is the difference between R2 and Adjusted R2?
**Answer:** **R2** measures the proportion of variance explained by the model, while **Adjusted R2** adjusts R2 for the number of predictors in the model, providing a more accurate measure when multiple predictors are used.

### 55. Where do you apply Regularization, and what kind of regularization have you applied, and why?
**Answer:** **Regularization** is applied to **prevent overfitting** in models by penalizing large coefficients. I have used **L1 (Lasso)** and **L2 (Ridge) regularization** in regression models to improve generalization.

### 56. What do you understand by multicollinearity and homoscedasticity in Dataset?
**Answer:** **Multicollinearity** occurs when independent variables in a model are **highly correlated**, making it difficult to estimate their individual effects. **Homoscedasticity** refers to **constant variance of errors** across all levels of an independent variable.

### 57. Can you please explain 1 example of Polynomial Regression and how to build a model for polynomial regression.
**Answer:** **Polynomial regression** is used when the relationship between the independent and dependent variables is **non-linear**. For example, predicting the growth of bacteria over time, where growth follows a **quadratic pattern**. The model includes polynomial terms (e.g., x^2) and can be built using **polynomial feature expansion** followed by **linear regression**.

### 58. There is some client who is intercepting a call like 3,4, or 5 people talking in a zoom call. Tell me the approach so that we can separate the voices of each and every person. (Hint: Speech Diarization)
**Answer:** The approach involves using **speech diarization techniques**, which segment audio into speaker-specific clusters. This can be achieved using **deep learning models** like neural networks that are trained to recognize and separate different speakers' voices.

### 59. In the case of a multilinear regression model, let’s suppose my number of features are 5. Can you explain what kind of line it draws? Explain.
**Answer:** A multilinear regression model with 5 features represents a **hyperplane in a 5-dimensional space**, where the predicted value is a **linear combination** of the 5 feature values.

### 60. List the number of algorithms that you know from clustering.
**Answer:** **K-means, hierarchical clustering, DBSCAN, Gaussian Mixture Models, Mean Shift,** and **Agglomerative Clustering**.

### 61. Tell me what is evaluation techniques for clustering algorithms. List some of those.
**Answer:** Evaluation techniques include **silhouette score, Davies-Bouldin index**, and **elbow method**.

### 62. Can you please explain the random state in train & test split function.
**Answer:** **Random state** is a **seed value** used to initialize the random number generator, ensuring **reproducibility** of the train-test split.

### 63. Let's suppose the client has provided me a data, how will you evaluate that the data is fit for model building?
**Answer:** Evaluate data quality by checking for **missing values, outliers, data distribution,** and **consistency**. Perform **exploratory data analysis (EDA)** to understand relationships and ensure the data aligns with the model's assumptions.

### 64. Have you ever worked in your last project from scratch? Or you started working in the middle? If you have started working from scratch, then what kind of work were you doing? And if you have started from the middle, then what was your responsibility?
**Answer:** I have worked on projects **from scratch**, involving **initial data collection, preprocessing, model building,** and **validation**. In projects where I joined **midway**, my responsibilities included **optimizing models, refining features**, and performing **extensive validation and testing**.

### 65. What do you understand by machine learning? How will you explain ML to Kids?
**Answer:** **Machine learning** is teaching computers to **learn from data** and make decisions. To explain to kids, it’s like showing a computer many pictures of **cats and dogs**, so it can learn to tell them apart on its own.

### 66. Let's suppose there is a project which I am going to start for a client (security & surveillance project). Client requirement is like this- they want to develop a system that can detect any kind of intrusion or unwanted or unclassified entity in the region. 1. What kind of solution will you provide to solve this requirement? 2. And what kind of feature will you be able to provide? Give a complete proposal for this solution.
**Answer:** I would propose a **computer vision-based solution** using **deep learning models** trained on images and videos to detect and classify intrusions. Features include **real-time monitoring, alert systems,** and integration with existing security infrastructure. The system would use **cameras, motion sensors,** and **AI algorithms** to analyze footage and detect suspicious activity.
