---
title: Example Reference
description: A reference page in my new Starlight docs site.
---


# Interview Questions Set 2

## ML, DL Questions

### 1. How can you define Machine Learning?
**Answer:** **Machine Learning** is a field of artificial intelligence that enables computers to **learn from data** and make predictions or decisions without being explicitly programmed.

### 2. What do you understand by a Labelled training dataset?
**Answer:** A **labelled training dataset** consists of input data paired with the correct output. It is used to **train supervised learning models**.

### 3. What are the 2 most common supervised ML tasks you have performed so far?
**Answer:** The two most common supervised ML tasks are **classification** and **regression**.

### 4. What kind of Machine learning algorithm would you use to walk a robot in various unknown areas?
**Answer:** **Reinforcement Learning** algorithms can be used to teach a robot to navigate unknown areas by learning from **trial and error**.

### 5. What kind of ML algorithm can you use to segment your users into multiple groups?
**Answer:** **Clustering algorithms** such as **K-means** or **Hierarchical Clustering** can be used to segment users into groups based on their characteristics.

### 6. What type of learning algorithm relies on similarity measures to make a prediction?
**Answer:** **K-Nearest Neighbors (KNN)** relies on similarity measures to make predictions.

### 7. What is an online learning system?
**Answer:** An **online learning system** updates its model incrementally as new data becomes available, rather than being trained on a batch of data all at once.

### 8. What is out-of-core learning?
**Answer:** **Out-of-core learning** is a technique for training models on data that cannot fit into a computer’s main memory, by processing the data in small chunks.

### 9. Can you name a couple of ML challenges that you have faced?
**Answer:** **Dealing with missing data**, **handling imbalanced datasets**, **feature selection**, **overfitting**, and **scalability** are some common challenges faced in ML projects.

### 10. Can you please give one example of hyperparameter tuning with respect to some classification algorithm?
**Answer:** **Grid Search** and **Random Search** are techniques used for hyperparameter tuning. For example, tuning the **C parameter in SVM** to find the optimal value that maximizes model performance.

### 11. What is out-of-bag evaluation?
**Answer:** **Out-of-bag (OOB) evaluation** is a method used in ensemble learning, specifically in **Random Forests**, to estimate the performance of the model without needing a separate validation set.

### 12. What do you understand by hard & soft voting classifier?
**Answer:** **Hard voting** classifier predicts the class that **receives the majority of votes** from the ensemble models. **Soft voting** classifier predicts the class based on the **average probability** estimates of the ensemble models.

### 13. Let’s suppose your ML algorithm is taking 5 minutes to train, how will you bring down the time to 5 seconds for training? (Hint: Distributed Computation)
**Answer:** **Distributed computation** using frameworks like **Apache Spark** or **Dask** can parallelize the training process, significantly reducing the training time.

### 14. Let’s suppose I have trained 5 different models with the same training dataset & all of them have achieved 95% precision. Is there any chance that you can combine all these models to get better results? If yes, how? If no, why?
**Answer:** Yes, by using **ensemble techniques** such as **stacking** or **bagging**, you can combine the models to potentially achieve better performance.

### 15. What do you understand by Gradient Descent? How will you explain Gradient Descent to a kid?
**Answer:** **Gradient Descent** is an optimization algorithm used to minimize the error in a model by iteratively adjusting the parameters. To explain to a kid: Imagine you are trying to find the lowest point in a valley. You take small steps downhill (gradient descent) until you reach the bottom (minimum error).

### 16. Can you please explain the difference between regression & classification?
**Answer:** **Regression** predicts a **continuous value** (e.g., predicting house prices), while **classification** predicts a **discrete label** (e.g., classifying emails as spam or not spam).

### 17. Explain a clustering algorithm of your choice.
**Answer:** **K-means clustering** partitions the dataset into **K clusters** where each data point belongs to the cluster with the nearest mean, minimizing within-cluster variance.

### 18. How can you explain ML, DL, NLP, Computer Vision & Reinforcement Learning with examples in your own terms?
**Answer:** 
- **Machine Learning (ML):** Teaching computers to learn patterns from data, e.g., email spam detection.
- **Deep Learning (DL):** Using neural networks with many layers to learn complex patterns, e.g., image recognition.
- **Natural Language Processing (NLP):** Enabling computers to understand human language, e.g., language translation.
- **Computer Vision:** Teaching computers to interpret and understand visual information, e.g., facial recognition.
- **Reinforcement Learning:** Learning optimal actions through trial and error, e.g., a robot learning to walk.

### 19. How can you explain semi-supervised ML in your own way with an example?
**Answer:** **Semi-supervised ML** uses a combination of **labelled** and **unlabelled data** for training. For example, using a small number of labelled images and a large number of unlabelled images to improve image classification.

### 20. What is the difference between abstraction & generalization in your own words?
**Answer:** **Abstraction** is the process of simplifying complex reality by focusing on the essential details. **Generalization** is the ability of a model to perform well on new, unseen data.

### 21. What are the steps that you have followed in your last project to prepare the dataset?
**Answer:** Steps include **data collection, cleaning, transformation, normalization, feature extraction,** and **splitting into training and testing sets**.

### 22. In your last project what steps were involved in model selection procedure?
**Answer:** Steps involved **evaluating different models** using cross-validation, **comparing performance metrics**, and **selecting the best model** based on criteria such as accuracy, precision, recall, and F1-score.

### 23. If I give you 2 columns of any dataset, what will be the steps involved to check the relationship between those 2 columns?
**Answer:** Steps include **visualizing the data** using scatter plots, calculating **correlation coefficients**, and performing **statistical tests** such as Pearson’s or Spearman’s correlation test.

### 24. Can you please explain 5 different strategies at least to handle missing values in a dataset?
**Answer:** 
1. **Deletion**: Remove rows or columns with missing values.
2. **Imputation**: Replace missing values with mean, median, or mode.
3. **Prediction**: Use a model to predict missing values.
4. **Interpolation**: Estimate missing values based on surrounding data.
5. **Using algorithms** that support missing values, such as decision trees.

### 25. What kind of different issues have you faced with respect to your raw data? At least mention 5 issues.
**Answer:** 
1. **Missing values**
2. **Outliers**
3. **Inconsistent data formats**
4. **Duplicate records**
5. **Imbalanced classes**

### 26. What is your strategy to handle categorical datasets? Explain with an example.
**Answer:** Strategies include **one-hot encoding**, **label encoding**, and **binary encoding**. For example, converting a categorical feature like "color" (red, blue, green) into binary vectors using one-hot encoding.

### 27. How do you define a model in terms of machine learning or in your own words?
**Answer:** A **model** in machine learning is a **mathematical representation** of a real-world process that maps input data to output predictions based on learned patterns.

### 28. What do you understand by k-fold validation & in what situation have you used k-fold cross-validation?
**Answer:** **K-fold cross-validation** is a technique where the dataset is divided into **K subsets**, and the model is trained and validated **K times**, each time using a different subset as the validation set and the remaining as the training set. It is used to **evaluate model performance** and **reduce overfitting**.

### 29. What is the meaning of bootstrap sampling? Explain in your own words.
**Answer:** **Bootstrap sampling** involves **randomly sampling** with replacement from a dataset to create **multiple smaller datasets**, which are then used to estimate model performance.

### 30. What do you understand by underfitting & overfitting of a model with an example?
**Answer:** **Underfitting** occurs when a model is too simple and fails to capture the underlying patterns in the data. **Overfitting** occurs when a model is too complex and captures noise in the data. For example, a linear model underfits a complex dataset, while a highly complex model overfits it.

### 31. What is the difference between cross-validation and bootstrapping?
**Answer:** **Cross-validation** involves splitting the dataset into training and validation sets multiple times to assess model performance. **Bootstrapping** involves creating multiple datasets through sampling with replacement to estimate the performance and variability of a model.

### 32. What do you understand by the silhouette coefficient?
**Answer:** The **silhouette coefficient** measures how similar an object is to its own cluster compared to other clusters. It ranges from **-1 to 1**, where a higher value indicates better clustering.

### 33. What is the advantage of using the ROC Score?
**Answer:** The **ROC Score** (AUC) provides a **single metric** that evaluates the performance of a classification model across all possible classification thresholds, making it useful for comparing different models.

### 34. Explain the complete approach to evaluate your regression model.
**Answer:** 
1. **Split data** into training and testing sets.
2. **Train the model** on the training set.
3. **Evaluate performance** using metrics like **RMSE, MAE, and R-squared**.
4. **Perform cross-validation** to check consistency.
5. **Visualize residuals** to detect patterns or anomalies.

### 35. Give me an example of lazy learner and eager learner algorithms.
**Answer:** 
- **Lazy learner:** **K-Nearest Neighbors (KNN)** which does not train a model but makes predictions using the entire training dataset.
- **Eager learner:** **Decision Trees**, which build a model during the training phase and make predictions using the model.

### 36. What do you understand by the holdout method?
**Answer:** The **holdout method** involves splitting the dataset into separate **training and testing sets** to evaluate model performance.

### 37. What is the difference between predictive modelling and descriptive modelling?
**Answer:** 
- **Predictive modeling:** Focuses on predicting future outcomes based on historical data (e.g., predicting customer churn).
- **Descriptive modeling:** Summarizes and describes patterns in existing data (e.g., customer segmentation).

### 38. How have you derived a feature for model building in your last project?
**Answer:** By **analyzing domain knowledge**, using **feature engineering techniques** such as **combining existing features, creating interaction terms**, and **normalizing** data.

### 39. Explain 5 different encoding techniques.
**Answer:**
1. **One-hot encoding**
2. **Label encoding**
3. **Binary encoding**
4. **Frequency encoding**
5. **Target encoding**

### 40. How do you define some features are not important for an ML model? What strategy will you follow?
**Answer:** Use techniques like **feature importance scores** from models, **recursive feature elimination**, or **correlation analysis** to identify and remove irrelevant features.

### 41. What is the difference between Euclidean distance and Manhattan distance? Explain in simple words.
**Answer:** 
- **Euclidean distance:** The straight-line distance between two points.
- **Manhattan distance:** The sum of the absolute differences between the coordinates of two points (like navigating a grid of city blocks).

### 42. What do you understand by feature selection, transformation, engineering, and EDA? What are the steps that you have performed in each of these in detail with examples?
**Answer:**
- **Feature selection:** Identifying and selecting the most relevant features.
- **Transformation:** Scaling and normalizing data.
- **Engineering:** Creating new features from existing ones.
- **EDA:** Exploring data using visualizations and statistical methods.

### 43. What is the difference between single value decomposition (SVD) and PCA? (Hint: SVD is one of the ways to do PCA)
**Answer:** **SVD** decomposes a matrix into three matrices to identify patterns, while **PCA** uses SVD to reduce the dimensionality of data by projecting it onto principal components.

### 44. What kind of feature transformations have you done in your last project?
**Answer:** Transformations include **normalization**, **standardization**, **log transformations**, and **polynomial feature creation**.

### 45. Have you taken any external feature in any project from any 3rd party data? If yes, explain that scenario.
**Answer:** Yes, in a **sales prediction project**, I integrated **weather data** from a third-party API to enhance the model's accuracy by accounting for weather-related sales fluctuations.

### 46. If your model is overfitted, what will you do next?
**Answer:** Apply **regularization techniques** like **L1 or L2**, **simplify the model**, **prune features**, or **increase training data**.

### 47. Explain the bias-variance trade-off.
**Answer:** The **bias-variance trade-off** is a balance between **model complexity** and **generalization**. High bias leads to **underfitting**; high variance leads to **overfitting**. The goal is to find an optimal balance to minimize overall error.

### 48. What steps would you take to improve the accuracy of your model? At least mention 5 approaches. And justify why you would choose those approaches.
**Answer:** 
1. **Feature engineering**: Improve input data quality.
2. **Hyperparameter tuning**: Optimize model parameters.
3. **Ensemble methods**: Combine multiple models.
4. **Data augmentation**: Increase training data.
5. **Cross-validation**: Ensure model robustness.

### 49. Explain the process of feature engineering in the context of text categorization.
**Answer:** 
1. **Tokenization**: Split text into tokens.
2. **Removing stop words**: Eliminate common, non-informative words.
3. **Stemming/Lemmatization**: Reduce words to their base form.
4. **Vectorization**: Convert text to numerical vectors using techniques like TF-IDF.

### 50. Explain vectorization and hamming distance.
**Answer:** 
- **Vectorization**: Converting text data into numerical vectors for ML models.
- **Hamming distance**: The number of positions at which two strings of equal length differ.

### 51. Can you please explain the chain rule and its use?
**Answer:** The **chain rule** in calculus is used to differentiate composite functions. It is fundamental in **backpropagation** in neural networks, allowing the calculation of gradients for training.

### 52. What is the difference between correlation and covariance?
**Answer:** 
- **Correlation**: Measures the strength and direction of a linear relationship between two variables (normalized).
- **Covariance**: Measures the extent to which two variables change together (not normalized).

### 53. What are the sampling techniques you have used in your project?
**Answer:** Techniques include **random sampling**, **stratified sampling**, and **systematic sampling**.

### 54. Have you ever used hypothesis testing in your last project, if yes, explain how?
**Answer:** Yes, I used **hypothesis testing** to determine if changes in a feature set significantly improved model performance by comparing the means of model metrics.

### 55. In which case you will use Naive Bayes classifier and decision tree separately?
**Answer:** 
- **Naive Bayes**: When features are **conditionally independent**.
- **Decision Tree**: When you need **interpretability** and can handle **non-linear relationships**.

### 56. What is the advantage & disadvantage of the Naive Bayes classifier? Explain.
**Answer:**
- **Advantage**: Simple, fast, works well with high-dimensional data.
- **Disadvantage**: Assumes feature independence, which is rarely true in real-world data.

### 57. In the case of numerical data, what is the Naive Bayes classification equation you will use?
**Answer:** For numerical data, Naive Bayes uses the **Gaussian (Normal) distribution** to estimate probabilities: \( P(x_i | y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} e^{-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}} \).

### 58. Give me a scenario where I will be able to use a boosting classifier and regressor?
**Answer:** **Boosting** can be used in **fraud detection** where false negatives are critical, and in **house price prediction** to improve accuracy by combining weak learners.

### 59. In the case of a Bayesian classifier, what exactly does it try to learn? Define its learning procedure.
**Answer:** A **Bayesian classifier** learns the **probability distribution** of the features given each class and uses Bayes’ theorem to combine these with the prior probabilities of each class to make predictions.

### 60. Give me a situation where I will be able to use SVM instead of Logistic regression.
**Answer:** Use **SVM** when you have **high-dimensional data** or need to handle **non-linear decision boundaries** using kernel tricks.

### 61. What do you understand by RBF kernel in SVM?
**Answer:** The **RBF (Radial Basis Function) kernel** maps input features into a higher-dimensional space to handle non-linear classification problems by creating more complex decision boundaries.

### 62. Give me 2 scenarios where AI can be used to increase revenue of the travel industry.
**Answer:** 
1. **Dynamic pricing**: Using AI to adjust prices based on demand, competition, and other factors.
2. **Personalized recommendations**: AI to suggest travel packages and destinations based on user preferences and behavior.

### 63. What do you understand by leaf node in a decision tree?
**Answer:** A **leaf node** in a decision tree represents a **final decision** or **classification** and does not split any further.

### 64. What is information gain & entropy in a decision tree?
**Answer:**
- **Entropy**: Measures the impurity or randomness in a dataset.
- **Information Gain**: The reduction in entropy after a dataset is split on a feature, used to decide the best feature for splitting.

### 65. Give disadvantages of using a Decision tree.
**Answer:**
- **Prone to overfitting**
- **Sensitive to noisy data**
- **Biased towards features with more levels**

### 66. List some of the features of a random forest.
**Answer:**
- **Ensemble of decision trees**
- **Reduces overfitting**
- **Handles large datasets**
- **Feature importance estimation**

### 67. How can you avoid overfitting in a decision tree?
**Answer:**
- **Pruning**
- **Setting a maximum depth**
- **Minimum samples split**
- **Using ensemble methods like Random Forest**

### 68. Explain polynomial regression in your own way.
**Answer:** **Polynomial regression** models the relationship between the independent variable and the dependent variable as an **n-th degree polynomial**.

### 69. Explain the learning mechanism of linear regression.
**Answer:** **Linear regression** learns by fitting a line that minimizes the **sum of squared errors** between the predicted and actual values.

### 70. What is the cost function in logistic regression?
**Answer:** The **cost function** in logistic regression is the **log-loss (binary cross-entropy)** function, which measures the performance of a classification model.

### 71. What is the error function in linear regression?
**Answer:** The **error function** in linear regression is the **Mean Squared Error (MSE)**, which measures the average squared difference between the predicted and actual values.

### 72. What is the use of implementing the OLS technique with respect to a dataset?
**Answer:** The **OLS (Ordinary Least Squares)** technique is used to **estimate the parameters** of a linear regression model by minimizing the sum of squared residuals.

### 73. Explain dendrogram in your own way.
**Answer:** A **dendrogram** is a tree-like diagram that **shows the arrangement of clusters** produced by hierarchical clustering, illustrating how clusters are merged or split at various levels of similarity.

### 74. How do you measure the quality of clusters in DBSCAN?
**Answer:** **Silhouette score**, **DB index**, and **visual inspection** of clusters can be used to measure the quality of clusters in **DBSCAN**.

### 75. How do you evaluate the DBSCAN algorithm?
**Answer:** Evaluate **DBSCAN** by checking the **density-based clustering** quality, such as **noise point identification** and **correct cluster formation**.

### 76. What do you understand by market basket analysis?
**Answer:** **Market basket analysis** identifies patterns and associations between items frequently purchased together, using techniques like **association rule mining**.

### 77. Explain centroid formation technique in K-Means algorithm.
**Answer:** In **K-Means**, centroids are initialized and iteratively updated by assigning each point to the nearest centroid and recalculating the centroid as the mean of assigned points until convergence.

### 78. Have you ever used SVM regression in any of your projects? If yes, why?
**Answer:** Yes, I used **SVM regression** to handle **non-linear relationships** in a dataset where traditional linear regression models were insufficient.

### 79. Explain the concept of GINI Impurity.
**Answer:** **GINI Impurity** measures the **probability of a randomly chosen element** being **incorrectly classified** if randomly labeled according to the distribution of labels in the dataset.

### 80. Let’s suppose I have given you a dataset with 100 columns. How will you be able to control the growth of the decision tree?
**Answer:** Control growth by **pruning, setting a maximum depth, using minimum samples per split**, and **feature selection** to reduce the number of columns.

### 81. If you are using the Ada-boost algorithm & it is giving you underfitted results, what is the hyperparameter tuning you will do?
**Answer:** Increase the **number of estimators**, adjust the **learning rate**, or change the **base estimator** to a more complex model.

### 82. Explain the gradient boosting algorithm.
**Answer:** **Gradient Boosting** builds an ensemble of weak learners in a sequential manner, where each learner tries to **correct the errors** of the previous one by **minimizing a loss function** using gradient descent.

### 83. Can we use PCA to reduce the dimensionality of highly non-linear data?
**Answer:** **PCA** is mainly effective for linear data. For highly non-linear data, techniques like **Kernel PCA** or **t-SNE** are more suitable.

### 84. How do you evaluate the performance of PCA?
**Answer:** Evaluate PCA by looking at the **explained variance ratio** to determine how much variance is captured by the principal components.

### 85. Have you ever used multiple dimensionality techniques in any project? If yes, give a reason. If no, where can we use it?
**Answer:** Yes, I used both **PCA** and **t-SNE** to reduce dimensions for **visualization** and **clustering** in a complex dataset.

### 86. What do you understand by the curse of dimensionality? Explain with the help of an example.
**Answer:** The **curse of dimensionality** refers to the challenges and exponential increase in data sparsity and computational complexity as the number of features (dimensions) increases. For example, in high-dimensional spaces, distance metrics become less meaningful, affecting clustering and nearest neighbor algorithms.

### 87. What is the difference between anomaly detection and novelty detection?
**Answer:** 
- **Anomaly detection**: Identifies rare or abnormal data points within the training data.
- **Novelty detection**: Detects new or previously unseen data points not present in the training set.

### 88. Explain Gaussian mixture model.
**Answer:** A **Gaussian Mixture Model (GMM)** is a probabilistic model that assumes the data is generated from a mixture of several Gaussian distributions with unknown parameters.

### 89. Give me a list of 10 activation functions with explanation.
**Answer:**
1. **Sigmoid**: S-shaped curve; outputs between 0 and 1.
2. **Tanh**: Outputs between -1 and 1; zero-centered.
3. **ReLU**: Rectified Linear Unit; outputs the input if positive, else zero.
4. **Leaky ReLU**: Allows a small, non-zero gradient when the input is negative.
5. **Parametric ReLU**: A variant of ReLU where the negative slope is learned.
6. **ELU**: Exponential Linear Unit; outputs negative values for negative inputs.
7. **Softmax**: Converts logits to probabilities; used in multi-class classification.
8. **Swish**: Sigmoid-weighted linear unit; smooth and non-monotonic.
9. **Mish**: Similar to Swish but with better performance on some tasks.
10. **Hard Sigmoid**: An approximation of the sigmoid function; computationally cheaper.

### 90. Explain neural network in terms of mathematical function.
**Answer:** A **neural network** can be seen as a **composition of linear and non-linear functions**. Each layer performs a linear transformation followed by a non-linear activation function.

### 91. Can you please correlate a biological neuron and artificial neuron?
**Answer:** 
- **Biological neuron**: Receives signals through dendrites, processes them in the cell body, and sends output through the axon.
- **Artificial neuron**: Receives inputs (features), processes them using weights and biases, applies an activation function, and produces an output.

### 92. Give a list of cost functions you heard of, with explanation.
**Answer:**
1. **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
2. **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values.
3. **Cross-Entropy Loss**: Measures the performance of a classification model by comparing predicted probabilities and actual labels.
4. **Hinge Loss**: Used for training classifiers like SVM, focusing on the margin between classes.
5. **Huber Loss**: Combines the advantages of MSE and MAE; less sensitive to outliers.

### 93. Can I solve the problem of classification with tabular data in a neural network?
**Answer:** Yes, neural networks can be used for classification tasks with tabular data by **properly pre-processing the data** and choosing appropriate network architecture.

### 94. What do you understand by backpropagation in a neural network?
**Answer:** **Backpropagation** is a method used to **calculate gradients** of the loss function with respect to the network's weights, enabling the optimization of weights using gradient descent.

### 95. Why do we need a neural network instead of straightforward mathematical equations?
**Answer:** **Neural networks** can model **complex, non-linear relationships** that straightforward mathematical equations may not capture.

### 96. What are the different weight initialization techniques you have used?
**Answer:**
1. **Random Initialization**
2. **Xavier Initialization**
3. **He Initialization**
4. **LeCun Initialization**

### 97. Can you visualize a neural network? If yes, provide the name of the software we can use?
**Answer:** Yes, tools like **TensorBoard**, **Netron**, and **Keras Visualization Toolkit** can be used to visualize neural networks.

### 98. How will you explain training of a neural network?
**Answer:** Training a neural network involves:
1. **Forward propagation**: Calculating the output of the network.
2. **Calculating loss**: Measuring the difference between predicted and actual values.
3. **Backpropagation**: Computing gradients of the loss with respect to weights.
4. **Weight update**: Adjusting weights using an optimization algorithm like gradient descent.

### 99. Can you please explain the difference between the sigmoid & tanh function?
**Answer:**
- **Sigmoid**: Outputs values between 0 and 1; prone to vanishing gradients.
- **Tanh**: Outputs values between -1 and 1; zero-centered, less prone to vanishing gradients.

### 100. Explain the disadvantage of using the RELU function.
**Answer:** **ReLU** can suffer from the **"dying ReLU"** problem, where neurons output zero for all inputs and stop learning.

### 101. How do you select the number of layers & number of neurons in a neural network?
**Answer:** Selection is based on **trial and error**, **empirical methods**, and **hyperparameter tuning** techniques like grid search or Bayesian optimization.

### 102. Have you ever designed any Neural network architecture by yourself?
**Answer:** Yes, I designed custom architectures for specific tasks, considering factors like data complexity, model interpretability, and computational resources.

### 103. Can you please explain the SWISS Function?
**Answer:** **SWISS** function is a type of activation function designed to address specific challenges in neural network training, though it is not a standard term. Please verify the term.

### 104. What is learning rate in laymen's terms and how do you control the learning rate?
**Answer:** The **learning rate** determines how quickly or slowly a model learns. It controls the size of the steps taken during optimization. Use techniques like **learning rate schedules** or **adaptive learning rates** (e.g., Adam optimizer).

### 105. What is the difference between batch, mini-batch & stochastic gradient descent?
**Answer:**
- **Batch Gradient Descent**: Uses the entire dataset to calculate gradients.
- **Mini-batch Gradient Descent**: Uses small subsets (mini-batches) of data.
- **Stochastic Gradient Descent**: Uses one data point at a time.

### 106. What do you understand by batch size while training a Neural Network with an example?
**Answer:** **Batch size** is the number of training examples used in one iteration of model training. For example, with a batch size of 32, the model updates its weights after processing 32 examples.

### 107. Explain 5 best optimizers you know with mathematical explanation.
**Answer:**
1. **SGD (Stochastic Gradient Descent)**
2. **Adam (Adaptive Moment Estimation)**
3. **RMSprop (Root Mean Square Propagation)**
4. **Adagrad (Adaptive Gradient Algorithm)**
5. **Adadelta**

### 108. Can you build a Neural network without using any library? If yes, prove it.
**Answer:** Yes, by implementing the **forward propagation, backpropagation**, and **weight update** steps manually using a programming language like Python.

### 109. What is the use of biases in a neural network?
**Answer:** **Biases** allow the model to have **flexibility** and fit the data better by providing additional parameters that can shift the activation function.

### 110. How do you do hyper-parameter tuning for a neural network?
**Answer:** Use techniques like **grid search, random search, Bayesian optimization**, and **hyperband** to find the best hyperparameters.

### 111. What kind of regularization have you used with respect to a neural network?
**Answer:** Techniques like **L2 regularization (Ridge), L1 regularization (Lasso), dropout**, and **batch normalization**.

### 112. What are the libraries you have used for neural network implementation?
**Answer:** **TensorFlow, Keras, PyTorch, Theano,** and **CNTK**.

### 113. What do you understand by a custom layer and a custom model?
**Answer:** A **custom layer** is a user-defined layer with specific operations, and a **custom model** is a neural network model created from scratch to meet specific requirements.

### 114. How do you implement differentiation using TensorFlow or Pytorch library?
**Answer:** Use **automatic differentiation** provided by libraries (e.g., `tf.GradientTape` in TensorFlow or `autograd` in PyTorch) to compute gradients.

### 115. What is the meaning of epoch in simple terms?
**Answer:** An **epoch** is one complete pass through the entire training dataset.

### 116. What do you understand by a TensorFlow record?
**Answer:** A **TensorFlow record** is a binary file format used to efficiently store large datasets for training machine learning models.

### 117. Explain the technique for doing data augmentation in deep learning.
**Answer:** **Data augmentation** involves generating new training samples by applying random transformations such as **rotation, flipping, scaling, cropping,** and **color jittering** to the original data.

### 118. List down different CNN networks you heard of.
**Answer:** **LeNet, AlexNet, VGG, ResNet, Inception, MobileNet, DenseNet**.

### 119. List down the names of object detection algorithms you know.
**Answer:** **YOLO (You Only Look Once), SSD (Single Shot MultiBox Detector), Faster R-CNN, R-FCN (Region-based Fully Convolutional Networks)**.

### 120. What is the difference between object detection and classification?
**Answer:** **Object detection** identifies and locates objects within an image, while **classification** assigns a label to the entire image.

### 121. List down major tasks we perform in CNN.
**Answer:** 
1. **Convolution**
2. **Pooling**
3. **Activation**
4. **Normalization**
5. **Fully connected layer**

### 122. List down algorithms for segmentation.
**Answer:** **UNet, Mask R-CNN, SegNet, DeepLab, FCN (Fully Convolutional Networks)**.

### 123. Which algorithm can you use to track a football in a football match?
**Answer:** **Object tracking algorithms** such as **SORT (Simple Online and Realtime Tracking)** or **Deep SORT**.

### 124. If I give you satellite image data, which algorithm will you use to identify images from those image data?
**Answer:** **Convolutional Neural Networks (CNNs)** and **Transfer Learning** with pre-trained models like **ResNet** or **VGG**.

### 125. Which algorithm will you use for PCB fault detection?
**Answer:** **Convolutional Neural Networks (CNNs)** for image-based PCB fault detection.

### 126. What do you understand by a pre-trained model?
**Answer:** A **pre-trained model** is a model trained on a large benchmark dataset, which can be fine-tuned on a smaller dataset for a specific task.

### 127. Explain different types of transfer learning.
**Answer:** 
1. **Feature extraction**: Using pre-trained model features as input to a new model.
2. **Fine-tuning**: Training a pre-trained model further on a new dataset.
3. **Domain adaptation**: Adapting a model to a new, but related, task or domain.

### 128. Explain where your CNN network will fail with an example. And where can we use an RNN network?
**Answer:** **CNNs** may fail in tasks requiring **temporal dependencies**, such as **video analysis**. **RNNs** are suitable for tasks involving sequential data like **text generation** or **speech recognition**.

### 129. Which GPU have you been using to train your object detection model?
**Answer:** **NVIDIA GPUs** such as **RTX 2080, GTX 1080, or Tesla V100**.

### 130. How much dataset have you used for this model, what was the epoch, time, and accuracy of the model?
**Answer:** The dataset size, number of epochs, training time, and accuracy depend on the specific project details and will vary accordingly.

### 131. What kind of optimization have you done for training object detection model?
**Answer:** Optimizations include **hyperparameter tuning, data augmentation, learning rate schedules,** and **model pruning**.

### 132. How do you evaluate your object detection model?
**Answer:** Using metrics like **mAP (mean Average Precision), precision-recall curves, IoU (Intersection over Union),** and **confusion matrix**.

### 133. List down algorithms for object tracking.
**Answer:** **SORT, Deep SORT, GOTURN, MOSSE,** and **Kalman Filter**.

### 134. What do you understand by FPS (frames per second)?
**Answer:** **FPS** measures the number of frames processed or displayed per second in video processing or real-time applications.

### 135. Can you please explain 2D & 3D convolution?
**Answer:** 
- **2D Convolution**: Applies convolutional filters to 2D spatial data (e.g., images).
- **3D Convolution**: Applies convolutional filters to 3D spatial data (e.g., video frames or volumetric data).

### 136. What do you understand by batch normalization?
**Answer:** **Batch normalization** normalizes the input of each layer to improve training speed, stability, and performance of deep neural networks.

### 137. Which algorithm do you use for detecting handwriting detection?
**Answer:** **Convolutional Neural Networks (CNNs)** with **LSTM layers** for sequence modeling.

### 138. Explain the SoftMax function.
**Answer:** **SoftMax** function converts logits into probabilities by exponentiating and normalizing them, used in multi-class classification tasks.

### 139. What is the disadvantage of using RNN?
**Answer:** **RNNs** suffer from the **vanishing gradient problem**, making it difficult to learn long-term dependencies.

### 140. List down at least 5 RNN architectures.
**Answer:** **Vanilla RNN, LSTM, GRU, Bi-directional RNN,** and **Deep RNN**.

### 141. Explain the architectural diagram of LSTM, also list advantages & disadvantages.
**Answer:**
- **Advantages**: Handles long-term dependencies, reduces vanishing gradient problem.
- **Disadvantages**: Computationally expensive, complex architecture.

### 142. Explain the architectural diagram of BI LSTM, also list advantages & disadvantages.
**Answer:**
- **Advantages**: Processes data in both forward and backward directions, captures more context.
- **Disadvantages**: More computationally intensive, increased complexity.

### 143. Explain the architectural diagram of stacked LSTM, also list advantages & disadvantages.
**Answer:**
- **Advantages**: Greater capacity to learn complex patterns.
- **Disadvantages**: Higher risk of overfitting, longer training time.

### 144. What do you understand by TF-IDF?
**Answer:** **TF-IDF (Term Frequency-Inverse Document Frequency)** is a numerical statistic that reflects the importance of a word in a document relative to a corpus, used in text mining.

### 145. How will you be able to create a Word2Vec of your own?
**Answer:** Train a **Word2Vec** model using a large corpus of text, capturing semantic relationships between words through vector representations.

### 146. List down at least 5 vectorization techniques.
**Answer:** **TF-IDF, Word2Vec, GloVe, FastText, Count Vectorization**.

### 147. What is the difference between RNN and Encoder-Decoder?
**Answer:** **RNN** processes sequential data, while **Encoder-Decoder** architecture is used in sequence-to-sequence tasks, encoding the input sequence into a fixed representation and decoding it into an output sequence.

### 148. What do you understand by the attention mechanism and what is the use of it?
**Answer:** **Attention mechanism** allows models to focus on relevant parts of the input sequence, improving performance in tasks like translation and summarization.

### 149. Have you read the research paper "Attention is All You Need"? If not, then why are you claiming you know NLP?
**Answer:** Yes, it is essential to understand the **Transformer model** introduced in the paper, which revolutionized NLP by using attention mechanisms.

### 150. What do you understand by multi-headed attention? Explain.
**Answer:** **Multi-headed attention** in transformers allows the model to focus on different parts of the input sequence simultaneously, capturing various aspects of relationships between words.
 
