---
title: Example Reference
description: A reference page in my new Starlight docs site.
---


# Interview Questions Set 3

## ML Questions Part 2

### 1. Tell me something about your project you have done in the past?
**Answer:** **In my past project**, I developed a **predictive maintenance system** for industrial equipment using machine learning algorithms to predict failures and schedule maintenance proactively.

### 2. What was your dataset size for the ML project?
**Answer:** The **dataset size** was approximately **100,000 records** with around **50 features**.

### 3. What is the type of your dataset?
**Answer:** The dataset was **structured** with **numerical and categorical** features.

### 4. What was the frequency of your dataset? (E.g., batch, streaming, etc.)
**Answer:** The dataset was processed in **batch mode** with daily updates.

### 5. What was the source system for your dataset? (E.g., sensor, satellite, Kafka, cloud, etc.)
**Answer:** The data was sourced from **sensors** installed on the industrial equipment, and the data was streamed through **Kafka** into a **cloud database**.

### 6. What kind of derived dataset have you mentioned in the project?
**Answer:** The derived dataset included **aggregated statistics** like mean, standard deviation, and rolling averages of sensor readings.

### 7. How have you done the validation of the dataset?
**Answer:** **Validation** was done by **cross-checking with historical maintenance records** and using **data quality checks** for missing values and outliers.

### 8. Have you created any pipeline to validate this dataset or were you using any tool?
**Answer:** Yes, I created a **data validation pipeline** using **Apache Airflow** to automate the validation process.

### 9. What do you understand by a data lake?
**Answer:** A **data lake** is a centralized repository that allows you to store all your structured and unstructured data at any scale.

### 10. What do you understand by data warehousing?
**Answer:** **Data warehousing** involves aggregating and storing large volumes of data from different sources in a central repository for analysis and reporting.

### 11. Can you name some validations that you have done on top of your data?
**Answer:** 
1. **Checking for missing values**
2. **Detecting outliers**
3. **Ensuring data type consistency**
4. **Validating data ranges**
5. **Duplicate record detection**

### 12. How have you handled a streaming dataset?
**Answer:** Streaming data was handled using **Apache Kafka** for ingestion and **Apache Spark Streaming** for real-time processing.

### 13. How many different types of environments were available in your project?
**Answer:** The project had **three environments**: **development, staging,** and **production**.

### 14. What was your delivery mechanism for the particular project?
**Answer:** Delivery was managed through a **CI/CD pipeline** using **Jenkins** for automated builds, tests, and deployments.

### 15. Have you used any OPS pipeline for this current project?
**Answer:** Yes, an **AI-Ops pipeline** was used for continuous monitoring and automation of IT operations.

### 16. How were you doing model retraining?
**Answer:** Model retraining was done using a **scheduled job** that triggered retraining processes on new data batches.

### 17. How have you implemented model retraining in your project?
**Answer:** Model retraining was implemented using **Apache Airflow** to schedule and manage the workflow, and **Docker** to ensure consistent environments.

### 18. How frequently have you been doing model retraining and what was the strategy for model retraining?
**Answer:** Model retraining was done **monthly**, and the strategy involved using the latest data to update the model and validate its performance.

### 19. What kind of evaluation were you doing in the production environment?
**Answer:** Evaluation included **monitoring model performance metrics** such as accuracy, precision, recall, and **anomaly detection** for significant performance drops.

### 20. What was the number of requests (hits) your model was receiving on a daily basis?
**Answer:** The model was receiving approximately **10,000 requests per day**.

### 21. How have you implemented logging in the project for any failure cases?
**Answer:** Logging was implemented using **ELK stack** (Elasticsearch, Logstash, Kibana) to capture and visualize logs and errors.

### 22. How have you integrated the notification (or Alarm) system for your project?
**Answer:** **Notifications** and **alerts** were integrated using **Slack API** and **PagerDuty** for real-time incident management.

### 23. How have you implemented model monitoring?
**Answer:** Model monitoring was implemented using **Prometheus** for metrics collection and **Grafana** for visualization.

### 24. How have you derived the final KPI (Key Performance Indicator) for your client?
**Answer:** Final KPIs were derived by aligning model outputs with client business goals and using **dashboard metrics** to track performance.

### 25. How many dashboards were there in your project?
**Answer:** There were **three dashboards**: **operational metrics, model performance,** and **business impact**.

### 26. On which platform have you productionized your model?
**Answer:** The model was productionized on **AWS SageMaker**.

### 27. What kind of API have you exposed to receive data for the model?
**Answer:** A **RESTful API** was exposed using **Flask** to receive data for the model.

### 28. What was the size of your final production environment (system configuration)?
**Answer:** The production environment included **4 vCPUs, 16 GB RAM,** and **100 GB SSD storage**.

### 29. What databases have you used in the project?
**Answer:** Databases used include **PostgreSQL** for structured data and **Amazon S3** for unstructured data storage.

### 30. What kind of optimization have you done in your project, till what depth & explain the example?
**Answer:** Optimization included **query optimization**, **model hyperparameter tuning**, and **resource allocation**. For example, **indexing** was used in PostgreSQL to speed up data retrieval.

### 31. Can you please talk about the complete team structure and team size?
**Answer:** The team consisted of **5 members**: **1 project manager, 2 data scientists, 1 data engineer,** and **1 DevOps engineer**.

### 32. What was the duration of your complete project?
**Answer:** The project duration was **6 months**.

### 33. What was your day-to-day responsibility in the last 2 months?
**Answer:** My responsibilities included **model development**, **data preprocessing**, **feature engineering**, and **collaborating with the team** on integration and deployment.

### 34. What kind of change requests have you been receiving after you productionized the project?
**Answer:** Change requests included **feature updates**, **performance improvements**, and **new data source integration**.

### 35. What kind of testing have you done in development, UAT, pre-prod, and prod?
**Answer:** 
- **Development:** Unit testing, integration testing
- **UAT (User Acceptance Testing):** Functional testing, user feedback
- **Pre-prod:** Performance testing, load testing
- **Prod:** Monitoring, validation, and rollback testing

### 36. Have you used some of the predefined AI-OPS pipelines? If yes, explain.
**Answer:** Yes, used predefined pipelines like **Kubeflow** for managing machine learning workflows and automating model retraining.

### 37. Who has implemented AI-OPS in your project?
**Answer:** The **DevOps team** along with **data scientists** collaborated to implement AI-OPS.

### 38. What was the OPS stack you have been using?
**Answer:** The OPS stack included **Docker, Kubernetes, Jenkins, Prometheus,** and **Grafana**.

### 39. What do you understand by CI-CD & have you implemented those in your project? If yes, what was the tech stack you used for the CI-CD pipeline?
**Answer:** **CI-CD (Continuous Integration/Continuous Deployment)** is a practice of automating the integration and deployment of code changes. Implemented using **Jenkins, Git, Docker,** and **Kubernetes**.

### 40. What was the biggest challenge you faced in the project and how have you resolved it?
**Answer:** The biggest challenge was **data inconsistency** across sources. Resolved by implementing **data validation pipelines** and using **data reconciliation techniques**.

### 41. Give me one scenario where you worked as a team player?
**Answer:** Worked with the **data engineering team** to integrate new data sources, ensuring data quality and consistency for the ML models.

### 42. What was your overall learning from the current project?
**Answer:** Gained experience in **end-to-end ML pipeline development**, **model deployment**, and **real-time data processing**.

### 43. How do you keep yourself updated with new technology?
**Answer:** Regularly read **research papers, blogs**, attend **webinars**, and participate in **online courses**.

### 44. Have you designed an architecture for this project? If yes, define a strategy with respect to your current project.
**Answer:** Yes, the architecture included **data ingestion pipelines, a centralized data lake, feature engineering modules, model training workflows, and deployment automation**.

## Questions for People with 7+ Years of Experience

### 45. What kind of discussions generally happen with clients?
**Answer:** Discussions include **requirement gathering, project updates, timelines, deliverables, performance metrics,** and **feedback sessions**.

### 46. What was your contribution to team building?
**Answer:** Assisted in **hiring** and **training new team members**, fostering a **collaborative work environment**, and encouraging **knowledge sharing**.

### 47. How have you defined the complete tech stack for AI?
**Answer:** Defined the tech stack based on project requirements, including **data processing tools (Apache Spark), machine learning frameworks (TensorFlow, Scikit-learn), and deployment platforms (AWS, Docker, Kubernetes)**.

### 48. What kind of benefit have you given to your current company in terms of cost-cutting and revenue?
**Answer:** Implemented **automated workflows** and **optimized resource usage**, resulting in **cost savings** and **improved operational efficiency**.

### 49. What kind of new innovations have you introduced?
**Answer:** Introduced **automated machine learning (AutoML)** techniques and **real-time data processing pipelines**.

### 50. How do you push your team for research or new implementation?
**Answer:** Encourage **regular brainstorming sessions**, allocate **dedicated research time**, and provide **access to learning resources**.

### 51. How many projects are you handling?
**Answer:** Handling **multiple projects** simultaneously, typically **2-3 projects**.

### 52. How many clients have you acquired?
**Answer:** Acquired **several clients** through successful project deliveries and client referrals.

### 53. If a new demand comes from a client, how do you evaluate that requirement?
**Answer:** Evaluate the requirement based on **feasibility, resources, time**, and **alignment with business goals**.

### 54. How do you prepare costing for the project?
**Answer:** Prepare costing by estimating **resource needs, time, software/hardware costs**, and **risk factors**.

### 55. What kind of skillsets do you look for in a person to handle the delivery of an upcoming project?
**Answer:** Look for **technical expertise, problem-solving skills, teamwork**, and **effective communication**.

### 56. On average, how much time do you take for building a new team?
**Answer:** Typically, it takes **4-6 weeks** to build a new team, including **recruitment, onboarding,** and **initial training**.

### 57. What kind of stack did you involve in the initial project?
**Answer:** Involved a stack including **Python, TensorFlow, Apache Spark, Docker, Kubernetes**, and **AWS services**.

### 58. How do you decide the timeline for project delivery?
**Answer:** Decide the timeline based on **project scope, resource availability, complexity**, and **client requirements**.

### 59. How do you keep track of project progress?
**Answer:** Use **project management tools** like **JIRA, Trello**, and **regular team meetings** to track progress.

### 60. How do you handle dependencies between the team?
**Answer:** Manage dependencies through **clear communication**, **documented processes**, and **coordination meetings**.

### 61. How much profit have you given to your previous organization?
**Answer:** Delivered projects that resulted in **significant revenue generation** and **cost savings**, contributing to overall profitability.

## Deep Learning & Vision Questions

### 62. How many images have you taken to train your DL model?
**Answer:** Used a dataset of **50,000 images** for training the DL model.

### 63. What is the size of your model that you have productionized?
**Answer:** The model size was approximately **200 MB**.

### 64. Have you tried optimizing this Vision or DL model?
**Answer:** Yes, optimization was done using techniques like **model pruning** and **quantization**.

### 65. Where have you hosted your Computer Vision model?
**Answer:** Hosted on **AWS EC2** instances with GPU support.

### 66. What was your frame per second?
**Answer:** Achieved **30 frames per second (FPS)** for real-time video processing.

### 67. What are the data filtration strategies you have defined for the CV project in production?
**Answer:** **Data filtration** included **preprocessing steps** like **resizing, normalization**, and **augmentation** to improve model performance.

### 68. Have you used any edge device in this project? If yes, why?
**Answer:** Yes, used edge devices like **NVIDIA Jetson Nano** for **real-time inference** to reduce latency and dependency on cloud processing.

### 69. What was the name of the camera & camera quality?
**Answer:** Used **Logitech C920** cameras with **1080p resolution**.

### 70. What was the final outcome you were generating from these devices?
**Answer:** Generated **real-time object detection** and **tracking** outputs.

### 71. Have you processed the data in the local system or in the cloud? Give a reason.
**Answer:** Processed data in the **cloud** for scalability and **local system** for real-time requirements and latency reduction.

### 72. How many devices have you productionized (camera, edge devices, etc.)?
**Answer:** Productionized around **20 devices** including **cameras and edge devices**.

### 73. Let’s suppose I am trying to build a solution to count the number of vehicles or to detect their number plate or track their speed. Then what is the dependency of distance, position & angle of the camera on your final model? What will happen to your model if we change position angle?
**Answer:** The **distance, position, and angle** of the camera significantly impact the model's accuracy. Changing these parameters can **affect detection accuracy** and may require **recalibration** or **model retraining**.

### 74. What was your data collection strategy in the CV project? Have you received data from the client or have you created the data? And how have you implemented it?
**Answer:** Data was collected using **cameras installed at different locations** and **synthetic data generation** to augment the dataset. Implemented by **automating data capture and labeling**.

### 75. What was the data labeling tool that you have used for your project?
**Answer:** Used tools like **LabelImg** and **Amazon SageMaker Ground Truth** for data labeling.

### 76. If I have to do OCR then which API will you use or have you used in your previous project?
**Answer:** Used **Google Cloud Vision API** and **Tesseract OCR** for Optical Character Recognition (OCR).

### 77. Suppose if my image data is blur, what will be your strategy to enhance the image quality?
**Answer:** Strategies include using **image enhancement techniques** like **deblurring algorithms**, **super-resolution models**, and **filtering**.

### 78. Have you implemented object tracking in any of your projects? If yes, give me a scenario.
**Answer:** Yes, implemented object tracking in a **retail analytics project** to monitor customer movement and behavior in the store.

### 79. Suppose there are 2 mobile devices that are moving, so if these two device positions overlap with each other, then what will be your strategy to avoid any error while tracking those devices using a camera?
**Answer:** Use **multi-object tracking algorithms** like **Deep SORT** that can handle **occlusions** and **overlapping objects**.

### 80. Have you implemented multicamera tracking? Do you have any idea about it?
**Answer:** Yes, implemented multi-camera tracking to **provide a continuous tracking system** by integrating feeds from multiple cameras.

### 81. Explain some real-life use cases of segmentation.
**Answer:** 
- **Medical imaging:** Segmenting tumors in MRI scans.
- **Autonomous driving:** Segmenting road, vehicles, and pedestrians.
- **Agriculture:** Segmenting crops and weeds in field images.

### 82. What kind of AI application will you build for a retail seller to increase their sales?
**Answer:** Develop an AI application for **personalized recommendations**, **customer behavior analysis**, and **inventory optimization**.

### 83. Let’s suppose I am trying to build an AI solution to monitor the productivity of a kid. What kind of feature would you like to give in that product?
**Answer:** Features include **activity tracking, focus monitoring, screen time analysis**, and **engagement metrics**.

## NLP

### 84. Have you productionized a BERT model? If yes, can you talk about hurdles that you have faced?
**Answer:** Yes, faced hurdles like **high computational requirements, managing large model sizes**, and **fine-tuning for specific tasks**.

### 85. How have you optimized your BERT-based solution?
**Answer:** Optimized using **distillation**, **model quantization**, and **parameter tuning**.

### 86. What kind of NLP tasks were you doing with respect to BERT?
**Answer:** Tasks include **text classification, named entity recognition (NER)**, and **question-answering**.

### 87. Have you implemented BERT base & BERT large?
**Answer:** Yes, implemented both **BERT base** for faster inference and **BERT large** for improved accuracy.

### 88. What are the disadvantages of using BERT?
**Answer:** Disadvantages include **high resource consumption**, **long training times**, and **complexity in fine-tuning**.

### 89. Can you please name one of the lighter versions of a transformer-based model?
**Answer:** **DistilBERT** and **ALBERT** are lighter versions of transformer models.

### 90. What was the accuracy that you were receiving with respect to specific tasks?
**Answer:** Achieved **accuracy** ranging from **85% to 92%** depending on the task and dataset.

### 91. Let’s suppose I have to build a language-based model and there are online solution providers but they are costly, so what will be your strategy?
**Answer:** Strategy would include **using open-source models**, **fine-tuning pre-trained models**, and **deploying on cost-effective cloud services**.

### 92. Have you used Hugging Face APIs?
**Answer:** Yes, used **Hugging Face Transformers** library for model deployment and **fine-tuning**.

### 93. What is the difference between BERT and GPT-based models?
**Answer:** 
- **BERT**: Encoder-only model, good for tasks requiring contextual understanding.
- **GPT**: Decoder-only model, excels in generative tasks like text generation.

### 94. There is no Decoder model in BERT then how do we get output if it's just an encoder-level model?
**Answer:** **BERT** outputs contextual embeddings that can be fed into a **task-specific layer** (e.g., classification layer) to get the final output.

### 95. How is masking implemented in BERT-based models and what are its disadvantages?
**Answer:** **Masking** is implemented by randomly masking input tokens to predict them during training. Disadvantages include **inefficiency** and **training complexity**.

### 96. How is masking implemented in GPT-based models and what are its disadvantages?
**Answer:** In **GPT**, masking is done using a causal mask to prevent attention to future tokens. Disadvantages include **limitation to sequential tasks**.

### 97. How does backpropagation happen in the BERT model? Explain.
**Answer:** Backpropagation in BERT involves **computing gradients of the loss** with respect to model parameters and **updating them** using optimization algorithms.

### 98. Can you explain Query, Key & Value in any Transformer-based model?
**Answer:** 
- **Query**: The input representation to be matched.
- **Key**: The input representation to be compared against.
- **Value**: The information retrieved based on the matching.

### 99. What are the main reasons behind the success of transformer-based models?
**Answer:** Reasons include **parallel processing**, **self-attention mechanism**, and **scalability** to large datasets.

### 100. Can we use BERT-based models to generate embedding? If yes, how? If no, why?
**Answer:** Yes, BERT-based models can generate embeddings by extracting the output of the **encoder layers** for input tokens.

### 101. What do you think about OpenAI GPT-3?
**Answer:** **GPT-3** is a powerful language model with state-of-the-art performance in various NLP tasks but requires significant **computational resources** and **fine-tuning**.

### 102. What is your thought about convolution autoencoder?
**Answer:** **Convolutional autoencoders** are effective for **unsupervised feature learning** and **image reconstruction**, capturing spatial hierarchies in data.

### 103. List down text summarization techniques, and which latest model you will prefer for text summarization?
**Answer:** Techniques include **extractive summarization** and **abstractive summarization**. Prefer models like **BART** and **T5** for abstractive summarization.

### 104. What is the meaning of multi-headed attention?
**Answer:** **Multi-headed attention** allows the model to **attend to different parts of the input sequence** simultaneously, capturing diverse information.

### 105. What do you understand by BLEU Score?
**Answer:** The **BLEU Score** measures the quality of machine-generated text against reference translations, focusing on **precision**.

### 106. What is gradient clipping?
**Answer:** **Gradient clipping** is a technique to **prevent exploding gradients** by capping the gradient values during backpropagation.

### 107. Can you please list down ways by which I will be able to split training across multiple GPUs?
**Answer:** 
1. **Data parallelism**
2. **Model parallelism**
3. **Horovod**
4. **DistributedDataParallel** in PyTorch

### 108. Explain the difference between GRU and LSTM.
**Answer:** 
- **GRU (Gated Recurrent Unit)**: Simpler architecture with fewer parameters.
- **LSTM (Long Short-Term Memory)**: More complex, with additional gates to manage long-term dependencies.

### 109. If I have to implement an NLP pipeline, what will be your approach? (Hint: using NLTK, Spacy, or state-of-the-art model)
**Answer:** Use **state-of-the-art models** for key tasks, leveraging **Spacy** for preprocessing and **transformers** for core NLP tasks like **NER, text classification**, and **summarization**.

### 110. What do you understand by Uni-gram, bi-gram, and tri-gram (N-gram)?
**Answer:** **N-grams** are contiguous sequences of **N items** from a given text. **Uni-gram** (single word), **bi-gram** (two words), **tri-gram** (three words).

### 111. What do you understand by stemming and lemmatization?
**Answer:** 
- **Stemming**: Reduces words to their base form by removing suffixes.
- **Lemmatization**: Converts words to their base or root form using vocabulary and morphological analysis.

### 112. For a conversational AI solution, will you use a predefined framework or create your own? In both cases, what are the advantages and disadvantages?
**Answer:** 
- **Predefined framework**: Faster development, proven tools (e.g., Google Dialogflow), but less flexibility.
- **Custom solution**: More control and customization, but requires more development time and resources.

### 113. Have you worked on Google Dialogflow, Azure-LUIS, IBM-Watson, or RASA-NLU?
**Answer:** Yes, worked on **Google Dialogflow** and **RASA-NLU** for building conversational AI systems.

### 114. What are the limitations of these respective platforms mentioned above?
**Answer:**
- **Google Dialogflow**: Limited customization.
- **Azure-LUIS**: Requires deep integration with Azure services.
- **IBM-Watson**: Higher cost.
- **RASA-NLU**: Requires more manual setup and configuration.

### 115. If I have to build a ticket rerouting system for a banking client, how will you design this complete system?
**Answer:** Design a system using **NLP** for **ticket classification**, a **rule-based engine** for rerouting, and **integration with ticketing systems** for automation.

### 116. Can you please explain how you will be able to design an app like In-Shorts? (Hint: text summarization & etc.)
**Answer:** Use **abstractive summarization models** like **BART** for generating concise summaries, with **topic modeling** to categorize news articles.

### 117. If you have to build a solution that can generate a summary of the entire online class meeting, what will be your approach and what kind of hurdles you may face?
**Answer:** Approach includes using **ASR (Automatic Speech Recognition)** for transcription, followed by **text summarization** models. Hurdles include **noisy audio**, **multiple speakers**, and **context understanding**.

### 118. If I have to create a Gmail kind of text generation system, what will be your approach?
**Answer:** Use **transformer models** like **GPT-3** for text generation, with **fine-tuning** on email-specific datasets for contextual accuracy.

### 119. If I have to create a document parsing and validation system for legal, what will be your approach?
**Answer:** Use **NLP techniques** for parsing, with **named entity recognition** and **rule-based validation** for ensuring document compliance.

### 120. If you have to build a voice-based automation system, how will you design system architecture?
**Answer:** Design using **ASR** for voice input, **NLP** for intent recognition, and **text-to-speech (TTS)** for voice output, integrated with backend systems for automation.

## Time Series

### 121. List down time series algorithms that you know?
**Answer:** **ARIMA, SARIMA, Prophet, LSTM, GRU, VAR (Vector AutoRegressive)**.

### 122. How can we solve TS problems in deep learning?
**Answer:** Use **LSTM**, **GRU**, and **Temporal Convolutional Networks (TCN)** for time series forecasting.

### 123. Give applications of TS in weather, financial, healthcare & network analysis?
**Answer:** 
- **Weather:** Forecasting temperature and precipitation.
- **Financial:** Predicting stock prices and market trends.
- **Healthcare:** Monitoring patient vital signs and disease progression.
- **Network analysis:** Detecting network anomalies and traffic patterns.

### 124. What is the difference between uptrend and downtrend in TS?
**Answer:** 
- **Uptrend:** A pattern where the time series data shows a consistent rise over time.
- **Downtrend:** A pattern where the time series data shows a consistent decline over time.

### 125. What do you understand by seasonality in TS?
**Answer:** **Seasonality** refers to **regular, periodic fluctuations** in time series data, often related to calendar cycles (e.g., monthly, quarterly).

### 126. What do you understand by cyclic pattern in your TS data?
**Answer:** **Cyclic patterns** are **long-term fluctuations** in time series data that are not of fixed periods, influenced by economic or environmental factors.

### 127. How will you find a trend in TS Data?
**Answer:** Use techniques like **moving averages**, **exponential smoothing**, or **decomposition** to isolate and identify trends in time series data.

### 128. Have you implemented the ARCH model in TS? If yes, give a scenario.
**Answer:** Yes, implemented the **ARCH model** for **financial time series** to model and predict **volatility**.

### 129. What is the VAR (vector autoregressive) model?
**Answer:** **VAR (Vector AutoRegressive)** model is used for **multivariate time series analysis**, capturing linear interdependencies among multiple time series.

### 130. What do you understand by univariate and multivariate TS Analysis?
**Answer:** 
- **Univariate TS Analysis:** Analyzing a single time series.
- **Multivariate TS Analysis:** Analyzing multiple, related time series simultaneously.

### 131. Give an example where you have created a multivariate model?
**Answer:** Created a **multivariate model** to forecast **energy consumption** based on multiple features like **temperature, humidity,** and **historical usage**.

### 132. What do you understand by p, d, & q in the ARIMA model?
**Answer:** 
- **p:** Number of lag observations in the model (autoregressive part).
- **d:** Number of times the raw observations are differenced (integrated part).
- **q:** Size of the moving average window (moving average part).

### 133. Tell me the mechanism by which I can find p, d, q in the ARIMA model?
**Answer:** Use **ACF (AutoCorrelation Function)** and **PACF (Partial AutoCorrelation Function)** plots to identify **p** and **q**. Use **differencing** and **Augmented Dickey-Fuller test** to determine **d**.

### 134. What is SARIMA and how is it different from ARIMA?
**Answer:** **SARIMA (Seasonal ARIMA)** incorporates **seasonal components** into the ARIMA model, handling seasonal effects in time series data.

### 135. What is the meaning of AR, MA, and I in the ARIMA model?
**Answer:** 
- **AR (Autoregressive):** Uses dependencies between an observation and a number of lagged observations.
- **MA (Moving Average):** Uses dependencies between an observation and a residual error from a moving average model.
- **I (Integrated):** Differencing of raw observations to make the time series stationary.

### 136. Can we solve TS problems with transformers? What is your thought on that? Why do you think in that way?
**Answer:** Yes, **transformers** can be used for TS problems as they can handle **sequential data** and **long-range dependencies** effectively.

### 137. Have you ever productionized a TS-based model using LSTM? What are the advantages and disadvantages?
**Answer:** Yes, productionized a TS-based model using LSTM. 
- **Advantages:** Captures long-term dependencies.
- **Disadvantages:** Computationally expensive and requires large datasets.

### 138. Can we solve TS problems using a regressive algorithm? If yes, why? If no, give a reason.
**Answer:** Yes, regressive algorithms like **linear regression** can be used for TS problems if the relationship between variables is linear. For non-linear relationships, other algorithms may be more suitable.
