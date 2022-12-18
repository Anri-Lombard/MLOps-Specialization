# Machine Learning Engineering for Production (MLOps) Specialization

## Description
I am fascinated by what goes into building and deploying models such as ChatGPT and Github Copilot, which are not only successful but have improved my life immeasurably. This repository follows along with the MLOps Specialization in Coursera to learn from the world's best how to build and deploy models in production.

## Course structure
1. Introduction to Machine Learning in Production
2. Machine Learning Data Lifecycle in Production
3. Machine Learning Modeling Pipeline in Production
4. Deploying Machine Learning Models in Production

## Course 1: Introduction to Machine Learning in Production
### Week 1: Overview of the ML Lifecycle and Deployment

__MACHINE LEARNING PROJECT LIFECYCLE__

![Lifecycle of Machine Learning Model](Images/lifecycle.webp)

While developing machine learning models can be a complex and challenging task, it is only a small part of the work involved in bringing a machine learning project to production. In fact, the model itself typically accounts for only 5-10% of the entire project. The rest of the work involves tasks such as preparing and cleaning the data, setting up and maintaining the infrastructure to train and deploy the model, and building the software that integrates the model into a larger system. This often requires significant software development work, such as designing APIs, implementing security and privacy controls, and handling various edge cases. As a result, building and deploying machine-learning systems in production requires a combination of machine-learning expertise and software engineering skills. Many people in the machine learning community may not be aware of this, leading to a misconception that the field is primarily about building models and not about the broader engineering work involved in bringing those models to production.

![ML Code](Images/mlcode.png)

The lifecycle of a machine learning model is a process that involves the following steps:
1. Scoping: The first step in any machine learning project is to scope out the problem and determine whether a machine learning approach is appropriate for solving it. This might involve conducting research, gathering data, and identifying potential challenges or limitations.
2. Data: The next step is to explore and preprocess the data to make it suitable for use in your machine-learning model. This might involve cleaning, transforming, and normalizing the data, and performing any necessary feature engineering.
3. Modeling: Once the data is ready, you will need to select and evaluate the appropriate machine-learning model for your problem. This might involve experimenting with different algorithms, hyperparameters, and techniques to find the best model for your data.
4. Deployment: Finally, you will need to deploy the model in a production environment, where it can be used to make predictions and solve problems. This will also involve ongoing maintenance and monitoring of the model, to ensure that it continues to perform well and to identify any potential issues or improvements.

__DEPLOYMENT__

One of the key challenges in deploying machine learning models is concept drift, which occurs when the underlying distribution of the data changes over time. This can lead to a decline in the performance of the model, as it is no longer able to make accurate predictions on the new data. Another related challenge is data drift, which occurs when there are changes in the quality or format of the data being used to train the model. This can also impact the performance of the model and make it less effective at making predictions. Both concept drift and data drift can be difficult to detect and manage, and they require ongoing monitoring and retraining of the model to ensure it remains accurate and effective.

There are a number of key challenges to consider when deploying machine learning models, including:
1. Realtime or batch: One challenge is deciding whether to use a real-time or batch-based approach for deploying the model. Realtime deployment allows the model to make predictions as new data becomes available, while batch deployment involves processing data in larger chunks at regular intervals.
2. Cloud vs Edge/Browser: Another challenge is deciding whether to deploy the model in the cloud or on edge devices or in a web browser. Deploying in the cloud allows for easier scalability and access to more compute resources, but it can also result in higher latency and increased data transfer costs. Deploying on-edge devices or in a web browser can reduce latency and improve privacy, but it may also require more complex infrastructure and management.
3. Compute Resources (CPU/GPU/Memory): Choosing the right to compute resources for deploying the model is also critical. The type and amount of CPU, GPU, and memory available will impact the performance of the model and its ability to make predictions in real-time or batch mode.
2. Latency/Throughput: The latency and throughput of the model are also important considerations. Latency refers to the time it takes for the model to make a prediction, while throughput refers to the number of predictions the model can make per second. Both latency and throughput can impact the user experience and the overall performance of the model.
3. Logging: Logging is an essential part of deploying machine learning models, as it allows for tracking and monitoring of the model's performance and accuracy over time. Proper logging can also help with troubleshooting and debugging any issues that may arise.
4. Security and Privacy: Ensuring the security and privacy of the data being used to train and evaluate the model is also a key challenge. This can involve implementing encryption and other security measures to protect the data, as well as complying with relevant laws and regulations.

![Deployment](Images/deployment.jpeg)

Shadow mode, canary, and blue-green are deployment strategies that are often used in the context of machine learning.

Shadow mode involves deploying a machine learning model alongside the existing model in production, but not using it to serve predictions to users. Instead, the shadow model generates predictions in the background, and these predictions are compared with the predictions of the existing model. This allows the performance of the new model to be evaluated without disrupting the user experience.

Canary deployment involves deploying a new machine-learning model to a subset of users, while the existing model continues to serve the majority of users. This allows the performance of the new model to be evaluated on a smaller scale before it is rolled out to all users.

Blue-green deployment involves deploying a new machine-learning model alongside the existing model but routing a portion of the incoming requests to the new model. This allows for a seamless transition from the old model to the new one, as the two models can be compared and the performance of the new model can be evaluated before it is used to serve all predictions.

__Automation__

The degree of automation refers to the extent to which the deployment process is automated. Different levels of automation can be applied to the deployment process, ranging from fully manual to fully automatic.

At the lowest level of automation, the deployment process is completely manual and requires manual intervention at every step. This can be time-consuming and error-prone, and it is not suitable for deploying machine learning models in production environments.

At the next level of automation, some steps in the deployment process are automated, while others are performed manually. This can include automated testing and deployment of the machine learning model, but manual steps may be required to configure the model and set up the deployment environment.

At the highest level of automation, the entire deployment process is fully automated, from model training and testing to deployment and monitoring. This allows for rapid and efficient deployment of machine learning models, with minimal manual intervention. This is often the preferred approach for deploying machine learning models in production environments.

__Monitoring and Maintenance__

![Iteration](Images/iteration.png)

Model maintenance is the process of keeping a machine learning model up to date and performing well over time. This can involve several different activities, including monitoring the model's performance, retraining the model on new data, and adjusting the model's hyperparameters to improve its performance. It is important to perform regular model maintenance to ensure that the model continues to make accurate predictions and adapt to changes in the data over time. Additionally, model maintenance can help to prevent overfitting, where the model performs well on the training data but poorly on new data.

### Week 2: Select and Train a Model

__Modeling__

In machine learning, the ultimate goal is to develop a model that not only performs well on the test set but also solves the business problem at hand. This is because the purpose of machine learning is to develop models that can be used to make predictions or take actions in real-world situations. If a model only performs well on the test set but does not solve the business problem, it is not useful in practice and will not provide any value to the business. Therefore, it is important to choose a model that not only performs well on the test set but also can solve a business problem and provide value in real-world situations.

__Baseline__

Measuring human-level performance is crucial when comparing a machine-learning model to a human benchmark because it enables us to see how well the model is performing compared to a known standard. This helps us identify areas where the model is outperforming or underperforming compared to human performance and guides us in determining where to focus our attention and efforts. By comparing the model to human-level performance, we can gain a better understanding of the model's capabilities and limitations, and determine whether it is achieving the desired level of accuracy and performance.

__Precision, Recall, and F1__

In a machine-learning context, precision and recall are two metrics that are used to evaluate the performance of a classifier. Precision is a measure of the fraction of correct positive predictions, and is calculated using the formula:

    Precision = True Positives / (True Positives + False Positives)

Recall, on the other hand, is a measure of the fraction of positive cases that are correctly predicted, and is calculated using the formula:

    Recall = True Positives / (True Positives + False Negatives)

In other words, precision is a measure of the accuracy of the classifier's positive predictions, while recall is a measure of its ability to identify all of the positive cases in the data. Together, these two metrics can provide a more complete picture of a classifier's performance than either one alone

The F1 score is a measure that combines precision and recall into a single metric. It is calculated using the harmonic mean of precision and recall, which is defined as:
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

The F1 score is a useful metric because it takes into account both the classifier's precision and recall, and provides a balance between the two. It is particularly useful in cases where you want to avoid either under-prediction or over-prediction of the positive class. For example, in a medical diagnostic setting, a high F1 score would indicate that the classifier has both high precision (i.e. it makes very few false positive predictions) and high recall (i.e. it correctly identifies most of the positive cases).

__Data-centric vs. Model-centric__

The terms data-centric and model-centric refer to two different approaches to AI development. Data-centric AI development focuses on the data that is used to train and evaluate the performance of machine learning models. This approach emphasizes the importance of using large, diverse, and high-quality datasets to train and evaluate machine learning models.

On the other hand, model-centric AI development focuses on machine-learning algorithms and models themselves. This approach emphasizes the importance of developing novel, state-of-the-art machine learning algorithms and models and using them to make predictions or decisions based on data.

One advantage of data-centric AI development is that it allows for the use of large, diverse, and high-quality datasets to train and evaluate machine learning models. This can help to improve the accuracy and reliability of the predictions or decisions made by the models, as well as to reduce the risk of bias or overfitting.

Another advantage of data-centric AI development is that it allows for the use of transfer learning, which is a technique that enables machine learning models to be trained on a large dataset and then fine-tuned on a smaller, specialized dataset. This can save time and resources, and allow for the development of more effective and efficient machine-learning models.

Overall, data-centric AI development can be more advantageous than model-centric AI development because it allows for the use of large, diverse, and high-quality datasets to train and evaluate machine learning models, which can improve the accuracy and reliability of the predictions or decisions made by the models.

__Data Augmentation__

Data augmentation is a technique used to artificially increase the size of a dataset by generating additional data points based on the existing data. This is typically done by applying various transformations to the existing data, such as rotating, scaling, or cropping images, or by adding random noise or perturbations to the data.

The goal of data augmentation is to improve the performance of machine learning models by providing them with more diverse and robust training data. This can help to reduce overfitting, improve generalization, and make the models more robust to changes in the data distribution.

Data augmentation is often used in the context of image classification and object detection, where it can be used to generate additional training data by applying various transformations to the existing images. This can help to improve the performance of the models by making them more robust to changes in the orientation, scale, or lighting of the images.

Overall, data augmentation is a useful technique for increasing the size and diversity of a dataset, which can help to improve the performance of machine learning models.

__Experiment Tracking__

Experiment tracking is the process of keeping track of the different experiments that you run in your machine learning development and the results that they produce. This can be useful for several reasons, including:

- Comparing the results of different experiments to see which approach worked best
- Identifying trends and patterns in your results, which can help you improve your models over time
- Reproducing your experiments in the future, for example, to confirm your results or to share your work with others


To track your experiments effectively, it's important to track the following information:

- The details of your experiments, including the code that you used, the hyperparameters that you chose, and the results that you obtained
- The specific changes that you made to your code and your model so that you can easily reproduce your experiments in the future
- The inputs and outputs of your experiments, including the datasets that you used, the models that you trained, and the predictions that they made
- Any additional information that might be relevant to your experiments, such as the hardware that you used, the runtime that your experiments took, and any notes or observations that you made


There are a few different tools that you can use to track your experiments, including:

- Machine learning frameworks or platforms that provide built-in support for experiment tracking, such as TensorFlow or PyTorch
- Dedicated experiment tracking tools, such as Weights & Biases or Comet.ml, can help you organize and analyze your experiments
- Version control systems, such as Git, can help you track the code and other artifacts associated with your experiments


When choosing a tool to track your experiments, you should look for the following features:

- Automated logging of experiment details, including code, hyperparameters, and results
Visualization and analysis capabilities, such as graphs, charts, and tables, to help you understand your results
- Integration with other tools, such as machine learning frameworks and version control systems, to make it easy to use
- Collaboration and sharing features, such as the ability to share your experiments with others, or to work on them together
- Scalability and flexibility, support a large number of experiments and accommodate different types of data and models.

### Week 3: Data Definition and Baseline

__Unstructured and Structured Data__
Unstructured data is data that does not have a pre-defined structure or format, making it difficult to process and analyze using traditional data management tools. Examples of unstructured data include text documents, audio and video files, and social media posts. In contrast, structured data is data that is organized into a pre-defined format or schema, making it easier to process and analyze. Examples of structured data include database records, spreadsheets, and CSV files.

In general, humans are better at labeling unstructured data than structured data, as unstructured data often contains more complex and nuanced information that is difficult for machines to understand. For example, a human might be able to accurately label the sentiment of a written review, but a machine might struggle to do so. On the other hand, structured data is often easier for machines to process, as it is organized in a way that is more predictable and consistent. This can make it easier for machines to automatically label and analyze structured data.

__Small Data and Big Data__

Small data and big data are terms used to describe the size and complexity of data sets. Small data sets are relatively small and can be easily managed and analyzed using standard data management and analysis tools, while big data sets are so large and complex that they require specialized tools and techniques in order to be processed and analyzed effectively. The main difference between small data and big data is the size and complexity of the data sets involved, which can affect how the data is managed and analyzed. Small data sets are more manageable and easier to analyze, while big data sets require specialized tools and techniques to process and analyze effectively.

__Label Consistency__

Label consistency in machine learning refers to the idea that the labels assigned to data points should be consistent and correct in order for the machine learning model to be effective. This means that the labels should be accurately applied to the data, and should not be conflicting or inconsistent in any way. Ensuring label consistency is important because if the labels are incorrect or inconsistent, the machine learning model will not be able to learn effectively and will not be able to make accurate predictions. This can lead to poor performance and potentially even incorrect or harmful decisions made by the model.

__Human Level Performance__

![Human Level Performance](./Images/hlp.jpeg)

HLP, or human-level performance, refers to the ability of a machine-learning model to perform at the same level as a human. This is often considered the ultimate goal of many machine learning projects, as achieving human-level performance would mean that the model can perform tasks that require human-like intelligence and reasoning. To achieve HLP, machine learning models typically require a large amount of high-quality training data and powerful computational resources. Additionally, the development process often involves iterative testing and refining of the model to improve its performance.

Although, as Andrew mentioned, it is not necessary to achieve HLP to build a useful machine-learning model.

__Meta-data, Data Provenance, and Data Lineage__

Meta-data is information that describes other data. It can include things like the date a file was created, the author of the file, and keywords that describe the contents of the file.

Data provenance is the record of where data comes from. It can include things like the source of the data, the process by which it was collected, and any transformations or modifications that have been made to the data.

Data lineage is the history of data, including its origins, where it has been stored and processed, and how it has been transformed over time. Data lineage is often used to trace the origins of data and to ensure its integrity and accuracy.

Together, these concepts are important for understanding the origins and quality of data, which is essential for making decisions and taking actions based on that data.

__Scoping__

Scoping in machine learning development is the process of defining the goals, boundaries, and constraints of a machine learning project to ensure that the model being developed is well-suited to solving the intended problem.

Process:
1. Ask about business problems, not AI problems.
2. Brainstorm AI solutions.
3. Assess the feasibility and value of the potential solutions.
4. Determine milestones.
5. Budget for resources.

## Course 2: Machine Learning Data Lifecycle in Production
### Week 1: Collecting, Labeling, and Validating Data

__Collecting Data__

There are several advantages to collecting data in machine learning production:
1. Increased accuracy: One of the main advantages of collecting data in machine learning production is that it allows you to improve the accuracy of your models. As you collect more data, you can train your models on a larger and more diverse set of examples, which can help them generalize better to new cases.
2. Improved performance: In addition to improving accuracy, collecting data in machine learning production can also help improve the performance of your models. For example, if you are working on a recommendation system, collecting data on user interactions and preferences can help you improve the relevance and quality of your recommendations.
3. Better decision-making: By collecting data in machine learning production, you can gain insights into how your models are performing in the real world and use this information to make informed decisions about how to improve your system.

However, there are also some disadvantages to collecting data in machine learning production:
1. Data privacy concerns: One of the main concerns with collecting data in machine learning production is the potential for data privacy violations. It is important to have robust policies in place to protect the privacy of the individuals whose data you are collecting and to be transparent about how you are using the data.
2. Cost: Collecting data can be expensive, particularly if you need to purchase data from third-party sources. This can be a significant disadvantage for companies with limited budgets.
3. Bias: Another potential issue with collecting data in machine learning production is the potential for bias. If the data you are collecting is not representative of the population you are trying to model, your models may be biased and produce inaccurate results. It is important to be aware of this and take steps to ensure that your data is representative and unbiased.

__Labeling Data__

There are several advantages to labeling data in machine learning production:
1. Improved model accuracy: One of the main advantages of labeling data in machine learning production is that it allows you to improve the accuracy of your models. By providing explicit labels for the data, you can train your models to accurately classify or predict outcomes for new cases.
2. Increased understanding: Labelling data can also help you gain a better understanding of the data you are working with. By explicitly assigning labels to data points, you can identify patterns and trends that may not be immediately apparent otherwise.
3. Enhanced decision-making: By labeling data, you can also improve your ability to make informed decisions about how to use your data. For example, if you are working on a natural language processing task, labeling data can help you identify specific words or phrases that are important for your task.

However, there are also some disadvantages to labeling data in machine learning production:
1. Time and cost: Labelling data can be a time-consuming and costly process, particularly if you have a large dataset. This can be a significant disadvantage for companies with limited budgets or resources.
2. Human error: Another potential issue with labeling data is the potential for human error. If the data is not labeled accurately, it can negatively impact the performance of your models. It is important to have processes in place to ensure the accuracy of your labels.
3. Limited generalizability: Finally, it is important to keep in mind that labeled data is only representative of the specific cases it includes. Your models may not generalize well to new cases that are not included in the labeled data. This can be a significant disadvantage if you are working on tasks that require your models to generalize to a wide range of cases.

__Validating data__

There are several advantages to validating data in machine learning production:
1. Improved model accuracy: One of the main advantages of validating data in machine learning production is that it allows you to improve the accuracy of your models. By testing your models on unseen data, you can ensure that they are not overfitting to the training data and can generalize well to new cases.
2. Enhanced decision-making: Validating data can also help you make informed decisions about how to use your data. For example, if you are working on a classification task, you can use validation data to determine the optimal threshold for classifying cases as positive or negative.
3. Increased confidence in results: By validating your models on unseen data, you can increase your confidence in the results they produce. This can be particularly important when you are using your models to make decisions with real-world consequences, such as in medical or financial applications.

However, there are also some disadvantages to validating data in machine learning production:
1. Time and cost: Validating data can be a time-consuming and costly process, particularly if you have a large dataset. This can be a significant disadvantage for companies with limited budgets or resources.
2. Limited generalizability: It is also important to keep in mind that validation data is only representative of a specific subset of your overall dataset. Your models may not generalize well to new cases that are not included in the validation data.
3. Human error: Finally, there is a potential for human error when selecting and preparing validation data. If the data is not representative of the population you are trying to model, or if there are errors in the labels or features, it can negatively impact the performance of your models. It is important to have processes in place to ensure the accuracy and representativeness of your validation data.

#### Week 2: Feature Engineering, Transformation, and Selection

__Feature Engineering__

Feature engineering is the process of creating and selecting features to use in a machine-learning model to improve model performance.

__Preprocessing__

Preprocessing in machine learning refers to the steps taken to prepare data for use in a machine learning model. Preprocessing usually involves cleaning and formatting the data in a way that is suitable for training the model, as well as selecting and extracting features from the data that are most relevant to the task at hand.

Many different preprocessing techniques can be used, depending on the nature of the data and the specific requirements of the machine learning task. Some common preprocessing steps include:

1. Handling missing or incomplete data: This may involve imputing missing values, or dropping rows or columns with too many missing values.
2. Handling outliers: Outliers can hurt the performance of some machine learning algorithms, so it may be necessary to identify and either remove or transform these data points.
3. Feature scaling: Some machine learning algorithms are sensitive to the scale of the input features, so it may be necessary to scale the features to a common range (e.g., 0-1).
4. Feature selection: In many cases, the raw data will contain many features that are not relevant or useful for the task at hand. Feature selection involves identifying and selecting the most relevant features for training the model.
5. Data transformation: This may involve applying transformations such as log transformations or normalization to the data to make it more suitable for use with certain machine learning algorithms.

Preprocessing is an important step in the machine learning process, as it can significantly impact the performance and accuracy of the trained model. It is important to carefully consider which preprocessing steps are appropriate for the specific data and task at hand.


__IN PROGRESS__