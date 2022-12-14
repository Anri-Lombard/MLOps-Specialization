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

In machine learning, the ultimate goal is to develop a model that not only performs well on the test set, but also solves the business problem at hand. This is because the purpose of machine learning is to develop models that can be used to make predictions or take actions in real-world situations. If a model only performs well on the test set but does not solve the business problem, it is not useful in practice and will not provide any value to the business. Therefore, it is important to choose a model that not only performs well on the test set, but also has the ability to solve the business problem and provide value in real-world situations.



__IN PROGRESS__