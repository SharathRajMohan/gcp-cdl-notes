# 02 Introduction to AI and Machine Learning on Google Cloud
## Introduction
* Layers of AI and ML solutions organization on GC:
    1. AI foundation layer: where you learn about cloud essentials like compute, storage, and network, and data tools such as data pipelines and data analytics.
    2.  AI development layer: where you explore different options to build an ML project, including out-of-the-box solutions, low-code or no-code, and DIY (do-it-yourself).
    3. Generative AI: and you learn how generative AI empowers the AI development and AI solutions layers.

## AI Foundations

* Google's AI offering milestones :
![alt text](image.png)

* Why google leads in the field of AI & ML innovations?
    1. State of the art AI models: Build on excellence.
    2. End to end development and MLOps: More efficient.
    3. Unified Data to AI platform: Develop with ease.
    4. Efficient & Scalable: get more for less.

* Responsible AI refers to the development and use of artificial intelligence systems in a way that prioritizes ethical considerations, fairness, accountability, safety, and transparency.

* 7 principles of responsible AI:
    * AI should be socially beneficial.
    * AI should avoid creating or reinforcing unfair bias.
    * AI should be built and tested for safety.
    * AI should be accountable to people.
    * AI should incorporate privacy design principles.
    * AI should uphold high standards of scientific excellence.
    * AI should be made available for uses that accord with these principles

* Course toolbox for AIML on GC:
    1. AI foundation layer: where you learn about cloud essentials like compute, storage, and network, and data tools such as data pipelines and data analytics.
        * Cloud essentials like compute and storage.
    2.  AI development layer: where you explore different options to build an ML project, including out-of-the-box solutions, low-code or no-code, and DIY (do-it-yourself).
        * Options to develop an ML model from beginning to end on Google Cloud.
            * Pre-trained APIs
            * BigQuery ML
            * AutoML
            * Custom Training
    3. Generative AI: and you learn how generative AI empowers the AI development and AI solutions layers.
        * Vertex AI
        * Vertex AI pipelines SDK

## Google Cloud Infrastructure:

* You can think of the Google Cloud infrastructure in terms of three layers.
    1. Networking & Security Layer
    2. Compute & Storage
    3. Data and AI Products

* Compute Solutions on GC
    * Google compute engine (GCE)
    * Google Kuberneteres Engine (GKE)
    * App Engine
    * Cloud Functions
    * Cloud Run:  a fully managed compute platform that enables you to run requests or event-driven stateless workloads without having to worry about servers.

* TPUs are Google’s custom-developed application-specific integrated circuits (ASICs) used to accelerate machine learning workloads.

* Google Cloud provides managed storage and database services that are scalable, reliable, and easy to operate. These are:
* Worldwide object storage:
    1. Cloud Storage
* Relational Database Services 
    1. Cloud SQL
    2. Cloud Spanner
    3. Big Query
* Non-Relational Database Services
    1. Firestore
    2. Cloud bigtable

* Google cloud Data and AI products
    * Ingestion and Process data: 
        * Pub/Sub
        * DataFlow
        * Cloud data fusion
        * Dataproc
    * Storage:
        * Cloud storage.
        * Cloud SQL
        * Cloud Spanner
        * Big Query
        * Firestore
        * Cloud Bigtable
    * Analytics
        * BigQuery
        * Looker (BI tool)
    * AI and ML
        * AI solutions
            * Documents AI
            * Contact center AI
            * Vertex AI Search for retail
            * Healthcare Data Engine
        * AI development
            * Vertex AI
                * AutoML
                * Workbench
                * Colab Enterprise
                * Vertex AI studio
                * Model Garden
        * These products are either integrated with generative AI or embedded with generative AI capabilities.

## ML model categories

* Artificial intelligence, or AI, is an umbrella term that includes anything related to computers mimicking human intelligence.

* Machine learning is a subset of artificial intelligence (AI) that allows computers to learn without being explicitly programmed.
    * Supervised Learning: deals with labeled data, is task-driven and identifies a goal.
        * Classification: Which identifies if a data point belongs to a certain category.
        * Regression: Which predicts the target variable which is a numerical data type.
    * Unsupervised learning: deals with unlabeled data, is data driven and identifies a pattern.
        * Clustering: Grouping like data points together.
        * Association: Identifying underlying relationships between variables.
        * Dimensionality Reduction: Reduces the number of dimensions.

* Deep Learning: subset of machine learning that adds layers in between input data and output results to make a machine learn at much depth.

* Big Query ML:
    * Fully managed storage facility
    * SQL based analytics engine

## AI Development Options

* Google's AI development options:
    * out-of-the-box
    * low-code and no-code
    * do-it-yourself.
* Google Cloud offers four options for building machine learning models:
    * Pre-Trained APIs: lets you use pre-trained machine learning models, so you don’t need to build your own if you don’t have training data or machine learning expertise in-house.
    * BigQueryML: uses SQL queries to create and execute machine learning models in BigQuery.
    * AutoML: a no-code solution that helps you build your own machine learning models on Vertex AI through a point-and-click interface.
    * Custom Training: through which you can code your very own machine learning environment, training, and deployment.
* BigQuery ML only supports tabular data, whereas the other three support tabular, image, text, and video.

![alt text](image-1.png)

## Google Pre-Trained APIs
* API stands for application programming interface, and they define how software components communicate with each other.
* Pre-Trained APIs: Are offered as services and can be plugged into the applications as a ready made AIML solution. 
* Natural Language API derives insights from text using pre-trained large language models.
    * Detects entities, sentiments, syntax & categories from text.

## Vertex AI

* Vertex AI: is the unified platform that supports various technologies and tools on Google Cloud to help you build an ML project from end to end.
* Google’s solution to many of the production and ease-of-use challenges is Vertex AI, a unified platform that brings all the components of the machine learning ecosystem and workflow together.
* Vertex AI verticals:
    * Vertex AI provides an end-to-end ML pipeline to prepare data, and create, deploy, and manage models over time, and at scale.
    * Vertex AI is a unified platform that encompasses both predictive AI and generative AI.

* Vertex AI allows users to build ML models with either AutoML, a no-code solution, or custom training, a code-based solution.
* Benefits of Vertex AI:
    1. Seamless: Vertex AI provides a smooth user experience from uploading and preparing data all the way to model training and production.
    2. Scalable: The machine learning operations (MLOps) provided by Vertex AI help to monitor and manage the ML production and therefore scale the storage and computing power automatically.
    3. Sustainable: All of the artifacts and features created using Vertex AI can be reused and shared.
    4. Speedy: Vertex AI produces models that have 80% fewer lines of code than competitors.
* In addition to AutoML and custom training, Vertex AI also provides tools for generative AI.

## AutoML
* AutoML, which stands for automated machine learning, aims to automate the process to develop and deploy an ML model.
* Four phases of AutoML:
    1. Data processing: 
        * After you upload a dataset, AutoML provides functions to automate part of the data preparation process.
    2. Searching the best model and fine-tuning parameters (AutoSearch): 
        * Two critical processes supports AutoSearch which are:
            * Neural architect Search:  helps search the best models and tune the parameters automatically.
                * The goal of neural architecture search is to find optimal models among many options. Tries different architectures and models, and compares against the performance between models to find the best ones.
            * Transfer Learning:  helps speed the searching by using the pre-trained models.
                * Transfer learning is a powerful technique that lets people with smaller datasets or less computational power achieve great results by using pre-trained models trained on similar, larger datasets.
    3. Packaging best model and preparing data for predictions
    4. Prediction

* By applying these advanced ML technologies, AutoML automates the pipeline from feature engineering, to architecture search, to hyperparameter tuning, and to model ensemble.

## Custom Training
* Custom Training: Allows you to create your own machine learning environment to experiment with and build your own ML pipelines.
* There are two options:
    1. Pre-built container:
    2. Custom container:
* Tools to code your ML model
    1. Vertex AI workbench:  a Jupyter notebook deployed in a single development environment that supports the entire data science workflow, from exploring to training and then deploying a machine learning model.
        * You can also use Colab Enterprise, which was integrated into Vertex AI Platform in 2023 so data scientists can code in a familiar environment.

*  TensorFlow: an end-to-end open platform for machine learning supported by Google. Tensorflow contains multiple abstraction layers. The TensorFlow APIs are arranged hierarchically, with the high-level APIs built on the low-level APIs.
    * The lowest layer is hardware: TensorFlow can run on different hardware platforms including CPU, GPU, and TPU.
    * The next layer is the low-level TensorFlow APIs, where you can write your own operations in C++ and call the core, basic, and numeric processing functions written in Python.
    * The third layer is the TensorFlow model libraries, which provide the building blocks such as neural network layers and evaluation metrics to create a custom ML model.
    * The high-level TensorFlow APIs like Keras sit on top of this hierarchy.

* Process behind building models with tensorflow (tf.keras)
    1. Create the model: where you piece together the layers of a neural network and configure the layers.
    2. Compile the model: specify hyperparameters such as performance evaluation and model optimization.
    3. Train the model: 

* Note that Vertex AI fully hosts TensorFlow from low-level to high-level APIs

* JAX: high-performance numerical computation library that is highly flexible and easy to use.