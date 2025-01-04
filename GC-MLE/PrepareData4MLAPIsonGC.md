# Prepare Data for ML APIs on Google Cloud
## Vertex  AI Qwik Start

- In this lab we use:
    - BigQuery: For data processing and EDA.
    - Vertex AI: For training and deploy a custom TF Regressor model to predict customer lifetime value.

- General Architecture of VertexAI
![alt text](image-3.png)

- Theoretical Concepts: 
    * Vertex AI Workbench
    * Vertex AI Instances
    * Big Query

## Dataprep: Qwik Start

In this lab, you learn how to use Dataprep to complete the following tasks:

- Import data
- Correct mismatched data
- Transform data
- Join data

Dataprep is used to add a dataset and create recipes to wrangle the data into meaningful results.

- Theoretical Concepts:
    * Dataprep
    * Data import from Cloud Storage into bigQuery
    * Recipes

## Dataflow: Qwik Start - Templates

In this lab, you learn how to create a streaming pipeline using one of Google's Dataflow templates. More specifically, you use the Pub/Sub to BigQuery template, which reads messages written in JSON from a Pub/Sub topic and pushes them to a BigQuery table.
- Create a BigQuery dataset and table
- Create a Cloud Storage bucket
- Create a streaming pipeline using the Pub/Sub to BigQuery Dataflow template

- Learning outcome: Created a streaming pipeline using the Pub/Sub to BigQuery Dataflow template, which reads messages written in JSON from a Pub/Sub topic and pushes them to a BigQuery table.

- Theoretical Concepts:
    * DataFlow
    * Using Templates.
    * BigQuery Dataset
    * Pub/Sub to BigQuery

## Dataproc: Qwik Start

Dataproc is a fast, easy-to-use, fully-managed cloud service for running Apache Spark and Apache Hadoop clusters in a simpler, more cost-efficient way. Operations that used to take hours or days take seconds or minutes instead. Create Dataproc clusters quickly and resize them at any time, so you don't have to worry about your data pipelines outgrowing your clusters.

In this lab, you learn how to:
- Create a Dataproc cluster using the command line
- Run a simple Apache Spark job
- Modify the number of workers in the cluster

## Cloud Natural Language API: Qwik Start

Natural language is the language that humans use to communicate with each other. Natural language processing (NLP) is a field of computer science that is concerned with the interaction between computers and human language. NLP research has the goal of enabling computers to understand and process human language in a way that is similar humans.

The Cloud Natural Language API is a cloud-based service that provides natural language processing capabilities. It can be used to analyze text, identify entities, extract information, and answer questions.

Cloud Natural Language API features:

- Entity Recognition: Identify entities in text, such as people, places, and things.

- Sentiment Analysis: Analyze the sentiment of text, such as whether it is positive, negative, or neutral.

- Information Extraction: Extract information from text, such as dates, times, and price.

- Question Answering: Answer questions about text.

- Integrated REST API: Access via REST API. Text can be uploaded in the request or integrated with Cloud Storage.

## Speech-to-text API: Qwik Start

The Speech-to-Text API enables easy integration of Google speech recognition technologies into developer applications. The Speech-to-Text API allows you to send audio and receive a text transcription from the service.

## Video Intelligence: Qwik Start

Google Cloud Video Intelligence makes videos searchable and discoverable by extracting metadata with an easy to use REST API. You can now search every moment of every video file in your catalog. It quickly annotates videos stored in Cloud Storage, and helps you identify key entities (nouns) within your video; and when they occur within the video. Separate signal from noise by retrieving relevant information within the entire video, shot-by-shot, -or per frame.

## Prepare Data for ML APIs on Google Cloud: Challenge Lab

Topics tested:

- Create a simple Dataproc job: 
    - Create a cluster with the required configuration.
    - Submit a Spark job to the cluster.
- Create a simple DataFlow job: In this task, you use the Dataflow batch template Text Files on Cloud Storage to BigQuery under "Process Data in Bulk (batch)" to transfer data from a Cloud Storage bucket (gs://cloud-training/gsp323/lab.csv).
- Perform two Google machine learning backed API tasks:
    - Use Google Cloud Speech-to-Text API to analyze the audio file gs://cloud-training/gsp323/task3.flac. Once you have analyzed the file, upload the resulting file to: Cloud Speech Location
    - Use the Cloud Natural Language API to analyze the sentence from text about Odin. The text you need to analyze is "Old Norse texts portray Odin as one-eyed and long-bearded, frequently wielding a spear named Gungnir and wearing a cloak and a broad hat." Once you have analyzed the text, upload the resulting file to: Cloud Natural Language Location


