# 03 Launching into ML
## Introduction
* Topics in the module:
    * Data Quality
    * Exploratory Data Analysis
    * Modeling using AutoML on Vertex AI
    * Modeling using AutoML on BigQuery
    * Optimize and Evaluate models using performance metrics.
    * create repeatable and scalable training, evaluation, and test datasets.

## Get to Know Your Data: Improve Data through Exploratory Data Analysis
*  In this module we look at how to improve the quality of our data and how to explore our data by performing exploratory data analysis.
### Improve Data Quality
* The two phases in ML:
    1. Training Phase
    2. Inference Phase
* In any ML project, after you define the best use case and establish the success criteria, the process of delivering an ML model to production involves the following steps:
    * Data Steps:
        1. Data Extraction: Can be real-time or batch
        2. Data Analysis
        3. Data Preparation:  includes data transformation, which is the process of changing or converting the format, structure, or values of data you've extracted into another format or structure.
            * Data Cleansing: remove superfluous(More than required) and repeated records from log data.

        * As a first step toward determining their data quality levels, organizations typically perform data asset inventories in which the relative accuracy, uniqueness, and validity of data is measured. 
        * Attributes related to data quality:
            * Data Accuracy
            * Data Consistency
            * Timeliness of Data: can be measured as the time between when information is expected and when it is readily available for use.
            * Completeness of Data

### What is EDA?
* In statistics, exploratory data analysis or EDA is an approach to analyzing data sets to summarize their main characteristics, often with visual methods.
* Exploratory data analysis is an approach for data analysis that employs a variety of techniques, mostly graphical, to maximize insight into a data set, uncover underlying structure, extract important variables, detect outliers and anomalies, test underlying assumptions, develop parsimonious models and determine optimal factor settings.
* The purpose of an EDA is to find insights which will serve for data cleaning, preparation, or transformation, which will ultimately be used in a machine learning algorithm.
* The three popular data analysis approaches are:
    * Classical
    * Exploratory data analysis 
    * Bayesian: to determine posterior probabilities based on prior probabilities and new information.
        * Posterior probabilities is a the probability an event will happen after all evidence or background information has been taken into account.
        * Prior probability is the probability an event will happen before you've taken adding new evidence into account.

* For exploratory data analysis, the focus is on the data, its structure, outliers and models suggested by the data.
* Main methods of EDA
    * Univariate Analysis:  analysis of a single variable. It doesn't deal with causes or relationships. Two types of variables are:
        * Categorical : can be classified into groups.
        * Continuous:  can be measured on a continuous scale.

    * Bivariate Analysis:  analysis of two variables. It deals with the relationship between two variables.
        * Factor Plot (Seaborn): draw a categorical plot up to a facet grid.
        * Seaborn's jointplot function draws a plot of two variables with bivariate and univariate graphs.
        * Seaborn's factorplot map method can map a factorplot onto a KDE, distribution or boxplot chart.

* A histogram is a graphical display of data using bars of different heights. In a histogram, each bar groups numbers into ranges. Taller bars show that more data falls in that range. A histogram displays the shape and spread of continuous sampled data.
* A scatter plot is a graph in which the values of two variables are plotted against two axes. The pattern of the resulting points revealing any correlation that may be present.
* A heatmap is a graphical representation of data that uses a system of color coding to represent different values.
