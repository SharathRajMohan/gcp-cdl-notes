# Exploring Data Transformation with Google Cloud

* Data powers AI-driven business insights, helps companies make better real-time decisions, and is the basis for how companies build and run their applications.
* With machine learning, or ML, and artificial intelligence, or AI, organizations can generate insights from data both past and present.
* An intelligent data Cloud is the key to unlocking more business value


* Data can be categorized into three main types
    * structured
    * semi-structured
    * unstructured

* Data management concepts:
    * Database: is an organized collection of data stored in tables and accessed electronically from a computer system.
        * Relational: can establish links or relationships between information by joining tables, and structured query language or SQL can be used to query and manipulate data.
        * Google Cloud relational database products include 
            * Cloud SQL 
            * Cloud Spanner, while Bigtable is a non relational database product.
        * Non-Relational: is less structured in format and doesn't use a tabular format of rows and columns like relational databases.
    * Data warehouse:  is an enterprise system used for the analysis and reporting of structured and semi-structured data from multiple sources.
        * BigQuery is Google Cloud's data warehouse offering.
        * Although data warehouses handle structured and semi structured data, they're not typically the answer for how to handle large amounts of available unstructured data like images, videos, and documents.
    * Data lakes:  is a repository designed to ingest, store, explore, process, and analyze any type or volume of raw data, regardless of the source, like operational systems, web sources, social media or Internet of things or IoT.
        * It can store different types of data in its original format, ignoring size limits and without much preprocessing or adding structure.
        * This differs from a data warehouse that contains structured data that has been cleaned and processed ready for the strategic analysis based on predefined business needs.
        * Data lakes often consist of many different products depending on the nature of the data that is ingested.

* First party data: is the proprietary customer datasets that a business collects from customer or audience transactions and interactions.
* Second party data:  often describes first-party data from another organization, such as a partner or other business in their supply chain that can be easily deployed to augment a company's internal datasets.
* third-party data: which are datasets collected and managed by organizations that do not directly interact with an organization's customers or business.

* Data value chain:
    1. Data genesis
    2. Data collection: brings that initial unit of data to the assembly line through ingestion.
        * The basic function of ingestion is to extract data from the system in which it's hosted and bring it to a new system.
    3. Data processing: is where the collected raw data is transformed into a form that's ready to derive insights from.
    4. Data storage: is where the data lands can be found and is ready for analysis and action.
    5. Data activation

### Google Cloud Data Management Solutions:

* Unstructured data storage:   
    * Google Cloud Storage
        * 4 kinds of storage classes:
            * Standard
            * Nearline
            * Coldline
            * Archival
        * Cloud storage also provides a feature called auto class which automatically transitions objects to appropriate storage classes based on each object's access pattern.
