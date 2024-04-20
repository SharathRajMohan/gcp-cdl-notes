# Infrastructure in the Google Cloud

## Where do I store this stuff?

### Storage options in the cloud

* A database is a collection of information that is organized so that it can easily be accessed and managed.

* A relational database is a type of database that stores and provides access to data points that are related to one another. They have a well defined structure and schema. And they are enforced at all times in a strict manner.

* Storage Offerings:
    * Relational Databases
    * Non-Relational Databases.
    * Worldwide object storage

* Google Cloud provides managed storage and database services that are scalable, reliable, and easy to operate. These are:
* Worldwide object storage:
    1. Cloud Storage
* Relational Database Services 
    1. Cloud SQL
    2. Cloud Spanner
* Non-Relational Database Services
    1. Firestore
    2. Cloud bigtable

* Cloud storage usecases:
1. Content storage and Delivery.
2. Data analytics and general compute
3. backup and Archival storage

* For users with databases, Google’s first priority is to help them migrate existing databases to the cloud and move them to the right service.The second priority is to help users innovate, build or rebuild for the cloud, offer mobile applications, and plan for future growth.


* Structured data: represents information stored in tables, rows, and columns.
* Structured data comes in two types: 
    * Transactional workloads : are used when fast data inserts and updates are required to build row-based records.
        * If your data is transactional and you need to access it using SQL, Cloud SQL and Cloud Spanner are two options.
            * Cloud Sql: works best for local to regional scalability
            * Cloud Spanner is best to scale a database globally
        * If the transactional data will be accessed without SQL, Firestore might be the best option.
            * Firestore is a transactional NoSQL, document-oriented database.
    * Analytical workloads: which are used when entire datasets need to be read.
        * If you have analytical workloads that require SQL commands, BigQuery may be the best option.
            * BigQuery: Google’s data warehouse solution, lets you analyze petabyte-scale datasets.
            * Cloud Bigtable: provides a scalable *NoSQL* solution for analytical workloads.

* Unstructured data is information stored in a non-tabular form such as documents, images, and audio files. Unstructured data is usually best suited to Cloud Storage.
* Unstructured Data can be stored using *Google Cloud Storage*
* Cloud Storage: is a fully managed scalable service that has a wide variety of uses. 
    * Cloud Storage’s primary use is whenever binary large-object storage (also known as a “BLOB”) is needed for online content such as videos and photos, for backup and archived data, and for storage of intermediate results in processing workflows.
* Object Storage: Object storage is a computer data storage architecture that manages data as “objects” and not as a file and folder hierarchy (file storage), or as chunks of a disk (block storage).
    * These objects are stored in a packaged format that contains the binary form of the actual data itself, relevant associated metadata (such as date created, author, resource type, and permissions), and a globally unique identifier.

* There are 4 basic cloud storage classes:
    1. Standard Storage: Standard Storage is considered best for frequently accessed, or “hot,” data.
    2. Nearline Storage: This is best for storing infrequently accessed data, like reading or modifying data once per month on average.
    3. Coldline Storage: Coldline Storage is meant for reading or modifying data, at most, once every 90 days.
    4. Archive Storage: for data that you plan to access less than once a year.

* Characteristics of cloud storage:
    * Unlimited storage
    * Worldwide access and locations
    * Low latency and high durability.
    * A uniform experience
    * Geo-redundancy (if data is stored in a multi-region or dual-region.)

* Cloud Storage files are organized into buckets.
    * A bucket needs a globally unique name and a specific geographic location for where it should be stored, and an ideal location for a bucket is where latency is minimized.
* The storage objects offered by Cloud Storage are “immutable,” which means that you do not edit them, but instead a new version is created with every change made.
* Administrators can either allow each new version to completely overwrite the older one or keep track of each change made to a particular object by enabling “versioning” within a bucket.
* Cloud Storage also offers lifecycle management policies for your objects. For example, you could tell Cloud Storage to delete objects older than 365 days, or to delete objects created before January 1, 2013, or to keep only the 3 most recent versions of each object in a bucket that has versioning enabled.
* Cloud Storage’s tight integration with other Google Cloud products and services means that there are many additional ways to move data into the service.
    * For example, you can import and export tables to and from both BigQuery and Cloud SQL.
    * Store App Engine logs, Firestore backups, and objects used by App Engine applications like images.
    * Cloud Storage can also store instance startup scripts, Compute Engine images, and objects used by Compute Engine applications.

* SQL Managed services:
    * Google Cloud offers two managed relational database services, Cloud SQL and Cloud Spanner.

* Cloud SQL: offers fully managed relational databases, including MySQL, PostgreSQL, and SQL Server as a service.

* Cloud Spanner: Cloud Spanner is a fully managed relational database service that scales horizontally, is strongly consistent, and speaks SQL.
    * Vertical scaling is where you make a single instance larger or smaller, while horizontal scaling is when you scale by adding and removing servers.
    * Cloud Spanner is especially suited for applications that require: 
        * An SQL relational database management system with joins and secondary indexes 
        * Built-in high availability 
        * Strong global consistency
        * And high numbers of input/output operations per second, such as tens of thousands of reads/writes per second.
    * Data is automatically and instantly copied across regions, which is called synchronous replication.
    * Google uses replication within and across regions to achieve availability, so if one region goes offline, a user’s data can still be served from another region.

* NoSQL Managed Services: Google offers two managed NoSQL database options, Firestore and Cloud Bigtable.

* Firestore is a fully managed, serverless NoSQL document store that supports ACID transactions.
    * Firestore is a flexible, horizontally scalable, NoSQL cloud database for mobile, web, and server development.
    * With Firestore, incoming data is stored in a document structure, and these documents are then organized into collections. 
    Documents can contain complex nested objects in addition to subcollections.
    * Firestore’s NoSQL queries can then be used to retrieve individual, specific documents or to retrieve all the documents in a collection that match your query parameters.
    * Queries can include multiple, chained filters and combine filtering and sorting options.
    * They're also indexed by default, so query performance is proportional to the size of the result set, not the dataset.
    * Firestore leverages Google Cloud’s powerful infrastructure: automatic multi-region data replication, strong consistency guarantees, atomic batch operations, and real transaction support.

* Cloud Bigtable is a petabyte scale, sparse wide column NoSQL database that offers extremely low write latency.
    * Bigtable is designed to handle massive workloads at consistent low latency and high throughput, so it's a great choice for both operational and analytical applications, including Internet of Things, user analytics, and financial data analysis.
    * Data can also be streamed in through various popular stream processing frameworks like Dataflow Streaming, Spark Streaming, and Storm. And if streaming is not an option, data can also be read from and written to Bigtable through batch processes like Hadoop MapReduce, Dataflow, or Spark.
    * The usual reason to store data in BigQuery is so you can use its big data analysis and interactive querying capabilities.
    
## There's an API for that
* What is an API?
* ANS: A clean well defined interface, whose underlying implementation can change, 

* APIs are used to simplify the way different, disparate, software resources communicate.

* REpresentational State Transfer, or REST, is currently the most popular architectural style for services.

* Cloud Endpoints is a distributed API management system that uses a distributed Extensible Service Proxy, which is a service proxy that runs in its own Docker container.
* Cloud Endpoints provides an API console, hosting, logging, monitoring, and other features to help you create, share, maintain, and secure your APIs.