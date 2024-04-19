## Notes
* Cloud Logging is not a place to retain or analyze external logs.
* Chronicle is a cloud service designed for enterprises to privately retain, analyze, and search the massive amounts of security and network telemetry they generate. Chronicle normalizes, indexes, correlates, and analyzes the data to provide instant analysis and context on risky activity.
* By dividing a large table into smaller partitions, you can improve query performance, and you can control costs by reducing the number of bytes read by a query.
    * With the on-demand pricing model in BigQuery, you are charged for the number of bytes processed by each query
* Some products like Cloud Run (serverless) are managed within Google Cloud infrastructure, but outside of customer VPCs.
* AppSheet is a no-code application development tool. It integrates with Google Sheets.
* Google Recommends to use Transfer Appliance when transferring over 60TB of data
* Cloud SQL supports MySQL, PostgreSQL, Microsoft SQL Server
* In ML, feature is input; label is output. A feature is one column of the data in your input set
* Managed GCP Services mapped with Open Source Services
```
DataFlow = Apache Beam (for data/batch processing)
DataFusion = CDAP
DataProc = Hadoop/Spark
Cloud Composer = Apache Airflow
```
* Compliance Report Manager consists of third-party audits and certifications, documentation, and contract commitments help support your compliance
* Google's Data Labeling Service lets you work with human labelers to generate highly accurate labels for a collection of data that you can use in machine learning models.
* Cloud Router enables you to dynamically exchange routes between your VPC and on-premises networks by using BGP
* Using WAAP (Web Application and API Protection) is the right protection plan: Anti-DDoS, anti-bot, WAF, and API protection help you protect against new and existing threats while helping you keep your apps and APIs compliant and continuously available.
* IAP (Identity Aware Proxy) lets you establish a central authorization layer for applications accessed by HTTPS, so you can use an application-level access control model instead of relying on network-level firewalls.
* Dataplex - Unified Data management across data lakes, data warehouses, and data marts
* Database Migration Service Available now for MySQL and PostgreSQL, with Oracle
* AlloyDB is a fully managed PostgreSQL-compatible database service
* VM instances that only have internal IP addresses (no external IP addresses) can use Private Google Access. They can reach the external IP addresses of Google APIs and services.
* Google provides unified billing for the open source tools that it has partenered with like MongoDB, Confluent, DataStax ...
* Leading software vendors provide virtual desktop solutions (VDI) on Google Cloud: Citrix, Itopia,
* When you enable committed use discount sharing, all of your current active committed use discounts in all the projects come under the same Cloud Billing account
* Notebooks is a managed service that offers an integrated and secure JupyterLab environment for data scientists
* Google Cloud Directory Sync integrates with most LDAP management systems and can synchronize identities like Active Directory
* In Google's CDN service, your sites gets a single global IP (anycast IP)
* Bare Metal machines give you the ability to install any software but Sole-tenant nodes are still virtualized
* Cloud Trace is used for analyzing latency
* Titan Security Keys provide the highest level of security more than MFA using Mobile app
* VPC Network Peering allows internal IP address connectivity across two Virtual Private Cloud (VPC) networks regardless of whether they belong to the same project or the same organization.
    * Shared VPC is only within an organization - it allows an organization to connect resources from multiple projects to a common Virtual Private Cloud (VPC) network, so that they can communicate with each other securely and efficiently using internal IPs from that network.
* Bring your own IP (BYOIP) lets you provision and use your own public IPv4 addresses for Google Cloud resources
* Cloud Identity Platform allows you to manage identity and credentials for your consumer facing applications. Wheread Cloud * Identity is for the enterprise. It provides an email id with the org domain name, but with no access to Google Workspace tools like Docs, Sheets, Slides, etc.
* BigQuery also supports streaming data and its possible to do real time analytics on it.
* You can associated a Google Billing Account with a GCP organization or GCP Projects. There is no concept of associating a billing account with a folder
* Data Studio is a free tool and integrates well with BigQuery. Both creating and viewing reports are free.
* Migrate for Compute Engine’s advanced replication migration technology copies instance data to Google Cloud in the background with no interruptions to the source workload that’s running.
* Operations Suite provides integrated monitoring, logging, and trace managed services for applications and systems running on Google Cloud and beyond.
* Lending DocAI is a pre-packaged AI solution that speeds "up the mortgage workflow processes to easily process loans and automate document data capture, while ensuring the accuracy and breadth of different documents (e.g., tax statements and asset documents)."
* Cloud Scheduler is a fully managed enterprise-grade cron job scheduler.
* Data is encrypted by Google Cloud at rest and in transit on Google Cloud. However, encryption at time of processing is optional.
    * Confidential Compute is the option that allows you to encrypt data in use—while it’s being processed.