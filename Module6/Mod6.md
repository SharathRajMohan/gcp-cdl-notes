# 06 Scaling with Cloud Operations
#### Introduction
 * “Scaling with Google Cloud Operations” was designed to help you learn how Google Cloud supports an organization's ability to control their cloud costs through financial governance, understand the fundamental concepts of modern operations, reliability, and resilience in the cloud, and explore how Google Cloud works to reduce our environmental impact and help organizations meet sustainability goals.

#### Financial Governance and Managing Cloud Costs
* fundamentals of cloud cost management
* cloud financial governance best practices
* ways to control access by using the resource hierarchy
* ways to control cloud consumption.

##### Fundamentals of cloud financial governance
* Cloud Financial Governance: A set of processes and controls that organizations use to manage cloud spend.
* As an organization adapts, it'll need a core team across technology, finance, and business functions to work together to stay on top of cloud costs and make decisions in real time.
* The variable nature of cloud costs impacts people, process, and technology.
* To manage cloud costs effectively, a partnership across finance, technology, and business functions is required.
* This partnership might already exist, or it may take the form of a centralized hub, such as a cloud center of excellence.
##### Cloud financial governance best practices
1. Identify who manages cloud costs
    * Defining clear ownership for projects and sharing cost views with the departments and teams that are using cloud resources helps establish this accountability culture and more responsible spending.
    * Google Cloud offers flexible options to organize resources and allocate costs to individual departments and teams.
    * Creating multiple budgets with meaningful alerts is an important practice for staying on top of your cloud costs.
2. Understand invoices versus cost management tools
    * An invoice is a document that is sent by a cloud service provider to a customer to request payment for the services that were used.
    * Google cloud cost management tools: They can provide granular data, uncover trends, and identify actions to take to control or optimize costs.
3. Use Google Cloud's cost management tools
    * Google Cloud Pricing Calculator: Pricing Calculator lets you estimate how changes to cloud usage will affect costs.

##### Using resource hierarchy to control access
* The Google Cloud resource hierarchy is a powerful tool that can be used to control access to cloud resources.
* Google Cloud’s resource hierarchy contains four levels, and starting from the bottom up they are:
    * Level 01: Resources --> virtual machines, Cloud Storage buckets, tables in BigQuery, or anything else in Google Cloud.
    * Level 02: Projects --> Resources are organized into projects
    * Level 03: Folders --> Projects can be organized into folders, or even subfolders.
    * Level 04: Organization node --> encompasses all the projects, folders, and resources in your organization.
* Benefits of GC Resource hierarchy
    1. Granular access control: can assign roles and permissions at different levels of the hierarchy, such as at the folder, project, or individual resource level.
    2. Inheritance and propagation rules: permissions set at higher levels of the resource hierarchy are automatically inherited by lower-level resources.
    3. Security and Compliance: (least privilege principles) assigning access permissions at the appropriate level in the hierarchy, you can ensure that users only have the necessary privileges to perform their tasks.
    4. Strong visibility and auditing capabilities: You can track access permissions and changes across different levels of the hierarchy, which makes it easier to monitor and review access controls.

##### Controlling Cloud Consumptions
* Google Cloud offers several tools to help control cloud consumption, including resource quota policies, budget threshold rules, and Cloud Billing reports.
* Resource quota policies: let you set limits on the amount of resources that can be used by a project or user.
* Budget threshold rules: which let you set alerts to be informed when your cloud costs exceed a certain threshold.
* Both resource quota policies and budget threshold rules are set in the Google Cloud console.
Cloud Billing Report: help you track and understand what you’ve already spent on Google Cloud resources and provide ways to help optimize your costs.

* If your workloads have predictable resource needs, you can purchase a Google Cloud commitment, which gives you discounted prices in exchange for your commitment to use a minimum level of resources for a specific term.

#### Operational Excellence and Reliability at Scale
* Operational excellence and reliability refers to the ability of organizations to optimize their operations and ensure uninterrupted service delivery, even as they handle increasing workloads and complexities in the cloud.

##### Fundamentals of cloud reliability
* DevOps is a software development approach that emphasizes collaboration and communication between development and operations teams to enhance the efficiency, speed, and reliability of software delivery.
* Site Reliability Engineering:  ensures the reliability, availability, and efficiency of software systems and services deployed in the cloud.
* Monitoring is the foundation of product reliability.
* 4 golden signals that measure a system's performance and reliability:
    1. Latency: measures how long it takes for a particular part of a system to return a result.
        
        a. It directly affects the user experience.

        b. Changes in latency could indicate emerging issues.

        c. Can estimate capacity demands.

        d. Can be used to measure system improvements.

    2. Traffic: Measures how many requests reach your system.
        
        a. Indicator of current system demands.

        b. Its historical trends are used for capacity planning.

        c. Its a core measure when calculating infrastructure spend.

    3. Saturation: measures how close to capacity a system is
        
        a. It's and indicator of how full the service is.

        b. It focuses on the most constrained resources.

        c. It's frequently tied to degrading performance as capacity is reached.

    4. Errors: are often raised when a flaw, failure, or fault in a computer program or system causes it to produce incorrect or unexpected results, or behave in unintended ways.
        
        a. they can indicate something is failing

        b. configuration or capacity issues

        c. service level objective violations

        d. time to send an alert.

* Three main concepts in site reliability engineering are :
    * Service-level indicators (SLIs): Service level indicators are measurements that show how well a system or service is performing.

    They’re specific metrics like response time, error rate, or percentage uptime–which is the amount of time a system is available for use–that help us understand the system's behavior and performance.
    * Service-level objectives (SLOs): are the goals that we set for a system's performance based on SLIs.

    They define what level of reliability or performance that we want to achieve.
    * Service-level agreements (SLAs): are agreements between a cloud service provider and its customers.They outline the promises and guarantees regarding the quality of service.

    SLAs include the agreed-upon SLOs, performance metrics, uptime guarantees, and any penalties or remedies if the provider fails to meet those commitments.

Note: They are all types of targets set for a system’s Four Golden Signal metrics.

##### Designing resilient infrastructure and processes

* When infrastructure and processes in a cloud environment are designed, they need to be resilient, fault-tolerant, and scalable, for high availability and disaster recovery.
* High availability: refers to the ability of a system to remain operational and accessible for users even if hardware or software failures occur.
* Disaster Recovery: refers to the process of restoring a system to a functional state after a major disruption or disaster.

* Key design considerations and their significance when designing infrastructure and processes in a cloud environment:
    * Redundancy: refers to duplicating critical components or resources to provide backup alternatives. Redundancy enhances system reliability and mitigates the impact of single points of failure.
    * Replication: involves creating multiple copies of data or services and distributing them across different servers or locations.
    * Regions: By distributing resources across regions, businesses can ensure that if an entire region becomes unavailable due to natural disasters, network issues, or other incidents, their services can continue running from another region. This approach improves resilience and reduces the risk of prolonged service interruptions.
    * Scalable Infrastructure: allows organizations to handle varying workloads and accommodate increased demand without compromising performance or availability.
    * Backups: Regular backups of critical data and configurations are crucial to ensure that if data loss, hardware failures, or cyber-attacks occur, organizations can restore their systems to a previous state.

    * It’s important to regularly test and validate these processes to ensure that they function as expected during real-world incidents.

    * Also, monitoring, alerting, and incident response mechanisms should be implemented to identify and address issues promptly, further enhancing the overall resilience and availability of the cloud infrastructure.

##### Modernizing operations by using Google Cloud
* Google's integrated observability tools.
* Observability involves collecting, analyzing, and visualizing data from various sources within a system to gain insights into its performance, health, and behavior.
* Google Cloud's operations suite: a comprehensive set of monitoring, logging, and diagnostics tools. It offers a unified platform for managing and gaining insights into the performance, availability, and health of applications and infrastructure deployed on Google Cloud.
* Managed services that constitute the operations suite:
    * Cloud Monitoring : 
        
        i. Provides a comprehensive view of cloud infrastructure and applications.

        ii. Collects metrics, logs and traces from applications and infrastructure, and provides insights into their performance,
        health and availability.

        iii. Lets you create alerting policies to notify you when metrics, health check results, and uptime check results meet specified criteria. 

    * Cloud Logging:  collects and stores all application and infrastructure logs. With real-time insights, you can use Cloud Logging to troubleshoot issues, identify trends, and comply with regulations.
    * Cloud Trace: helps identify performance bottlenecks in applications. It collects latency data from applications, and provides insights into how they’re performing.
    * Cloud Profiler: identifies how much CPU power, memory, and other resources an application uses. It continuously gathers CPU usage and memory-allocation information from production applications and provides insights into how applications are using resources.
    * Error Reporting: counts, analyzes, and aggregates the crashes in running cloud services in real-time. A centralized error management interface displays the results with sorting and filtering capabilities. A dedicated view shows the error details: time chart, occurrences, affected user count, first- and last-seen dates, and a cleaned exception stack trace. Error Reporting supports email and mobile alerts notification through its API.


##### Google Cloud Customer Care
* Any cloud adoption program can encounter challenges, so it's important to have an effective and efficient support plan from your cloud provider.

* There are four different service levels, which lets you choose the one that’s right for your organization.
    * Basic support
    * Standard Support
    * Enhanced Support
    * Premium Support

##### The life of a support case.
* Any Google Cloud customer on the Standard, Enhanced, or Premium Support plan can use the Google Cloud console to create and manage support cases.
* Life of a support case:
    1. Case creation: First, the customer initiates the support request by creating a case in the Google Cloud Console. Only users who were assigned the Tech Support Editor role within an organization can do this.
    2. Case triage: The team reviews the information provided by the customer to understand the problem and determine its severity and impact on the customer's business operations.
    3. Case assignment
    4. Troubleshooting and investigation: They analyze the provided information, review system logs, and conduct various diagnostic tests to identify the root cause of the issue.
    5. Communication and updates
    6. Escalation:  is meant for flagging process breaks or for the rare occasion that a case is stuck because a customer and the Customer Care team aren’t fully in sync, despite actively communicating the issue to determine the next steps.
    Escalation is a tool that can be used to regain traction on a stuck case.
    7. Resolution and mitigation
    8. Validation and testing
    9. Case closure

#### Sustainability with Google Cloud
* Altogether, existing data centers use nearly 2% of the world’s electricity.
*  Google's data centers were the first to achieve ISO 14001 certification, which is a standard that outlines a framework for an organization to enhance its environmental performance through improving resource efficiency and reducing waste.
* Google’s data center in Hamina, Finland is one of the most advanced and efficient data centers in the Google fleet. Its cooling system, which uses sea water from the Bay of Finland, reduces energy use and is the first of its kind anywhere in the world.
* In our founding decade, Google became the first major company to be carbon neutral.
* In our second decade, we were the first company to achieve 100% renewable energy.
* And by 2030, we aim to be the first major company to operate completely carbon free.
* BigQuery and Looker Studio dashboards provide granular insights.

