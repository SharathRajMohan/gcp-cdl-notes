# 04 Modernize Infrastructure and Applications with Google Cloud

#### Introduction
* Scale is the norm.
* How do orgs adopt or move to cloud?
* Can leverage clouds to develop new apps that are more advanced and efficient.
* What this module focuses on:
    - Common terminology related to cloud and application modernization.
    - Options to run compute workloads in the cloud, VMs, Containers & Serverless architectures
    - Modernize Application development through rehosting and APIs

* Important Cloud Migration Terms
1. Workload: A specific application, service or capability that can be run in the cloud or OnPrem. Eg. Container, DBs, VMs.
2. Retired: Means removing a workload from the platform.
3. Retained: Workloads that are Intentionally kept, either on premises or hybrid cloud environment by users.
4. Rehosted: Workloads that are migrated to the cloud without any changes in the code or architecture. AKA Lift n Shift
5. Replatform: Migrating the workload to the cloud, while making some changes to the code or architecture. AKA MovenImprove.
6. Refactored: Process of changing the code of a workload.
7. Reimagined: Reimagine refers to the process of rethinking how an organization uses technology to achieve its business goals. It can involve changing stratergy to accomodate new and improved technologies and cloud services.

#### Modernizing Infrastructure in the cloud.
##### Introduction
* Compute: A machine's ability to process information.
* Associated tasks include storing, retrieving, comparing, and analyzing the information.
* In this section of the course, you'll learn about the benefits that Cloud computing can bring to an organization and explore three Cloud computing options:
    - virtual machines
    - containers 
    - serverless
* Benefits of running compute workloads in the cloud.
    - Total Cost of Ownership (TCO): Pay for only what you use and long term commitment discount.
    - Scalability: Ability to increase or decrease resources to meet changing demands with time.
    - Reliability: High degree of reliability and Uptime.
    - Security: 
        - data encryption
        - identity and access management
        - network security
        - virtual private Clouds
        - monitoring services that can detect and respond to security threats in real time.
    - Flexibility: Organizations can choose the Cloud services that best meet their needs at any point in time, and then change or adapt those services when necessary.
    - Abstraction: refers to how Cloud providers remove the need for customers to understand the finer details of the infrastructure implementation by providing management of the hardware, software, and certain aspects of security and networking.

##### Virtual Machines 

* Virtualization: is a form of resource optimization that lets multiple systems run on the same hardware. These systems are called VMs. They share the same pool of processing, storage and networking resources.

* Compute Engine (GCE): #IaaS allows users to create and run VMs on google Infrastructure.

* Characteristics of GCE:
    * No upfront investments.
    * Can be configured in terms of :
        * CPU power and memory.
        * Storage needed.
        * OS needed.
* A GCE VM can be instantiated either using:
    * Cloud Console.
    * Cloud CLI: Infra automation tools such as Terraform or Compute Engine API.

* An Application Programming Interface (API) is a set of instructions that allows different software programs to communicate with each other.

* Pre-emptible or SpotVM: Is a kind of GCEVM which has the permission to terminate the VM if its resources are needed elsewhere. Remember to always setup backups while using such VMs. They can only run for 24hrs.

##### Containers
* Containers: provide isolated environments to run software services and optimize resources from one piece of hardware. Containers only virtualize the software layers above the operating system level.
* They start faster and come with less overhead in terms of memory when compared to a VM that has to boot an entire OS.
* A container is packaged with your application and all of its dependencies, so it has everything it needs to run.
* Containers can be independently developed, tested, and deployed, and are well suited for a microservices based architecture.
* This architecture is made up of smaller individual services that run containerized applications, that communicate with each other through APIs or other lightweight communication methods, such as REST or gRPC.
* Containers can run virtually and anywhere, which makes development and deployment easy.

* Features of Containers: 
    * Improve agility
    * Enhance Security
    * Optimize Resources
    * Simplify managing applications in the cloud.

##### Kubernetes

* Kubernetes: An open-source platform for managing containerizedd workloads and services.
* Kubernetes allows managing many containers by: 
    * Allowing them to run on many hosts.
    * Scale them based on demand.
    * Deploy Rollouts and rollbacks
* Hence, Kubernetes improves reliability, time and resources.

* Google Kubernetes Engine (GKE): Google Kubernetes Engine or GKE is a Google hosted, managed Kubernetes service in the Cloud. The GKE environment consists of multiple machines, specifically compute engine instances grouped to form a cluster. GKE clusters can be customized, and they support different machine types, numbers of nodes, and network settings.
* GKE makes it easy to deploy applications by providing an API and a Web based console.
* GKE also provides many features that can help monitor applications, manage resources, and troubleshoot problems.
* GKE-Autopilot: A mode that enables full management of an entire cluster's infrastructure and provides per-pod billing.

* Google Cloud Run: is a fully managed serverless platform to deploy and run containerized applications without needing to worry about the underlying infrastructure. 
* GCR facilitates automatic scaling of applications and managing underlying infrastructure.
* In summary, GKE is ideal when lots of control is required over a Kubernetes Environment and there are complex applications to run.Alternatively, Cloud Run is ideal for when a simple, fully managed serverless platform that can scale up and down quickly is required.

##### Serverless Computing
* Serverless computing refers to automatic provisioning of compute power on the fly in the background.

* One type of serverless computing solution is called 'function as a service'.

* Google's serverless computing products:
    1. Cloud Run: A fully managed environment for running containerized apps.
    2. Cloud Functions: the platform for hosting simple single purpose functions that are attached to events emitted from your Cloud infrastructure and services.
    3. App Engine: service to build and deploy web applications.
* Benefits of serverless compute:
    * Reduced operational costs.
    * Cloud provider handles infrastructure management and maintenance.
    * Provides automatic scaling of computing resources based on the applications demand.
    * The development process is simplified because developers can focus on the application's logic and not on the underlying infrastructure.
    * Serverless computing offers improved resilience and availability as the Cloud provider automatically manages the infrastructure's failover and disaster recovery capabilities.


#### Modernizing Applications in the cloud.
* Application: an application is a computer, program or software that helps users do something.
* With cloud technology, businesses can modernize, develop, and manage applications in new ways, which makes them more agile and responsive to user needs.
##### Benefits of modern cloud application development.
* With modern cloud application development, software development is flexible, scalable, and uses the latest cloud computing technologies to build and deploy applications.
* Monolithic application development: #Traditional required all the components of an application to be developed and deployed as a single, tightly coupled unit, typically using a single programming language.
* Benefits of modern cloud application development approach:
    * Architecture: Applications are decoupled or broken down into collection of microservices. These microservices are independently deployed, scaled and maintained.
    * Deployment: Managed services take care of the day-to day management of cloud based infrastructure, such as patching, upgrades and monitoring.
    * Cost: Cost effective, since its a pay as you go model.
    * Scalability
    * High availability
    * In built load balancing
    * Automatic Failover
##### Rehosting legacy applications in the cloud.
* Legacy applications: Are applications that have been developed and deployed on premises.
* Cloud Native approach: Cloud native is the software approach of building, deploying, and managing modern applications in cloud computing environments.
* Cloud Native Applications: Cloud-native applications are software programs that consist of multiple small, interdependent services called microservices. 
* Rehost Migration Path: #LiftandShift an application is moved from an on premises environment to a cloud environment without making any changes to the application itself.
* Benefits of rehosting: 
    * Cost savings.
    * Scalability.
    * Reliability.
    * Security.

* Drawbacks of rehosting:
    * Complexity
    * Risk
    * Vendor Lockin
* Google's solutions for rehosting specialized legacy applications
    * Google Cloud VMware Engine: helps migrate existing VMware workloads to the cloud without having to re architect the applications or retool operations.
    * Bare Metal solution: For organizations with legacy applications on oracle. This is a fully managed cloud infrastructure solution that lets organizations run their Oracle workloads on dedicated bare metal servers in the cloud.
##### Application Programming Interface (APIs)
* An API is a set of instructions that lets different software programs communicate with each other.
* Think of it as an intermediary between two different programs, which provides a standardized and predictable way for them to exchange data and interact.

* Apigee API Management: Google cloud's API management service to operate APIs with enhanced scale, security and automation.
* Benefits of Apigee:
    * It helps organizations secure their API's by providing features such as authentication, authorization and data encryption.
    * It tracks and analyzes API usage with real time analytics and historical reporting.
    * It helps with developing and deploying API's through a visual API editor and a test sandbox.
    * It offers API versioning, API documentation, and even API throttling, which is the process of limiting the number of API requests a user can make in a certain period.
##### Hybrid and multi-cloud
* Hybrid Cloud: A hybrid cloud environment comprises some combination of on premises or private cloud infrastructure and public cloud services.
* Multi Cloud: A multi-cloud environment is where an organization uses multiple public cloud providers as part of its architecture.

* Benefits of hybrid or multi-cloud:
    * Keep parts of the system's infrastructure on-premises while they move other parts to the cloud.
    * Move only specific workloads to the cloud
    * Flexibility, scalability and lower computing costs.
    * Add specialized services to the organizations's computing resources toolkit.

* GKE Enterprise: Production-ready platform for running Kubernetes applications across multiple cloud environments. It provides a consistent way to manage Kubernetes, clusters, applications and services regardless of where they are running.
* Benefits of GKE Enterprise:
    * Multi-cloud & hybrid-cloud support.
    * Centralized management GKE Enterprise provides a single centralized console for managing Kubernetes clusters and applications, security and compliance.









