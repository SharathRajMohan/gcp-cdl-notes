# 01 Digital Transformation with Google Cloud
* It's not just about technology, it's about how technology is used.
#### Introduction

* Digital transformation: when an organization uses new digital technologies, such as public, private, and hybrid cloud platforms to create or modify business processes, culture, and customer experiences to meet the needs of changing business and market dynamics.

Digital transformation helps organizations change how they operate and redefine relationships with their customers, employees, and partners by modernizing their applications, creating new services, and delivering value.

* Cloud: The cloud is a metaphor for the network of data centers which store and compute information that’s available through the internet.

* On-premises IT infrastructure, which is often abbreviated to “on-prem,” refers to hardware and software applications that are hosted on-site, located and operated within an organization's data center to serve their unique needs.

* A private cloud is a type of cloud computing where the infrastructure is dedicated to a single organization instead of the general public. This type is also known as single-tenant or corporate cloud.

* The public cloud is where on-demand computing services and infrastructure are managed by a third-party provider, such as Google Cloud, and shared with multiple organizations or “tenants” through the public internet. This sharing is why public cloud is known as multi-tenant cloud infrastructure, but each tenant’s data and applications running in the cloud are hidden from other tenants.

Because public cloud has on-demand availability of computing and infrastructure resources, organizations don't need to acquire, configure, or manage those resources themselves, and they only pay for what they use.

* There are typically three types of cloud computing service models available in public cloud:
    * Infrastructure as a service (IaaS), which offers compute and storage services.
    * Platform as a service (PaaS), which offers a develop-and-deploy environment to build cloud apps.
    * Software as a service (SaaS), which delivers apps as services, where users get access to software on a subscription basis.

* Other ways organizations can implement cloud: 
    * Hybrid Cloud: When the organization's applications run in a combination of different environments. (public+onPrem/private)
    * Multi Cloud: When the organization utilizes resources from multiple public cloud provider.

* Benefits of cloud computing:
    * Scalable: Cloud computing gives organizations access to scalable resources and the latest technologies on-demand, so they don’t need to worry about capital expenditures or limited fixed infrastructure.
    * Flexible: Organizations and their users can access cloud services from anywhere scaling services up or down as needed to meet business requirements.
    * Agile: Organizations can develop new applications and rapidly get them into production, without worrying about the underlying infrastructure.
    * Strategic Value: Because cloud providers stay updated with the latest innovations and offer them as services to customers, organizations can get more competitive advantages and a higher return on investment—than if they’d invested in soon-to-be obsolete technologies. This lets organizations innovate and try new ideas faster.
    * Secure: Cloud computing security is recognized as stronger than that in enterprise data centers, because of the depth and breadth of the security mechanisms and dedicated teams that cloud providers implement.
    * Cost-effective: No matter which cloud computing service model organizations implement, they only pay for the computing resources they use.

* The reality is that digital transformation is an ongoing process, not a one-time effort.

* It's critical that organizations embrace new technology as an opportunity to evolve, serve their customers better, and gain a competitive advantage. This is where cloud computing plays a significant role.

* Digital transformation is more than simply migrating and shifting systems to the cloud for cost saving and convenience.

* A transformation cloud is a new approach to digital transformation. Digitalization is now fundamental, and this era is about spreading transformation among all teams in an organization. It provides an environment for app and infrastructure modernization, data democratization people connections and trusted transactions.

* Biggest challenges and needs to accelerate digital transformation
    * they want to be the best at understanding and using data.
    * they want the best technology infrastructure.
    * they want to create the best hybrid workplace.
    * it's critical for organizations to know that their data, systems, and users are secure.
    * organizations are prioritizing sustainability as a critical, board-level topic.

* 5 Primary capabilities that form the basis of the transformation cloud:
    * Data
    * Open Infrastructure
    * Collaboration
    * Trust
    * Sustainable technology & Solutions

* A data cloud is a unified solution to manage data across the entire data lifecycle, regardless of whether it sits in Google Cloud or in other clouds.

* Organizations choose to modernize their IT systems on Google’s open infrastructure cloud because it gives them freedom to securely innovate and scale from on-premises, to edge, to cloud on an easy, transformative, and open platform. Instead of relying on a single service provider or closed technology stack, today most organizations want the freedom to run applications in the place that makes the most sense, using hybrid and multicloud approaches based on open source software.
    * Facilitates faster innovation 
    * Reduces lock-in to a single cloud provider
    * flexibility to build, migrate, and manage their applications across on-premises and multiple clouds.

* Open Standard vs Open Source:
    * Open standard refers to software that follows particular specifications that are openly accessible and usable by anyone. They have guidelines for software functionality, which help avoid vendor lock-in and ensure that the products that use these standards perform in an interoperable way.

    * Open source refers to software whose source code is publicly accessible and free for anyone to use, modify, and share.

* Another way we provide flexibility is through hybrid and multicloud environments managed by products like Anthos, which is built on open technologies like Kubernetes, Istio, and Knative.

* The Google Cloud adoption framework is more than just a model. It's also a map to real, tangible tasks that organizations need to adopt the cloud.

#### Fundamental Cloud Concepts

* Total Cost of Ownership: 
    * The cost of on-premises infrastructure is dominated by the initial purchase of hardware and software, but cloud computing costs are based on monthly subscriptions or pay-per-use models.
* Capex vs Opex: 
    * Capital expenditures, or CapEx, are upfront business expenses put toward fixed assets.
    * Operating expenses, or OpEx, which are recurring costs for a more immediate benefit. This represents the day-to-day expenses to run a business.
    * In the on-premises CapEx model, cost management and budgeting are a one-time operational process completed annually.
    * Moving to cloud’s on-demand OpEx model enables organizations to pay only for what they use and only when they use it.

* Network Performance: 
    * Bandwidth: is a measure of measure of how much data a network can transfer in a given amount of time. This ​​rate of data transfer is typically measured in terms of “megabits per second” (or Mbps) or “gigabits per second” (or Gbps). 
    * Latency: is the amount of time it takes for data to travel from one point to another. Often measured in milliseconds, latency, sometimes called lag, describes delays in communication over a network. Ideally, latency should be as close to zero as possible.
    * No matter how much data you can send and receive at once, it can only travel as fast as network latency allows.

* Google Cloud Regions & Zones:
    * Google Cloud’s infrastructure is based Google Cloud’s infrastructure is based in five major geographic locations: North America, South America, Europe, Asia, and Australia.
    * Regions: represent independent geographic areas and are composed of zones. Each Region has three or more zones
    * Zones: is an area where Google Cloud resources are deployed. Each Zone has one or more discrete clusters

    * You can run resources in different regions. This is useful for bringing applications closer to users around the world, and also for protection in case there are issues with an entire region, such as a natural disaster.
    * Some of Google Cloud’s services support placing resources in what we call a multi-region.

    * A network's edge is defined as a place where a device or an organization's network connects to the Internet. It's called "the edge" because it's the entry point to the network.
    * Google aims to deliver its services with high performance, high reliability and low latency for users.

#### Cloud computing models and Shared responsibility.
* In cloud computing, the cloud service provider owns, manages, and maintains the resources.
* Abstraction: As you move up the layers from one model to another, each model requires less knowledge and management of the underlying infrastructure. In cloud architecture, as the level of abstraction increases, less is known about the underlying implementation.

* IaaS: is a computing model that offers the on-demand availability of almost infinitely scalable infrastructure resources, such as compute, networking, storage, and databases as services over the internet.
    * IaaS allows organizations to lease the resources they need instead of having to buy hardware outright, and they only pay for what they use.
    * Compute Engine and Cloud Storage are examples of Google Cloud IaaS products.
    * The flexibility and scalability of IaaS is useful for organizations that: Have unpredictable workload volumes or need to move quickly in response to business fluctuations.

* PaaS:  is a computing model that offers a cloud-based platform for developing, running, and managing applications.  PaaS provides a framework for developers that they can build upon and use to create customized applications.
    * Cloud Run and BigQuery are examples of Google Cloud PaaS products.
        * Cloud Run: is a fully managed, serverless platform for developing and hosting applications at scale, which takes care of provisioning servers and scaling app instances based on demand.
        * Big Query: is a fully managed enterprise data warehouse that manages and analyzes data, and can be queried to answer big data questions with zero infrastructure management.
    * PaaS is suitable for organizations that: Want to create unique and custom applications without investing a lot in owning and managing infrastructure.
    * Container as a Service (CaaS) - Containers instead of apps
    * Function as a Service (FaaS) - Functions instead of apps
    * Databases

* SaaS: is a computing model that offers an entire application, managed by a cloud provider, through a web browser.
    * Google Workspace, which includes tools such as Gmail, Google Drive, Google Docs, and Google Meet, is a Google Cloud SaaS product.

* Simply put, the cloud provider is responsible for the security of the cloud, while the customer is responsible for security in the cloud.

* What is a MicroService?
* Services that have a specific goal of doing something. And multiple microservices make up an entire application. For example, An E-Commerce web application may consist of following microservices:

    * A backend MS
    * A frontend MS
    * A webserver MS to handle HTTP reqs
    * A database MS and so on...

    