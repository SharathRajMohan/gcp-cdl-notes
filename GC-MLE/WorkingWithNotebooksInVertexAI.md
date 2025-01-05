# Working with Notebooks in Vertex AI
=====================================
## Introduction

In this course, You learn to: 
- Explain Vertex AI Notebook solutions.
- Describe Vertex AI Colab Enterprise notebooks
- Describe Vertex AI Workbench Instances Notebooks 
- Use Vertex AI Notebook Solutions.

## Vertex AI Notebook Solution

The ML Worflow has three major phases:
    * Data Preparation: In data preparation, you’re collecting data based upon a business problem or use case and preparing it for input into a machine learning model.
    * Model Training: In model training, you’re using the data you collected to inform predictions for the use case.
    * Model Serving: serve that model into a production environment so that it can be used to make real-world decisions or predictions.

There are three tools from GCP that you can use for carrying out ML workflows:
* AutoML: which allows you to implement the workflow without writing a single line of code.
* BigQuery ML: which allows you to use SQL (or Structured Query Language) to implement the model training and serving phases
* Custom Training:  which allows you to use a programming language such as Python or TensorFlow to implement every phase of the workflow. Requires writing code.

Google Vertex AI offers a one stop shop for managing the machine learning workflow.

Vertex AI Notebooks offers: 
* Fully managed compute with admin control: A Jupyter-based fully managed, scalable, enterprise-ready compute infrastructure with easily enforceable policies and user management.
* Fast workflow for data task: Seamless visual and code-based integrations with data & analytics services And at-your-fingertips integration, meaning you can load and share notebooks alongside your AI and data tasks.4
* Accessible Integration: load and share notebooks alongside your AI and data tasks. Run tasks without extra code.

Vertex AI provides two notebook solutions:
* Colab Enterprise:
    - zero config, serverless and collaborative.
    - Managed compute. No need to manage the infrastructure.
    - Great for projects that can be encapsulated to a single notebook.
    - Built in version control. No need to use a separate tool.
* Vertex AI Workbench instances
    - Built on standard jupyter lab.
    - Flexible, familiar and customizable.
    - Great for projects spanning multiple files, with complex dependencies.
    - Native git support.

Let’s briefly look at how Vertex AI Notebook solutions help manage the machine learning workflow.
* Data Preparation: Vertex AI Jupyter notebooks facilitate seamless data access and integration with Google Cloud Storage, which allows for direct retrieval and manipulation of datasets stored in the cloud. Additionally, integration with services like BigQuery, Dataproc, and Spark streamlines data ingestion, preprocessing, and exploration tasks.

* Model Training: Vertex AI Notebooks provide a scalable and managed environment for training machine learning models. They come with pre-installed libraries and frameworks, such as TensorFlow, PyTorch, and scikit-learn. Allowing users to run different machine learning model experiments, perform hyperparameter tuning and model evaluations, all can be  done in the library you choose.

* Model Deployment: Vertex AI Jupyter notebooks streamline model deployment to production environments by using Vertex AI Pipelines, which automates the deployment process and ensures consistency between training and production environments. Vertex AI Monitoring allows for continuous monitoring of model performance in production, alerts users to potential issues, and ensures optimal model performance over time.

### Vertex AI Colab Enterprise

With colab:
- Users can start operating with zero-configuration experience.
- Be more productive with native integration with Duet AI.
- easy collaboration capabilities like IAM based notebook sharing, commenting, and co-editing, achieve greater productivity and deployment speed.


Vertex AI Colab Enterprise components consist of:
- A Notebook Editor : The notebook editor allows you to edit and execute notebooks. When you execute a notebook cell the system automatically connects to an ipython kernel on a runtime. The code is executed by that kernel.
- Notebook storage: Colab Enterprise uses Dataform to store notebooks (ipynb files). Colab Enterprise’s notebook storage offers similar functionality to Google Drive: versioning and sharing.
- Runtimes: Are VMs that your code executes on. They come in two varieties:
* Default runtimes (Pre-defined runtimes): They are ephemeral, meaning they are deleted
* Templatized runtimes (Long lived runtimes): They are not deleted when you stop them. They are useful for long-running. These runtimes are flexible, configurable, and long-lived. Administrators can create them for users to use and create new runtimes. It is recommended to use templatized runtimes in the following situations:
    * When you need to use GPUs.
    * When you need to use large machine shapes.
    * Install packages.

### Vertex AI Workbench instance Notebooks

Vertex AI Workbench is a Jupyter notebook-based environment provided through virtual machine (VM) instances with features that support the entire data science workflow.

Vertex AI Workbench instances are built on standard Jupyter Lab and are flexible, familiar, and customizable.

A Vertex AI Workbench Instance gives you a collaborative environment and version control when you use GitHub.

Vertex AI Workbench Instances comes with a preinstalled suite of deep learning packages, including support for the TensorFlow and PyTorch frameworks.

Dataproc is a fully managed cloud service for running big data processing, analytics, and machine learning workloads on Google Cloud.

Vertex AI Workbench is a Jupyter notebook-based environment provided through virtual machine (VM) instances that supports the entire data science workflow. It's particularly suitable for projects that require control and customizability, such as complex projects with multiple files and dependencies, or for data scientists transitioning to the cloud from local workstations.

**Key Features:**

- **Collaborative Environment:** Integrates with GitHub, allowing for code collaboration, version control, and project management.

- **Preinstalled Deep Learning Packages:** Includes support for TensorFlow and PyTorch frameworks, facilitating machine learning model development.

- **Storage and Execution:** Enables storage of ML models, features, and training sets, and supports running ML applications.

- **Notebook Editor and Storage:** Provides a notebook editor and storage, similar to Colab Enterprise, with additional customization options.

**Customization Options:**

Vertex AI Workbench Instances offer extensive customization to tailor the environment to specific workflows and project requirements.

**Creating a New Instance with Advanced Options:**

1. **Configure Instance Details:**
   - Assign a meaningful name and select the geographical location.
   - Enable access to the Dataproc kernel for big data processing.
   - Add labels and tags for resource identification and network management.

2. **Configure Instance Environment:**
   - Uses JupyterLab 3 by default, with the latest NVIDIA GPU and Intel libraries.
   - Option to specify previous versions and add metadata.

3. **Select Machine Type:**
   - Determine specifications like memory, virtual cores, and disk limits.
   - Ensure compatibility with GPUs if needed.
   - Option to enable Shielded VM for enhanced security.
   - Enable idle shutdown to reduce costs during inactivity.

4. **Select Data Disk Type:**
   - Choose between standard persistent disks (cost-effective) and SSDs (higher performance).
   - Specify disk size and consider encryption for data security.

5. **Select Network Values:**
   - Assign an external IP address or enable Private Google Access for internet connectivity.

6. **IAM and Security Options:**
   - Define access permissions using Identity and Access Management (IAM).
   - Decide on service accounts and security features like terminal access.

7. **Configure System Health:**
   - Enable environment auto-upgrade and system health reporting.
   - Install Cloud Monitoring for system and application metrics.

**Modifying Instance Configurations:**

- To add a GPU, stop the instance, modify the hardware configuration by selecting the GPU type and quantity, then submit the changes.

**Integration and Support:**

- **BigQuery Integration:** Access BigQuery tables and author SQL queries directly from JupyterLab.

- **Dataproc Support:** Create, manage, and run computations on Dataproc clusters within the environment.

**Notebook Customization:**

- Control main menu selections, with frequently used items like "Run" and "Kernel" to efficiently execute code and manage the kernel.

## Summary

Vertex AI Jupyter notebooks provide a unified, collaborative, and scalable environment for data preparation, model development, deployment, and monitoring, enhancing the machine learning workflow for practitioners of all levels.

**Key Features:**

- **Custom Training Support:** Ideal for complex use cases requiring custom coding, such as modifying Generative AI Studio prompts using the Python SDK, or building and training models with PyTorch, Python, or TensorFlow.

- **Collaboration and Code Sharing:** Facilitates sharing and collaboration during data preparation, model training, and deployment to production environments.

- **Integration with Google Cloud AI Services:** Seamlessly integrates with services like Vertex AutoML for automated model development, Vertex AI Predictions for real-time model inference, and Vertex AI Explainable AI for interpreting model predictions, providing a comprehensive suite of tools for machine learning tasks.

- **Scalability and Resource Management:** Allows scaling computational resources up or down based on needs, ensuring optimal resource utilization and cost-effectiveness with pay-as-you-go pricing.

- **Security and Reliability:** Built on Google Cloud's secure and reliable infrastructure, ensuring data privacy, protection, and integrity, with regular security updates and patches to maintain a secure environment and mitigate cyber threats.

**Use Cases:**

- **Data Analysts:** Utilize Vertex AI Notebooks for data analysis tasks.

- **Data Scientists:** Employ Vertex AI Notebooks to train machine learning models.

- **Machine Learning Engineers:** Use Vertex AI Notebooks to deploy machine learning models into production.

By leveraging Vertex AI Jupyter notebooks, professionals can streamline their machine learning workflows without the need to manage underlying infrastructure, allowing them to focus on developing and deploying effective models.

 

 





