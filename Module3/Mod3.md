# 03 Innovating with Google Cloud Artificial Intelligence

#### AI and ML Fundamentals
* Artificial intelligence is a broad field which refers to the use of technologies to build machines and computers that can mimic cognitive functions associated with human intelligence.These functions include, being able to see, understand, and respond to spoken or written language, analyze data, make recommendations and more.
* AI is a set of technologies implemented in a system to let it reason, learn, and act to solve a complex problem.
* Machine Learning is a subset of AI that lets a machine learn from data without being explicitly programmed.
* Generative AI:  is a type of artificial intelligence that can produce new content, including text, images, audio, and synthetic data.
* ML is suited to solve four common business problems:
    1. Replacing rule based systems
    2. Automating processes
    3. Understanding unstructured data like images, videos, and audio.
    4. Personalization

* It's important to remember that ML models aren't standalone solutions and that solving complex business challenges requires combinations of models.

* Data must adhere to the following  in order to be of high quality:

    * Completeness : The completeness of data refers to whether all the required information is present.
    * uniqueness : If a model is trained on a data set with a high number of duplicates, the ML model may not be able to learn accurately.
    * timeliness : refers to whether the data is up-to-date and reflects the current state of the phenomenon that's being modeled.
    * validity : data conforms to a set of predefined standards and definitions, such as type and format.
    * accuracy : reflects the correctness of the data, such as the correct birth date or the accurate number of units sold.
    * consistency :  refers to whether the data is uniform and doesn't contain any contradictory information.

    -- NOTE: Validity focuses on type, format, and range, while accuracy is focused on form and content.

* Google has established principles that guide Google AI applications, best practices to share our work with communities outside of Google and programs to operate rationalize our efforts.
    * The principles state that:
    
    * AI should be socially beneficial 
    * avoid creating or reinforcing unfair bias
    * be built and tested for safety
    * be accountable to people
    * incorporate privacy design principles
    * uphold high standards of scientific excellence

    And be made available for uses that accord with these principles.

* In addition to these principles, Google will not design or deploy AI in the following application areas:
    * Technologies that cause or are likely to cause overall harm.
    * Weapons or other technologies whose principal purpose or implementation is to cause or directly facilitate injury to people.
    * Technologies that gather or use information for surveillance, violating internationally accepted norms.
    * Technologies whose purpose contravenes widely accepted principles of international law and human rights.

#### Google's AI and ML Solutions

* Google Cloud offers four options for building machine learning models.
    1. BigQueryML: This is a tool for using SQL queries to create and execute machine learning models in BigQuery.
        * BigQuery ML democratizes the use of machine learning by empowering data analysts, but primary data warehouse users, to build and run models by using existing business intelligence tools and spreadsheets.
        * Using Python or Java to program an ML solution isn't necessary. Models are trained and access directly in BigQuery by using SQL, which is a language familiar to data analysts.
        * It also increases speed of production because moving and formatting large amounts of data for Python-based ML frameworks is not required for model training in BigQuery.
        * BigQuery ML also integrates with Vertex AI, Google Cloud's end to end AI and ML platform.
        * When BigQuery ML models are registered to the Vertex AI model registry, they can be deployed to endpoints for online prediction.
    2. Pre trained APIs: This option lets you use machine learning models that were built and trained by Google, so you don't have to build your own ML models if you don't have enough training data or sufficient machine learning expertise in house.
        * These are ideal in situations where an organization doesn't have specialized data scientists, but it does have business analysts and developers.
        * Google Cloud's pre trained API's can help developers build smart apps quickly by providing access to ML models for common tasks like analyzing images, videos, and text.
    3. AutoML: which is a no code solution, letting you build your own machine learning models on Vertex AI through a point and click interface.
        * Vertex AI brings together Google Cloud services for building ML under one unified user interface.
        * AutoML from Vertex AI lets you build and train machine learning models from end to end by using graphical user interfaces.
        * AutoML is a great option for businesses that want to produce a customized ML model, but are not willing to spend too much time coding and experimenting with thousands of models.
    * Vertex AI is also the essential platform for creating custom end to end machine learning models.
    4. Custom Training: through which you can code your very own machine learning environment,the training, and the deployment, which gives you flexibility and provides control over the ML pipeline.

* TensorFlow:is an end to end open source platform for machine learning.
    * TensorFlow has a flexible ecosystem of tools, libraries, and community resources that enable researchers to innovate in ML and developers to build and deploy ML powered applications.
    * TensorFlow takes advantage of the Tensor Processing Unit, or TPU, which is Google's custom developed application specific integrated circuit used to accelerate machine learning workloads.
    * Cloud TPUs have been integrated across Google products, and this state of the art hardware and supercomputing technology is available with Google Cloud products and services.

* Beyond the customizable options, Google Cloud has also created a set of full AI solutions aimed to solve specific business needs.
    * Contact Center AI provides models for speaking with customers and assisting human agents, increasing operational efficiency, and personalizing customer care to transform your contact center.
    * Document AI unlocks insights by extracting and classifying information from unstructured documents such as invoices, receipts, forms, letters, and reports.
    * Discovery AI for retail uses machine learning to select the optimal ordering of products on a retailer's e-commerce site when shoppers choose a category like winter jackets or kitchen ware.
    * Cloud Talent Solution uses AI with job search and talent acquisition capabilities, matches candidates to ideal jobs faster, and allows employers to attract and convert higher quality candidates.

* Google Cloud offers a range of AI and ML solutions and products, but there are several decisions and trade-offs to consider when selecting which to employ.
    * Speed:
        * Pre-trained API's require no model training, because that time-consuming task has already been carried out.
        * Custom training usually takes the longest time because it builds the ML model from the beginning, unlike autoML and Big query ML.
    * Differentiation:
        * To what level of customization does your AI solution require?
    * Expertise: 
        * Solutions vary depending on the use case and the level of expertise that the team possesses. how ever Google cloud has solutions to cater to needs of teams, whose expertise ranges from beginner to expert.
    * Effort:
        * This depends on several factors, including the complexity of the problem, the amount of data available, and the experience of the team.

* However, any AI undertaking will generally require much time, effort, and expertise to have a worthwhile impact on business operations.
