Fake News Detection with Advanced NLP
This repository contains the code and resources for a project focused on detecting fake news using advanced Natural Language Processing (NLP) techniques. [cite_start]The project aims to provide users with a tool to discern reliable and unreliable news articles by analyzing their content through a robust backend API and a high-performing classification model.
Project Overview
The proliferation of fake news online is a significant challenge. [cite_start]This project builds upon an earlier phase by refining the approach with automated URL analysis and sophisticated NLP. [cite_start]The core task is to classify articles as either "reliable" or "unreliable/fake". [cite_start]By empowering critical evaluation and combating misinformation, this project contributes to a more informed society.
Features
 * [cite_start]Backend API (Flask): A robust Flask API receives news article URLs and serves predictions.
 * [cite_start]Article Content Fetching: Utilizes the requests library to fetch HTML content from submitted URLs.
 * [cite_start]Data Extraction: Employs Beautiful Soup to parse HTML and extract relevant article text and headlines.
 * [cite_start]Advanced NLP Feature Extraction: Processes text and headlines using NLTK for basic tasks and spaCy/Transformers for advanced feature extraction (e.g., part-of-speech tagging, named entity recognition, sentiment analysis, semantic embeddings).
 * [cite_start]High-Performing Classification Model: Integrates and trains models (e.g., scikit-learn, Transformers) to categorize articles as "reliable" or "unreliable/fake".
 * [cite_start]Model Interpretability: Aims to maintain a degree of interpretability to understand linguistic features driving predictions (e.g., feature importance analysis).
 * [cite_start]Real-time Prediction: Generates predictions by feeding extracted NLP features into the loaded model.
 * [cite_start]Result Display: Sends prediction results to the frontend, highlighting key linguistic indicators for "unreliable/fake" articles.
Project Workflow
 * [cite_start]User Input: User enters the URL of a news article into the web application's input field.
 * [cite_start]URL Submission: User submits the URL, sending it to the backend.
 * [cite_start]Backend Receives URL: The Flask backend API receives the submitted URL.
 * Article Content Fetching: The backend fetches HTML content from the URL using requests. [cite_start]Error messages are generated if fetching fails.
 * [cite_start]Data Extraction: Beautiful Soup parses the HTML to extract article text and headlines.
 * [cite_start]NLP Feature Extraction: Extracted text is processed using NLTK, spaCy, and Transformers for various NLP tasks.
 * [cite_start]Model Loading: The trained fake news detection model (machine learning or deep learning) is loaded into memory.
 * [cite_start]Prediction Generation: NLP features are fed into the model to classify the article as "reliable" or "unreliable/fake".
 * [cite_start]Result Display: The prediction is sent to the frontend, with explanations for "unreliable/fake" classifications.
 * [cite_start]User Interaction: User views the prediction and explanation to understand potential misinformation.
Data Description
 * [cite_start]Primary Data Source: Content of news articles obtained by scraping user-submitted URLs.
 * [cite_start]Type of Data: Primarily text data (article text and headlines)[cite_start], with potential structured data for metadata.
 * [cite_start]Dataset: The model training utilizes a static dataset, while real-time data processed from user-submitted URLs is dynamic.
 * [cite_start]Target Variable: The model predicts the reliability of a news article, with the target being a binary categorical variable ("reliable" or "unreliable/fake").
Data Preprocessing
 * Handle Missing Values: Missing values in the training dataset will be handled by removing articles with high missing data or imputing numerical features. [cite_start]Critical missing content during scraping will lead to discarding the scrape.
 * Remove Duplicate Records: Exact duplicate articles in the training dataset will be removed. [cite_start]Duplicate URL submissions will be handled by caching or prevention mechanisms.
 * Data Type Conversion and Consistency: Data types will be converted as needed (e.g., dates to datetime, text to numerical vectors). [cite_start]Consistency will be enforced through standardization of text encoding, case, whitespace, and categorical representations.
 * [cite_start]Encode Categorical Variables: The target variable will be label-encoded, and nominal categorical features (e.g., article source) will be one-hot encoded using scikit-learn and pandas.
 * [cite_start]Normalize/Standardize Features: Numerical features will be normalized or standardized to ensure consistent scaling and improve model performance.
Model Building
[cite_start]At least two machine learning models will be selected and implemented for the fake news detection task. [cite_start]Suitable models include Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, Support Vector Machines (SVM), Gradient Boosting algorithms (XGBoost, LightGBM), and Transformer-based models (e.g., fine-tuned BERT or ROBERTA).
[cite_start]The data will be split into training and testing sets (commonly 80% training, 20% testing), with stratification applied if the target variable is imbalanced. [cite_start]Models will be trained on the training set and evaluated using appropriate metrics like accuracy, precision, and recall on the testing set.
Tools and Technologies Used
 * [cite_start]Programming Language: Python
 * [cite_start]Web Framework (Backend): Flask
 * [cite_start]Frontend Technologies: HTML, CSS, JavaScript
 * [cite_start]IDE/Notebook: VS Code, Jupyter Notebook, Google Colab
 * [cite_start]Data Handling Libraries: pandas, NumPy
 * [cite_start]Web Scraping Libraries: requests, Beautiful Soup
 * [cite_start]NLP Libraries: NLTK, spaCy, Transformers (Hugging Face), TextBlob
 * [cite_start]Machine Learning Library: scikit-learn
 * [cite_start]Visualization Libraries: matplotlib, seaborn, wordcloud
 * [cite_start]Optional Tools (for Deployment): Docker, Cloud hosting platforms (AWS, GCP, Heroku), WSGI servers (Gunicorn/uWSGI)
Team Members and Contributions
 * [cite_start]Mohammed Sakhee.B: Model development
 * [cite_start]Mohammed Sharuk.I: Feature Engineering
 * [cite_start]Mubarak Basha.S: EDA
 * [cite_start]Naseerudin: Data Cleaning
 * [cite_start]Rishi Kumar Baskar: Documentation and Reporting
GitHub Repository
[cite_start]https://github.com/rishikumar287/Exposing-the-truth-with-advanced-fake-news-detection-powered-by-natural-language-.git
