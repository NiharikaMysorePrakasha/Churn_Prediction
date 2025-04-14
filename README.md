# Churn_Prediction
# Customer Churn Prediction Using PySpark

## 📄 Project Abstract

This project aims to predict customer churn using big data analytics and machine learning techniques in PySpark. Churn prediction identifies customers who are likely to stop using a company's service, enabling organizations to proactively implement retention strategies. Leveraging the distributed computing capabilities of Apache Spark, this solution ensures scalable processing and efficient analysis of large telecom datasets.

## 📌 Project Overview

Customer churn, the rate at which customers discontinue using a service, significantly impacts profitability. This project uses telecom data to train classification models that distinguish churners from non-churners. The project pipeline includes:

- Data ingestion and preprocessing
- Feature engineering and summary statistics
- Correlation and exploratory analysis
- Model training, selection, and evaluation using:
  - Naïve Bayes
  - Random Forest
  - Gradient Boosted Trees
- Performance comparison via accuracy, AUC, and confusion matrix

The Gradient Boosted Tree model achieved the best results with an accuracy of **92.05%**, demonstrating the viability of PySpark MLlib for churn prediction.

## 🔧 Tech Stack

- **Language:** Python
- **Big Data Framework:** Apache Spark (PySpark)
- **Machine Learning Library:** Spark MLlib
- **IDE/Environment:** Databricks, Jupyter Notebook
- **Visualization:** Seaborn, Matplotlib
- **Version Control:** GitHub

## 🛠️ Tools Used

- **Apache Spark MLlib** for machine learning algorithms and cross-validation  
- **Seaborn & Matplotlib** for visual analysis and correlation plotting  
- **StringIndexer, VectorAssembler, Pipeline** for Spark ML preprocessing  
- **MulticlassMetrics, BinaryClassificationEvaluator** for model evaluation  

## 🧩 Pipelines

Spark ML Pipelines were employed to streamline the machine learning workflow. A pipeline chains multiple stages—such as data transformation, feature indexing, model training, and evaluation—into a single cohesive flow. This ensured reproducibility, modularity, and scalability of the predictive modeling process.

Key components:
- **StringIndexer** for converting categorical variables into numeric labels
- **VectorAssembler** for aggregating features into a single vector column
- **Estimator (e.g., RandomForestClassifier)** as the training algorithm
- **Pipeline API** to encapsulate and automate all these steps

Pipelines allowed iterative tuning and model selection through cross-validation and parameter grid search, enhancing both efficiency and performance.

## 📊 Model Performance

| Model                  | Accuracy (%) | AUC (Area Under ROC) |
|------------------------|--------------|------------------------|
| Naïve Bayes            | 53.67%       | 0.61                   |
| Random Forest          | 88.91%       | 0.88                   |
| Gradient Boosted Trees | **92.05%**   | **0.86**               |

The Gradient Boosted Tree classifier outperformed other models, showing high accuracy and strong generalization capabilities.

## 📈 Dataset

Telecom datasets were sourced from Kaggle and contain customer attributes including usage patterns, service plans, and customer support interactions.

- [Telecom Churn Dataset 1](https://www.kaggle.com/code/bandiatindra/telecom-churn-prediction)  
- [Telecom Churn Dataset 2](https://www.kaggle.com/datasets/muhammedsar/churn-datacsv?select=churn_data.csv)

Each row in the dataset represents a customer, with columns reflecting account length, call minutes, service plans, and churn status.

## 🚀 Future Scope

- **Real-time churn monitoring:** Deploying models in streaming environments for live prediction.  
- **ETL Pipelines:** Automating data movement from SQL servers to data lakes using Apache Spark.  
- **Model Ensemble:** Incorporating ensemble methods such as SVMs, Decision Trees, and boosting techniques.  
- **External Data Integration:** Enhancing prediction with demographic and behavioral data.  
- **Business Integration:** Linking predictions to CRM systems to trigger retention campaigns.  

## 👩‍💻 Contributors

- **Gahana Nagaraja** – Data gathering, preprocessing, and presentation design  
- **Namratha Nagathihalli Anantha** – Model development and implementation  
- **Niharika Mysore Prakasha** – Model evaluation and report documentation  

Guided by: **Prof. Denghui Zhang**  
*Stevens Institute of Technology | Course: BIA 678-A Big Data Technologies*

---
