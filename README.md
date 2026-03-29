# Customer Churn Data Analysis with ML Pipeline

## Table of Contents
* [Project Overview](#project-overview)
* [Data Sources](#data-sources)
* [Tech Stack](#tech-stack)
* [Project Workflow](#project-workflow)
  * [Stage 1: Data Preparation (Task 1)](#stage-1-data-preparation-task-1)
  * [Stage 2: Analysis & Modeling (Task 2)](#stage-2-analysis--modeling-task-2)
    * [Part A: Exploratory Data Analysis (Q1–Q2)](#part-a-exploratory-data-analysis-q1q2)
    * [Part B: Machine Learning Pipeline (Q3–Q5) \[My main contribution\]](#part-b-machine-learning-pipeline-q3q5-my-main-contribution)
* [Results and Key Findings](#results-and-key-findings)
* [My Contribution Summary](#my-contribution-summary)

## Project Overview
This project focuses on discovering the link between customer characteristics and churn, then to build a machine learning model that can identify customers likely to leave in Databricks with PySpark.

My primary responsibility was the design and implementation of the **machine learning pipeline**. The overall system includes data ingestion, cleaning, transformation, and predictive modeling.

## Data Sources
Three CSV files about the **characteristics and service usage of a fictional telecoms company's clients** were provided as part of the coursework. The datasets contain 

## Tech Stack 
- Databricks Free Edition
- Apache Spark (PySpark)

## Project Workflow
### Stage 1: Data Preparation (Task 1)

This stage focuses on transforming raw datasets into a clean, consistent, and analysis-ready format.

Key steps include:
- Handling inconsistent and non-standard values
  - Converting `"NAN"` strings to null values
  - Standardizing categorical encodings (`"0"/"1"` → `"No"/"Yes"`)
    ```python
    # Replace string 'NAN' with SQL NULL
    df_c2 = df_c2_raw.na.replace('NAN', None)
    
    # Fix Categorical Columns (0 -> No, 1 -> Yes)
    categorical_fix_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                            'TechSupport', 'StreamingTV', 'StreamingMovies', 
                            'PaperlessBilling', 'Churn']
    
    for column in categorical_fix_cols:
        df_c2 = df_c2.withColumn(column, 
            when(col(column) == "0", "No")
            .when(col(column) == "1", "Yes")
            .otherwise(col(column)))
    ```
- Merging datasets
- Removing duplicate records based on `customerID`
  ```python
  df_unique = combined_df.dropDuplicates(['customerID'])
  ```
- Filtering invalid rows (e.g., zero or null values)
  ```python
  df_final = df_cast.na.drop(subset=["TotalCharges", "tenure", "MonthlyCharges"]) \
                  .filter(col("TotalCharges") > 0)
  ```

### Stage 2: Analysis & Modeling (Task 2)

This stage is divided into two parts: Exploratory Data Analysis (EDA) and Machine Learning Pipeline Development.

#### Part A: Exploratory Data Analysis (Q1–Q2)

Initial analysis was performed to understand the dataset and identify patterns related to churn:

- Distribution of churn vs non-churn customers
- Identification of class imbalance (~26.5% churn rate)
- Descriptive statistics for key variables (e.g., tenure, charges)
  <img width="1344" height="718" alt="image" src="https://github.com/user-attachments/assets/ea28f139-c1e1-498e-b2a3-79f42dd3f341" />

Feature-level analysis using bar charts and pie charts:
- Gender vs churn
- Senior citizen vs churn
- Tenure correlation with churn
  <img width="1583" height="1189" alt="image" src="https://github.com/user-attachments/assets/e9945bba-021f-4015-bc12-b501c7b6bdcf" />

#### Part B: Machine Learning Pipeline (Q3–Q5) [My main contribution]

This stage focuses on building a reusable and scalable PySpark ML pipeline for churn prediction using the cleaned data.

Key components:

**1. Data Preparation for ML**

- Train-test split
  ```python
  # Split the data - 70% for training, 30% for testing
  train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)
  ```

**2. Feature Engineering**

- Encoding categorical variables:
  - `StringIndexer`
  - `OneHotEncoder`
  ```python
    # 1. StringIndexer for original categorical columns
    indexers = [
        StringIndexer(
            inputCol=column, 
            outputCol=column + "_index", 
            handleInvalid="keep"
        ) 
        for column in categorical_cols
    ]
    
    # 2. OneHotEncoder for indexed categorical columns
    encoders = [
        OneHotEncoder(
            inputCol=column + "_index", 
            outputCol=column + "_encoded"
        ) 
        for column in categorical_cols
    ]
    ```

- Handling numerical features:
  - `Imputer` (for missing values)
  - `QuantileDiscretizer` (for binning)
  ```python
    # 3. Imputer - Handle missing values in numerical columns
    imputer = Imputer(
        inputCols=numeric_cols,
        outputCols=[col + "_imputed" for col in numeric_cols],
        strategy="median"  # Using median as it's not significantly affected by outliers
    )
    
    # 4. QuantileDiscretizer - Bin continuous variables 
    discretizers = []
    
    for col_name in numeric_cols:
        input_col = col_name + "_imputed"
        discretizer = QuantileDiscretizer(
            inputCol=input_col,
            outputCol=col_name + "_binned",
            numBuckets=5,  # Creating 5 quantile-based bins
            handleInvalid="keep"
        )
        discretizers.append(discretizer)
    
    # 5. StringIndexer for binned numerical features
    binned_indexers = [
        StringIndexer(
            inputCol=col_name + "_binned",
            outputCol=col_name + "_binned_index",
            handleInvalid="keep"
        )
        for col_name in numeric_cols
    ]
    
    # 6. OneHotEncoder for binned numerical features
    binned_encoders = [
        OneHotEncoder(
            inputCol=col_name + "_binned_index",
            outputCol=col_name + "_binned_encoded"
        )
        for col_name in numeric_cols
    ]
    ```
    
- Combining features using `VectorAssembler`
  ```python
  # 7. VectorAssembler  - Combine all features (ENHANCED)
  # include both original categorical features and binned numerical features
  assembler_inputs = (
      [col + "_encoded" for col in categorical_cols] +  # Original categoricals
      [col + "_binned_encoded" for col in numeric_cols]  # Binned numericals
  )
  
  assembler = VectorAssembler(
      inputCols=assembler_inputs,
      outputCol="features",
      handleInvalid="keep"
  )
  ```
  
 
**3. Model Training**

- Training classification model(s) for churn prediction using Spark ML
  ```python
  from pyspark.ml.classification import LogisticRegression

  # 1. Initialize Logistic Regression model
  lr = LogisticRegression(
      labelCol="label",        # from label_indexer in Q4
      featuresCol="features",
      maxIter=20,
      regParam=0.0,
      elasticNetParam=0.0
  )
  
  # 2. Combine stages from Q4 and LR model
  complete_pipeline_stages = enhanced_pipeline_stages + [lr]
  ```

**4. Pipeline Construction**

- Integrating all preprocessing steps and model training into a single `Pipeline` object
- Ensuring consistent and repeatable transformations and model training
  ```python
    # 3. Create the complete pipeline
    complete_pipeline = Pipeline(stages=complete_pipeline_stages)
  ```

**5. Model Evaluation**

- Evaluating performance using ROC/AUC
  ```python
  # BINARY CLASSIFICATION METRICS (for bin_eval_df)
  binary_evaluator = BinaryClassificationEvaluator(
      labelCol="label_binary", 
      rawPredictionCol="score"
  )
  
  # Calculate Area Under ROC
  auc_roc = binary_evaluator.evaluate(bin_eval_df, {binary_evaluator.metricName: "areaUnderROC"})
  
  # Calculate Area Under PR (useful for imbalanced data)
  auc_pr = binary_evaluator.evaluate(bin_eval_df, {binary_evaluator.metricName: "areaUnderPR"})
  ```

## Results and Key Findings

**Churn Distribution**
- 26.54% of customers churned, while 73.46% remained
- The dataset is imbalanced, which impacts model performance

**Key Factors Influencing Churn**
- Customers with shorter tenure are more likely to churn
- Higher monthly charges are associated with higher churn rates
- Certain customer segments (e.g., senior citizens) show different churn patterns

**Model Performance**
- Model evaluated using ROC/AUC and PR
- The model demonstrates the ability to identify potential churn customers
- Results:
  - Area Under ROC Curve (AUC-ROC): 0.8438
  - Area Under PR Curve (AUC-PR): 0.6716
   <img width="706" height="552" alt="image" src="https://github.com/user-attachments/assets/4375aa42-dee3-422f-82c3-f4f99c80a554" />


## My Contribution Summary

I was responsible for Task 2 (Q3–Q5), focusing on the machine learning pipeline and modeling components.

My contributions include:

- Designing and implementing the PySpark ML preprocessing pipeline
- Performing feature engineering for both categorical and numerical data
- Constructing reusable transformation workflows using Pipeline
- Training and evaluating churn prediction model
