
# Spark MLlib Cheat Sheet (Beginner to Advanced)

## Table of Contents
1. [Introduction to MLlib](#introduction)
2. [Data Preparation](#data-preparation)
3. [Basic Machine Learning Algorithms](#basic-ml-algorithms)
4. [Advanced Machine Learning Techniques](#advanced-ml)
5. [Model Evaluation](#model-evaluation)
6. [Pipeline and Cross Validation](#pipeline-and-cross-validation)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Feature Engineering](#feature-engineering)
9. [Advanced Topics](#advanced-topics)

---

## <a name="introduction"></a> 1. Introduction to MLlib
Apache Spark's MLlib is a scalable machine learning library that provides various algorithms and utilities for machine learning. MLlib works seamlessly with Spark DataFrames, allowing easy integration with big data pipelines.

### Basic Steps in MLlib:
- Load and Prepare Data
- Feature Engineering
- Model Training
- Model Evaluation
- Model Tuning and Deployment

---

## <a name="data-preparation"></a> 2. Data Preparation

### Loading Data
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("MLlib").getOrCreate()

# Load CSV as DataFrame
data = spark.read.csv("data.csv", header=True, inferSchema=True)
```

### Handling Missing Values
```python
# Drop rows with missing values
data = data.na.drop()

# Fill missing values
data = data.na.fill({'column_name': 0})
```

### StringIndexer: Convert Categorical Variables to Numeric
```python
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="category", outputCol="category_index")
data = indexer.fit(data).transform(data)
```

### VectorAssembler: Combine Feature Columns
```python
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=["col1", "col2", "col3"], outputCol="features")
data = assembler.transform(data)
```

---

## <a name="basic-ml-algorithms"></a> 3. Basic Machine Learning Algorithms

### Logistic Regression
```python
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol='features', labelCol='label')
model = lr.fit(training_data)
predictions = model.transform(test_data)
```

### Decision Tree
```python
from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(featuresCol='features', labelCol='label')
model = dt.fit(training_data)
predictions = model.transform(test_data)
```

### Random Forest
```python
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol='features', labelCol='label')
model = rf.fit(training_data)
predictions = model.transform(test_data)
```

### Linear Regression
```python
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol='features', labelCol='label')
model = lr.fit(training_data)
predictions = model.transform(test_data)
```

---

## <a name="advanced-ml"></a> 4. Advanced Machine Learning Techniques

### Gradient-Boosted Trees
```python
from pyspark.ml.classification import GBTClassifier
gbt = GBTClassifier(featuresCol='features', labelCol='label', maxIter=10)
model = gbt.fit(training_data)
predictions = model.transform(test_data)
```

### Support Vector Machines
```python
from pyspark.ml.classification import LinearSVC
svm = LinearSVC(featuresCol='features', labelCol='label')
model = svm.fit(training_data)
predictions = model.transform(test_data)
```

### K-Means Clustering
```python
from pyspark.ml.clustering import KMeans
kmeans = KMeans(featuresCol='features', k=3)
model = kmeans.fit(data)
predictions = model.transform(data)
```

---

## <a name="model-evaluation"></a> 5. Model Evaluation

### Classification Metrics
```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
```

### Regression Metrics
```python
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
```

---

## <a name="pipeline-and-cross-validation"></a> 6. Pipeline and Cross Validation

### Building a Pipeline
```python
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[indexer, assembler, lr])
model = pipeline.fit(training_data)
predictions = model.transform(test_data)
```

### Cross Validation
```python
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).build()
crossval = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
cv_model = crossval.fit(training_data)
```

---

## <a name="hyperparameter-tuning"></a> 7. Hyperparameter Tuning

### Grid Search
```python
paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [10, 20, 30]).build()
crossval = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
cv_model = crossval.fit(training_data)
```

### Random Search
```python
from pyspark.ml.tuning import TrainValidationSplit

paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).build()
tvs = TrainValidationSplit(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, trainRatio=0.8)
tvs_model = tvs.fit(training_data)
```

---

## <a name="feature-engineering"></a> 8. Feature Engineering

### StandardScaler: Normalize Features
```python
from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(data)
scaled_data = scaler_model.transform(data)
```

### OneHotEncoder: Encoding Categorical Variables
```python
from pyspark.ml.feature import OneHotEncoder
encoder = OneHotEncoder(inputCols=["category_index"], outputCols=["category_vector"])
data = encoder.fit(data).transform(data)
```

---

## <a name="advanced-topics"></a> 9. Advanced Topics

### Collaborative Filtering (ALS - Matrix Factorization)
```python
from pyspark.ml.recommendation import ALS

als = ALS(userCol="userId", itemCol="itemId", ratingCol="rating")
model = als.fit(training_data)
predictions = model.transform(test_data)
```

### Saving and Loading Models
```python
# Save the model
model.save("path/to/model")

# Load the model
from pyspark.ml.classification import LogisticRegressionModel
loaded_model = LogisticRegressionModel.load("path/to/model")
```

### Custom Transformers and Estimators
You can extend `Transformer` and `Estimator` classes to build custom machine learning components.