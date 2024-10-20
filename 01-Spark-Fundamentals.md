
# Apache Spark Data Engineering Cheat Sheet

## 1. SparkSession and Context
- **SparkSession** is the entry point for creating DataFrames and working with Spark SQL.

```python
from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder \
    .appName("DataEngineering") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
```

- **SparkContext** (inside SparkSession)
```python
sc = spark.sparkContext
```

---

## 2. Reading and Writing Data (I/O Operations)

### a. Reading Data
- **CSV**
```python
df = spark.read.csv("path/to/csv", header=True, inferSchema=True)
```
- **Parquet**
```python
df = spark.read.parquet("path/to/parquet")
```
- **JSON**
```python
df = spark.read.json("path/to/json")
```
- **JDBC (SQL Databases)**
```python
df = spark.read.format("jdbc") \
    .option("url", "jdbc:postgresql://host:port/dbname") \
    .option("dbtable", "schema.table") \
    .option("user", "username") \
    .option("password", "password") \
    .load()
```

### b. Writing Data
- **CSV**
```python
df.write.csv("path/to/output_csv", mode="overwrite", header=True)
```
- **Parquet**
```python
df.write.parquet("path/to/output_parquet", mode="overwrite")
```
- **JSON**
```python
df.write.json("path/to/output_json", mode="overwrite")
```
- **JDBC**
```python
df.write.format("jdbc") \
    .option("url", "jdbc:postgresql://host:port/dbname") \
    .option("dbtable", "schema.output_table") \
    .option("user", "username") \
    .option("password", "password") \
    .save()
```

### c. Partitioning and Bucketing when Writing
- Partitioning:
```python
df.write.partitionBy("year", "month").parquet("path/to/output_parquet")
```
- Bucketing:
```python
df.write.bucketBy(4, "column_name").saveAsTable("bucketed_table")
```

---

## 3. DataFrame Operations (Transformations and Actions)

### a. Basic Transformations
- **Select Columns**
```python
df.select("name", "age").show()
```
- **Filter Rows**
```python
df.filter(df["age"] > 30).show()
df.where(df["age"] < 30).show()  # Alias of filter
```
- **With Column**
```python
from pyspark.sql.functions import col

# Create a new column
df.withColumn("age_plus_10", col("age") + 10).show()
```
- **Drop Column**
```python
df.drop("age").show()
```
- **Rename Column**
```python
df.withColumnRenamed("name", "full_name").show()
```

## Up Next: [02 - Advanced Operations and Optimizations](./02-Advanced-Operations-Optimizations.md)