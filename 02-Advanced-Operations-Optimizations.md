
# Advanced Spark Operations, Configurations, and Optimizations

## 4. Advanced DataFrame Operations

### a. Handling Missing Data
- **Drop Null Values**
```python
df.na.drop().show()
```
- **Fill Null Values**
```python
df.na.fill(0).show()  # Fill with zero for numeric columns
df.na.fill("Unknown", subset=["name"]).show()  # Fill with "Unknown" for specific columns
```
- **Replace Values**
```python
df.replace("old_value", "new_value").show()
```

### b. Union and Intersection
- **Union**
```python
df1.union(df2).show()
```
- **Intersection**
```python
df1.intersect(df2).show()
```

### c. Handling Duplicates
```python
df.dropDuplicates(["name", "age"]).show()
```

---

## 5. Optimization Techniques

### a. Repartition and Coalesce
- **Repartition** (Increase number of partitions for parallelism)
```python
df.repartition(10).write.parquet("output/")
```
- **Coalesce** (Reduce the number of partitions)
```python
df.coalesce(2).write.parquet("output/")
```

### b. Broadcast Join (To optimize joins with small tables)
```python
from pyspark.sql.functions import broadcast

df1.join(broadcast(df2), "id").show()
```

### c. Caching and Persistence
- **Cache** DataFrame in memory
```python
df.cache()
```
- **Persist** DataFrame in memory and disk (if memory is insufficient)
```python
df.persist()
```
- **Unpersist** (free up memory when no longer needed)
```python
df.unpersist()
```

---

## 6. Spark Configuration Tips for Optimization

### a. Shuffle Partitions
- Adjust the number of shuffle partitions for better performance:
```python
spark.conf.set("spark.sql.shuffle.partitions", "200")  # Default is 200
```

### b. Executor and Driver Memory
- Allocate sufficient memory for your executors and driver based on your cluster and workload:
```bash
--executor-memory 4g
--driver-memory 4g
```

### c. Parallelism and Task Management
- Configure the level of parallelism based on data size and cluster capacity:
```bash
--conf spark.default.parallelism=1000
```

### d. Enable Kryo Serialization for faster serialization:
```bash
spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
```

---

## 7. Extra Optimization Tips

### a. Avoid using `count()` on large DataFrames
- Use `take()` to inspect data samples instead of `count()` for better performance:
```python
df.take(10)
```

### b. Filter Data Early
- Apply `filter()` transformations as early as possible in your data pipeline to reduce the volume of data processed:
```python
df.filter(df["age"] > 30)
```

### c. Use DataFrame API instead of RDD API
- The DataFrame API is optimized for query execution and will provide better performance compared to RDD API.

### d. Avoid Shuffles where possible
- Shuffles are expensive; try to avoid wide transformations like `groupByKey()`, `reduceByKey()` in favor of narrow transformations.

---

## 8. Monitoring and Debugging

- **Explain** a DataFrame’s execution plan:
```python
df.explain()
```

- **Job Progress Monitoring**
Check the Spark UI to see stages, tasks, and executors’ progress in real-time.

## Up Next: [03 - Spark SQL Syntax](./03-Spark-SQL-Synthax.md)
