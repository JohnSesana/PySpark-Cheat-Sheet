# Spark SQL
Spark SQL (spark.sql()): Allows you to write SQL-like queries to operate on DataFrames or tables registered as views.

## Register the DataFrame as a SQL view
```python
df.createOrReplaceTempView("customers")
```

## Using Spark SQL
```python
result_sql = spark.sql("""
  SELECT id, FIRST(name) AS name, address
  FROM customers
  WHERE address = 'california'
  GROUP BY id
""")
result_sql.show()

# Or you can write your view to save your result
df.write.parquet("path/to/output_parquet", mode="overwrite")
```

## Up Next: [04 - Spark MLlib](./04-Spark-MLlib.md)
