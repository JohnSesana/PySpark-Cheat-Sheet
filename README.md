<div align="center">
  <img src="https://github.com/user-attachments/assets/c03ee96d-846e-42a6-bf2d-879f917d340c" alt="Spark Logo" width="350" />
</div>

# PySpark Cheat Sheets Repository

### Table of Contents

- [01 - Spark Fundamentals](./01-Spark-Fundamentals.md)
- [02 - Advanced Operations and Optimizations](./02-Advanced-Operations-Optimizations.md)
- [03 - Spark SQL Syntax](./03-Spark-SQL-Synthax.md)
- [04 - Spark MLlib](./04-Spark-MLlib.md)

Each of these cheat sheets offers detailed breakdowns and examples to help you master different aspects of PySpark, from basic syntax to advanced machine learning techniques.

---

### How to Use This Repository:

1. **Spark Syntax Fundamentals**: Start here if you're new to Spark or want to brush up on the core DataFrame API, transformations, and actions.
2. **Advanced Operations and Optimizations**: Learn how to optimize your Spark jobs for performance, repartitioning, and minimizing shuffles.
3. **Spark SQL Syntax**: Understand how to query DataFrames using SQL and make use of Spark SQL's powerful optimizer, Catalyst.
4. **Spark MLlib**: Dive into Spark's machine learning library, including regression, classification, and model evaluation techniques.


## Introduction to Spark

Apache Spark is a unified analytics engine designed for large-scale data processing. It provides high-level APIs in Java, Scala, Python (PySpark), and R, and an optimized engine that supports general execution graphs. Spark is known for its in-memory processing, which makes it much faster than traditional big data processing frameworks like Hadoop.

Spark's primary advantages:
- **In-Memory Processing**: Spark processes data in memory, significantly speeding up operations compared to disk-based systems.
- **Distributed Computing**: Spark can process large datasets across a cluster of machines.
- **Wide Range of Applications**: It supports various applications, including batch processing, real-time stream processing, machine learning, and graph processing.

### Key Components of Apache Spark:

![image](https://github.com/user-attachments/assets/389c0656-8a9e-437e-9c4e-b1c5df43ee6a)

1. **Spark Core**: The foundation for all other Spark components, providing in-memory computing and distributed execution.
2. **Spark SQL**: Allows querying of structured data via SQL or DataFrame API.
3. **Spark Streaming**: Enables real-time processing of data streams.
4. **MLlib (Machine Learning Library)**: Provides scalable machine learning algorithms like classification, regression, clustering, and collaborative filtering.
5. **GraphX**: A library for graph processing and analysis.


## Up Next: [01 - Spark Fundamentals](./01-Spark-Fundamentals.md)
