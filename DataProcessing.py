import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F

# Start timer for performance comparison
start_time = time.time()

# Step 1: Initialize Spark session
spark = SparkSession.builder \
    .appName("Cardiovascular Disease Analysis") \
    .config("spark.driver.bindAddress", "192.168.13.140") \
    .config("spark.driver.port", "7077") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Step 2: Load the dataset
df = spark.read.csv("/mnt/shared_folder/CVD_cleaned.csv", header=True, inferSchema=True)

# Step 3: Preprocessing - Convert categorical features to ordinal values
# Convert 'Checkup' column
df = df.withColumn('Checkup', 
    when(col('Checkup') == 'Within the past year', 1)
    .when(col('Checkup') == 'Within the past 2 years', 2)
    .when(col('Checkup') == 'Within the past 5 years', 3)
    .when(col('Checkup') == '5 or more years ago', 4)
    .otherwise(5))

# Convert 'General_Health' column
df = df.withColumn('General_Health', 
    when(col('General_Health') == 'Excellent', 1)
    .when(col('General_Health') == 'Very Good', 2)
    .when(col('General_Health') == 'Good', 3)
    .when(col('General_Health') == 'Fair', 4)
    .otherwise(5))

# Convert 'Age_Category' column to ordinal values
df = df.withColumn('Age_Category', 
    when(col('Age_Category') == '18-24', 1)
    .when(col('Age_Category') == '25-29', 2)
    .when(col('Age_Category') == '30-34', 3)
    .when(col('Age_Category') == '35-39', 4)
    .when(col('Age_Category') == '40-44', 5)
    .when(col('Age_Category') == '45-49', 6)
    .when(col('Age_Category') == '50-54', 7)
    .when(col('Age_Category') == '55-59', 8)
    .when(col('Age_Category') == '60-64', 9)
    .when(col('Age_Category') == '65-69', 10)
    .when(col('Age_Category') == '70-74', 11)
    .when(col('Age_Category') == '75-79', 12)
    .when(col('Age_Category') == '80+', 13)
    .otherwise(None))

# Step 4: Convert boolean features to integers
boolean_cols = [
    'Exercise', 'Heart_Disease', 'Skin_Cancer', 
    'Other_Cancer', 'Depression', 'Diabetes', 
    'Arthritis', 'Smoking_History', 'Sex'
]

# Convert 'Yes'/'No' to 1/0, and 'Male'/'Female' to 1/0
for col_name in boolean_cols:
    if col_name == 'Sex':
        df = df.withColumn(col_name, when(col(col_name) == 'Male', 1).otherwise(0))
    else:
        df = df.withColumn(col_name, when(col(col_name) == 'Yes', 1).otherwise(0))

# Step 5: Scale numerical features
scale_cols = [
    'Alcohol_Consumption', 'Fruit_Consumption', 
    'Green_Vegetables_Consumption', 'FriedPotato_Consumption',
    'Height_(cm)', 'Weight_(kg)', 'BMI'
]

# Assemble the features into a single vector column for scaling
assembler = VectorAssembler(inputCols=scale_cols, outputCol="features_to_scale")
df_scaled = assembler.transform(df)

# Create MinMaxScaler
scaler = MinMaxScaler(inputCol="features_to_scale", outputCol="scaled_features")

# Fit and transform the data
scaler_model = scaler.fit(df_scaled)
df = scaler_model.transform(df_scaled)
df = df.repartition(4)

# Step 6: Correlation Analysis
features_to_correlate = ['BMI', 'Alcohol_Consumption', 'Smoking_History', 'Age_Category']
correlation_matrix = {}
for feature in features_to_correlate:
    correlation = df.stat.corr(feature, "Heart_Disease")
    correlation_matrix[feature] = correlation
    print(f"Correlation between {feature} and Heart_Disease: {correlation}")

# Step 7: Logistic Regression
assembler = VectorAssembler(inputCols=features_to_correlate, outputCol="features")
df = assembler.transform(df)

# Train/test split
train, test = df.randomSplit([0.8, 0.2], seed=1234)

# Initialize logistic regression model
lr = LogisticRegression(labelCol="Heart_Disease", featuresCol="features", maxIter=10)

# Train the model
lr_model = lr.fit(train)

# Make predictions
predictions = lr_model.transform(test)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="Heart_Disease")
accuracy = evaluator.evaluate(predictions)
print(f"Logistic Regression Model Accuracy: {accuracy}")

# End timer and calculate execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time} seconds")

# Stop the Spark session
spark.stop()

