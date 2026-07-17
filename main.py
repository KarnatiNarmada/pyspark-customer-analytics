from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, mean, sum as spark_sum, when, 
    year, month, dayofweek, round as spark_round,
    log, stddev, min as spark_min, max as spark_max,
    corr, lit
)
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
)
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time

# ============================================
#  PYSPARK CUSTOMER ANALYTICS PIPELINE
#  Processing 3,000,000+ Records
# ============================================

def create_spark_session():
    """Initialize Spark session"""
    spark = SparkSession.builder \
        .appName("CustomerAnalytics") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.hadoop.fs.defaultFS", "file:///") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_data(spark, filepath):
    """Step 1: Load data into PySpark DataFrame"""
    print("\n" + "=" * 60)
    print("📊 STEP 1: LOADING DATA")
    print("=" * 60)
    
    start = time.time()
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    
    print(f"   Records: {df.count():,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Load time: {time.time() - start:.1f}s")
    print(f"\n   Schema:")
    df.printSchema()
    
    return df


def exploratory_data_analysis(df):
    """Step 2: EDA — Explore patterns and distributions"""
    print("\n" + "=" * 60)
    print("🔍 STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Basic stats
    print("\n--- Basic Statistics ---")
    df.select("age", "purchase_amount", "quantity", "rating", 
              "review_length", "total_past_purchases").describe().show()
    
    # Churn distribution
    print("--- Churn Distribution ---")
    churn_dist = df.groupBy("churned").count()
    churn_dist.show()
    total = df.count()
    churned = df.filter(col("churned") == 1).count()
    print(f"   Churn rate: {churned/total:.1%}")
    
    # Churn by category
    print("--- Churn Rate by Category ---")
    df.groupBy("category") \
        .agg(
            count("*").alias("total_customers"),
            spark_round(mean("churned"), 3).alias("churn_rate"),
            spark_round(mean("rating"), 2).alias("avg_rating"),
            spark_round(mean("purchase_amount"), 2).alias("avg_purchase")
        ) \
        .orderBy(col("churn_rate").desc()) \
        .show(10)
    
    # Churn by device
    print("--- Churn Rate by Device ---")
    df.groupBy("device") \
        .agg(
            count("*").alias("total"),
            spark_round(mean("churned"), 3).alias("churn_rate")
        ).show()
    
    # Churn by membership
    print("--- Churn Rate: Members vs Non-Members ---")
    df.groupBy("is_member") \
        .agg(
            count("*").alias("total"),
            spark_round(mean("churned"), 3).alias("churn_rate"),
            spark_round(mean("purchase_amount"), 2).alias("avg_purchase")
        ).show()
    
    # Age distribution by churn
    print("--- Average Age: Churned vs Retained ---")
    df.groupBy("churned") \
        .agg(
            spark_round(mean("age"), 1).alias("avg_age"),
            spark_round(mean("days_since_last_purchase"), 1).alias("avg_days_inactive"),
            spark_round(mean("returns"), 2).alias("avg_returns")
        ).show()
    
    return df


def feature_engineering(df):
    """Step 3: Create new features for modeling"""
    print("\n" + "=" * 60)
    print("🛠️ STEP 3: FEATURE ENGINEERING")
    print("=" * 60)
    
    df = df \
        .withColumn("spend_per_item", spark_round(col("purchase_amount") / col("quantity"), 2)) \
        .withColumn("return_rate", spark_round(col("returns") / (col("total_past_purchases") + 1), 4)) \
        .withColumn("is_high_spender", when(col("purchase_amount") > 100, 1).otherwise(0)) \
        .withColumn("is_inactive", when(col("days_since_last_purchase") > 90, 1).otherwise(0)) \
        .withColumn("is_low_rating", when(col("rating") <= 2, 1).otherwise(0)) \
        .withColumn("is_frequent_buyer", when(col("total_past_purchases") > 20, 1).otherwise(0)) \
        .withColumn("is_high_returner", when(col("returns") > 5, 1).otherwise(0)) \
        .withColumn("engagement_score", 
                    spark_round(
                        col("total_past_purchases") * 0.3 +
                        col("rating") * 0.3 +
                        (365 - col("days_since_last_purchase")) * 0.01 +
                        col("review_length") * 0.01 -
                        col("returns") * 0.5,
                    2)) \
        .withColumn("purchase_year", year(col("purchase_date"))) \
        .withColumn("purchase_month", month(col("purchase_date"))) \
        .withColumn("purchase_day", dayofweek(col("purchase_date"))) \
        .withColumn("log_purchase", spark_round(log(col("purchase_amount") + 1), 4))
    
    new_features = [
        "spend_per_item", "return_rate", "is_high_spender", "is_inactive",
        "is_low_rating", "is_frequent_buyer", "is_high_returner",
        "engagement_score", "purchase_year", "purchase_month", 
        "purchase_day", "log_purchase"
    ]
    
    print(f"   Created {len(new_features)} new features:")
    for f in new_features:
        print(f"   ✅ {f}")
    
    print(f"\n   Total features now: {len(df.columns)}")
    
    return df


def prepare_ml_data(df):
    """Step 4: Prepare data for ML modeling"""
    print("\n" + "=" * 60)
    print("⚙️ STEP 4: PREPARING ML DATA")
    print("=" * 60)
    
    # Index string columns
    category_indexer = StringIndexer(inputCol="category", outputCol="category_index")
    gender_indexer = StringIndexer(inputCol="gender", outputCol="gender_index")
    device_indexer = StringIndexer(inputCol="device", outputCol="device_index")
    payment_indexer = StringIndexer(inputCol="payment_method", outputCol="payment_index")
    
    # Select numeric features
    numeric_features = [
        "age", "purchase_amount", "quantity", "rating", "review_length",
        "days_since_last_purchase", "total_past_purchases", "returns",
        "discount_used", "is_member", "spend_per_item", "return_rate",
        "is_high_spender", "is_inactive", "is_low_rating", "is_frequent_buyer",
        "is_high_returner", "engagement_score", "purchase_month",
        "purchase_day", "log_purchase",
        "category_index", "gender_index", "device_index", "payment_index"
    ]
    
    # Assemble features into vector
    assembler = VectorAssembler(inputCols=numeric_features, outputCol="features_raw")
    
    # Scale features
    scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                           withStd=True, withMean=True)
    
    # Build pipeline
    pipeline = Pipeline(stages=[
        category_indexer, gender_indexer, device_indexer, payment_indexer,
        assembler, scaler
    ])
    
    # Fit and transform
    model = pipeline.fit(df)
    df_prepared = model.transform(df)
    
    # Split data
    train, test = df_prepared.randomSplit([0.8, 0.2], seed=42)
    
    print(f"   Feature columns: {len(numeric_features)}")
    print(f"   Training records: {train.count():,}")
    print(f"   Testing records: {test.count():,}")
    
    return train, test


def train_and_evaluate(train, test):
    """Step 5: Train models and evaluate"""
    print("\n" + "=" * 60)
    print("🤖 STEP 5: MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    results = {}
    
    # Evaluators
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="churned", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    acc_evaluator = MulticlassClassificationEvaluator(
        labelCol="churned", predictionCol="prediction", metricName="accuracy"
    )
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="churned", predictionCol="prediction", metricName="f1"
    )
    
    # Model 1: Logistic Regression
    print("\n--- Model 1: Logistic Regression ---")
    start = time.time()
    lr = LogisticRegression(featuresCol="features", labelCol="churned", maxIter=20)
    lr_model = lr.fit(train)
    lr_pred = lr_model.transform(test)
    
    lr_auc = auc_evaluator.evaluate(lr_pred)
    lr_acc = acc_evaluator.evaluate(lr_pred)
    lr_f1 = f1_evaluator.evaluate(lr_pred)
    lr_time = time.time() - start
    
    print(f"   ROC-AUC:  {lr_auc:.4f}")
    print(f"   Accuracy: {lr_acc:.4f}")
    print(f"   F1 Score: {lr_f1:.4f}")
    print(f"   Time:     {lr_time:.1f}s")
    results['Logistic Regression'] = {'AUC': lr_auc, 'Accuracy': lr_acc, 'F1': lr_f1, 'Time': lr_time}
    
    # Model 2: Random Forest
    print("\n--- Model 2: Random Forest ---")
    start = time.time()
    rf = RandomForestClassifier(
        featuresCol="features", labelCol="churned",
        numTrees=100, maxDepth=8, seed=42
    )
    rf_model = rf.fit(train)
    rf_pred = rf_model.transform(test)
    
    rf_auc = auc_evaluator.evaluate(rf_pred)
    rf_acc = acc_evaluator.evaluate(rf_pred)
    rf_f1 = f1_evaluator.evaluate(rf_pred)
    rf_time = time.time() - start
    
    print(f"   ROC-AUC:  {rf_auc:.4f}")
    print(f"   Accuracy: {rf_acc:.4f}")
    print(f"   F1 Score: {rf_f1:.4f}")
    print(f"   Time:     {rf_time:.1f}s")
    results['Random Forest'] = {'AUC': rf_auc, 'Accuracy': rf_acc, 'F1': rf_f1, 'Time': rf_time}
    
    # Feature importance
    print("\n--- Top 10 Feature Importances (Random Forest) ---")
    feature_names = [
        "age", "purchase_amount", "quantity", "rating", "review_length",
        "days_since_last_purchase", "total_past_purchases", "returns",
        "discount_used", "is_member", "spend_per_item", "return_rate",
        "is_high_spender", "is_inactive", "is_low_rating", "is_frequent_buyer",
        "is_high_returner", "engagement_score", "purchase_month",
        "purchase_day", "log_purchase",
        "category", "gender", "device", "payment_method"
    ]
    importances = rf_model.featureImportances.toArray()
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    for name, imp in feat_imp[:10]:
        bar = "█" * int(imp * 100)
        print(f"   {name:30s} {imp:.4f} {bar}")
    
    # Model 3: Gradient Boosted Trees
    print("\n--- Model 3: Gradient Boosted Trees ---")
    start = time.time()
    gbt = GBTClassifier(
        featuresCol="features", labelCol="churned",
        maxIter=50, maxDepth=6, seed=42
    )
    gbt_model = gbt.fit(train)
    gbt_pred = gbt_model.transform(test)
    
    gbt_auc = auc_evaluator.evaluate(gbt_pred)
    gbt_acc = acc_evaluator.evaluate(gbt_pred)
    gbt_f1 = f1_evaluator.evaluate(gbt_pred)
    gbt_time = time.time() - start
    
    print(f"   ROC-AUC:  {gbt_auc:.4f}")
    print(f"   Accuracy: {gbt_acc:.4f}")
    print(f"   F1 Score: {gbt_f1:.4f}")
    print(f"   Time:     {gbt_time:.1f}s")
    results['Gradient Boosted Trees'] = {'AUC': gbt_auc, 'Accuracy': gbt_acc, 'F1': gbt_f1, 'Time': gbt_time}
    
    return results, feat_imp


def create_visualizations(results, feat_imp):
    """Step 6: Create and save visualizations"""
    print("\n" + "=" * 60)
    print("📈 STEP 6: CREATING VISUALIZATIONS")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("PySpark Customer Churn Analytics — 3M+ Records", fontsize=14, fontweight='bold')
    
    # Chart 1: Model comparison
    models = list(results.keys())
    aucs = [results[m]['AUC'] for m in models]
    colors = ['#2E5090', '#E74C3C', '#27AE60']
    axes[0].barh(models, aucs, color=colors)
    axes[0].set_xlabel("ROC-AUC Score")
    axes[0].set_title("Model Comparison (ROC-AUC)")
    for i, v in enumerate(aucs):
        axes[0].text(v + 0.005, i, f"{v:.3f}", va='center', fontweight='bold')
    axes[0].set_xlim(0, 1.1)
    
    # Chart 2: Feature importance
    top_features = feat_imp[:8]
    names = [f[0] for f in top_features][::-1]
    imps = [f[1] for f in top_features][::-1]
    axes[1].barh(names, imps, color='#2E5090')
    axes[1].set_xlabel("Importance")
    axes[1].set_title("Top 8 Feature Importances")
    
    # Chart 3: All metrics comparison
    metrics = ['AUC', 'Accuracy', 'F1']
    x = range(len(models))
    width = 0.25
    for i, metric in enumerate(metrics):
        vals = [results[m][metric] for m in models]
        axes[2].bar([xi + i * width for xi in x], vals, width, label=metric, alpha=0.85)
    axes[2].set_xticks([xi + width for xi in x])
    axes[2].set_xticklabels(models, fontsize=9)
    axes[2].set_ylabel("Score")
    axes[2].set_title("All Metrics Comparison")
    axes[2].legend()
    axes[2].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig("data/model_results.png", dpi=150, bbox_inches='tight')
    print("   ✅ Saved: data/model_results.png")
    plt.close()


def print_summary(results):
    """Final summary"""
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE — SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Model':<25} {'ROC-AUC':>10} {'Accuracy':>10} {'F1':>10} {'Time':>10}")
    print("-" * 65)
    
    best_model = None
    best_auc = 0
    
    for model, metrics in results.items():
        print(f"{model:<25} {metrics['AUC']:>10.4f} {metrics['Accuracy']:>10.4f} {metrics['F1']:>10.4f} {metrics['Time']:>9.1f}s")
        if metrics['AUC'] > best_auc:
            best_auc = metrics['AUC']
            best_model = model
    
    print(f"\n🏆 Best Model: {best_model} (ROC-AUC: {best_auc:.4f})")
    print(f"\n📊 Dataset: 3,000,000+ customer records")
    print(f"🛠️ Features: 25+ engineered features")
    print(f"🤖 Models: 3 compared (Logistic Regression, Random Forest, GBT)")
    print(f"📈 Visualization: data/model_results.png")


# ============================================
#  RUN THE PIPELINE
# ============================================

if __name__ == "__main__":
    print("\n🚀 PySpark Customer Analytics Pipeline")
    print("   Processing 3,000,000+ records...\n")
    
    total_start = time.time()
    
    # Initialize Spark
    spark = create_spark_session()
    
    # Run pipeline
    df = load_data(spark, "file:///Users/yogan/Documents/Narmada/Learning/GitHub/pyspark-customer-analytics/data/customer_data.csv")
    df = exploratory_data_analysis(df)
    df = feature_engineering(df)
    train, test = prepare_ml_data(df)
    results, feat_imp = train_and_evaluate(train, test)
    create_visualizations(results, feat_imp)
    print_summary(results)
    
    total_time = time.time() - total_start
    print(f"\n⏱️ Total pipeline time: {total_time:.1f}s")
    
    # Stop Spark
    spark.stop()
    print("\n✅ Spark session stopped. Done!")