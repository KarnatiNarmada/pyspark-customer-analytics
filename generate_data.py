import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

print("Generating 3,000,000 customer records...")

np.random.seed(42)
n = 3_000_000

# Customer IDs
customer_ids = [f"CUST_{i:07d}" for i in range(1, n + 1)]

# Product categories
categories = ['Electronics', 'Clothing', 'Books', 'Home & Kitchen', 
              'Sports', 'Beauty', 'Toys', 'Automotive', 'Food', 'Health']

# Generate data
data = {
    'customer_id': customer_ids,
    'age': np.random.randint(18, 75, n),
    'gender': np.random.choice(['M', 'F'], n),
    'category': np.random.choice(categories, n),
    'purchase_amount': np.round(np.random.exponential(50, n) + 5, 2),
    'quantity': np.random.randint(1, 10, n),
    'rating': np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.10, 0.20, 0.35, 0.30]),
    'review_length': np.random.randint(10, 500, n),
    'days_since_last_purchase': np.random.randint(0, 365, n),
    'total_past_purchases': np.random.randint(0, 100, n),
    'returns': np.random.randint(0, 10, n),
    'discount_used': np.random.choice([0, 1], n, p=[0.6, 0.4]),
    'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Cash'], n),
    'device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n, p=[0.5, 0.35, 0.15]),
    'is_member': np.random.choice([0, 1], n, p=[0.4, 0.6]),
}

# Generate dates
start_date = datetime(2020, 1, 1)
data['purchase_date'] = [start_date + timedelta(days=random.randint(0, 1825)) for _ in range(n)]

# Create churn label (target variable)
# Customers who haven't purchased in 90+ days AND have low ratings = churned
churn_prob = (
    (np.array(data['days_since_last_purchase']) > 90).astype(float) * 0.3 +
    (np.array(data['rating']) <= 2).astype(float) * 0.3 +
    (np.array(data['returns']) > 5).astype(float) * 0.2 +
    (np.array(data['total_past_purchases']) < 5).astype(float) * 0.2
)
churn_prob = np.clip(churn_prob, 0, 1)
data['churned'] = np.random.binomial(1, churn_prob)

df = pd.DataFrame(data)
df.to_csv('data/customer_data.csv', index=False)
print(f"Generated {len(df):,} records")
print(f"Churn rate: {df['churned'].mean():.1%}")
print(f"File size: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"Saved to data/customer_data.csv")