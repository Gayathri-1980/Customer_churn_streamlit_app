import pandas as pd
import numpy as np
import os

# Output dataset path
OUTPUT_PATH = "data/customer_data.csv"
os.makedirs("data", exist_ok=True)

# Function to generate a synthetic churn dataset
def generate_churn_dataset(num_samples=500):
    np.random.seed(42)

    age = np.random.randint(18, 70, num_samples)
    income = np.random.randint(20000, 120000, num_samples)
    region = np.random.choice(["North", "South", "East", "West"], num_samples)
    tenure = np.random.randint(1, 10, num_samples)
    support_calls = np.random.randint(0, 10, num_samples)

    churn_prob = (
        (age < 30) * 0.4 +
        (income < 40000) * 0.3 +
        (tenure < 3) * 0.2 +
        (support_calls > 5) * 0.4
    )

    churn = np.random.rand(num_samples) < churn_prob

    df = pd.DataFrame({
        "Age": age,
        "Income": income,
        "Region": region,
        "Tenure": tenure,
        "SupportCalls": support_calls,
        "Churn": churn.astype(int)
    })

    return df

if __name__ == "__main__":
    df = generate_churn_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset created successfully at: {OUTPUT_PATH}")
