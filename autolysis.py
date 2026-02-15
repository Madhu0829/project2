# /// script
# requires-python = ">=3.11"
# dependencies = [
# "pandas",
# "matplotlib",
# "seaborn",
# "openai"
# ]
# ///

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from openai import OpenAI

# -----------------------------
# 1. Load dataset
# -----------------------------

file_path = sys.argv[1]

try:
    df = pd.read_csv(file_path, encoding="latin1")
except:
    df = pd.read_csv(file_path)

print("Dataset loaded successfully")

# -----------------------------
# 2. Generic Analysis
# -----------------------------

numeric_df = df.select_dtypes(include='number')

if len(numeric_df.columns) > 1:
    corr = numeric_df.corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("correlation.png")
    plt.close()

# -----------------------------
# 3. Prepare Summary for LLM
# -----------------------------

summary = {
    "columns": list(df.columns),
    "shape": df.shape,
    "missing_values": df.isnull().sum().to_dict(),
}

if len(numeric_df.columns) > 0:
    summary["describe"] = numeric_df.describe().to_dict()

# -----------------------------
# 4. Setup LLM (AI Proxy / OpenRouter)
# -----------------------------

client = OpenAI(
    api_key=os.environ["AIPROXY_TOKEN"],
    base_url="https://openrouter.ai/api/v1"
)

# -----------------------------
# 5. Generate Story
# -----------------------------

prompt = f"""
Write a story-style analysis of this dataset.

Include:

1. dataset overview
2. analysis performed
3. key insights
4. implications

Data summary:
{summary}
"""

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[{"role":"user","content":prompt}]
)

story = response.choices[0].message.content

# -----------------------------
# 6. Save README.md
# -----------------------------

with open("README.md","w",encoding="utf-8") as f:
    f.write(story)

print("README.md created successfully")
