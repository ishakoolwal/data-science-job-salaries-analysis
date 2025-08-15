# data_science_salaries.py

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


CSV_PATH = "C:/Users/MSI/Downloads/data_science.csv"  
OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("ggplot") 


def save_and_show(fig_name, dpi=200):
    out_path = OUT_DIR / fig_name
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    print("Saved:", out_path)
    plt.show()


if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

print("Initial shape:", df.shape)
print("\nColumns in CSV:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())

df.columns = [c.strip() for c in df.columns]


numeric_cols = ['salary', 'salary_in_usd', 'remote_ratio']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')




if 'job_title' in df.columns and 'salary_in_usd' in df.columns:
    avg_salary = df.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10,6))
    avg_salary.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Top 10 Average Salaries by Job Title")
    plt.ylabel("Average Salary (USD)")
    save_and_show("avg_salary_by_job_title.png")


if 'experience_level' in df.columns and 'salary_in_usd' in df.columns:
    plt.figure(figsize=(8,6))
    df.boxplot(column='salary_in_usd', by='experience_level', grid=False)
    plt.title("Salary Distribution by Experience Level")
    plt.suptitle("")  # Remove automatic title
    plt.xlabel("Experience Level")
    plt.ylabel("Salary (USD)")
    save_and_show("salary_by_experience.png")


if 'remote_ratio' in df.columns and 'salary_in_usd' in df.columns:
    plt.figure(figsize=(8,6))
    plt.scatter(df['remote_ratio'], df['salary_in_usd'], alpha=0.6, color='orange', edgecolors='black')
    plt.title("Remote Ratio vs Salary")
    plt.xlabel("Remote Ratio (%)")
    plt.ylabel("Salary (USD)")
    save_and_show("remote_ratio_vs_salary.png")


if 'employee_residence' in df.columns and 'salary_in_usd' in df.columns:
    avg_country_salary = df.groupby('employee_residence')['salary_in_usd'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10,6))
    avg_country_salary.plot(kind='bar', color='green', edgecolor='black')
    plt.title("Top 10 Countries by Average Salary")
    plt.ylabel("Average Salary (USD)")
    save_and_show("avg_salary_by_country.png")


if 'company_size' in df.columns and 'salary_in_usd' in df.columns:
    avg_size_salary = df.groupby('company_size')['salary_in_usd'].mean().sort_values(ascending=False)
    plt.figure(figsize=(8,5))
    avg_size_salary.plot(kind='bar', color='purple', edgecolor='black')
    plt.title("Average Salary by Company Size")
    plt.ylabel("Average Salary (USD)")
    save_and_show("avg_salary_by_company_size.png")


num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) > 1:
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    cax = ax.matshow(corr, cmap='coolwarm')
    plt.colorbar(cax)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=90)
    ax.set_yticklabels(num_cols)
    plt.title("Correlation Matrix", pad=20)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha='center', va='center', color='black', fontsize=8)
    save_and_show("correlation_matrix.png")


