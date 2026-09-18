# ============================================================
# SUPERMARKET SALES - EXPLORATORY DATA ANALYSIS
# ============================================================
# Author: Dharamveer Prakash P
# Project: Supermarket Sales EDA
# Tools: Python, Pandas, NumPy, Matplotlib, Seaborn
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visualization style
sns.set_theme(style="whitegrid")

# Display options
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.max_rows", 100)

print("=" * 70)
print("SUPERMARKET SALES - EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ============================================================
# 2. LOAD DATASET
# ============================================================

file_path = "data/supermarket_sales.csv"

if not os.path.exists(file_path):
    print("\nDataset not found!")
    print("Please place supermarket_sales.csv inside the data folder.")
    exit()

df = pd.read_csv(file_path)

print("\nDataset loaded successfully.")


# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC DATASET INFORMATION")
print("=" * 70)

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nDataset shape:")
print(df.shape)

print("\nNumber of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()


# ============================================================
# 4. STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

print("\nNumerical columns:")
print(df.describe())

print("\nCategorical columns:")
print(df.describe(include="object"))


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

missing_values = df.isnull().sum()

print("\nMissing values:")
print(missing_values)

missing_percentage = (
    df.isnull().sum() / len(df) * 100
)

print("\nMissing percentage:")
print(missing_percentage)


# ============================================================
# 6. CHECK DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE ANALYSIS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("\nNumber of duplicate rows:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")

else:
    print("No duplicate rows found.")


# ============================================================
# 7. CHECK UNIQUE VALUES
# ============================================================

print("\n" + "=" * 70)
print("UNIQUE VALUE ANALYSIS")
print("=" * 70)

for column in df.columns:
    print("\nColumn:", column)
    print("Unique values:", df[column].nunique())

    if df[column].nunique() < 20:
        print(df[column].unique())


# ============================================================
# 8. DATA CLEANING
# ============================================================

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)

# Remove unnecessary spaces from column names
df.columns = df.columns.str.strip()

# Remove leading/trailing spaces from object columns
object_columns = df.select_dtypes(include="object").columns

for column in object_columns:
    df[column] = df[column].astype(str).str.strip()

print("\nColumn names after cleaning:")
print(df.columns.tolist())


# ============================================================
# 9. DATE AND TIME PROCESSING
# ============================================================

print("\n" + "=" * 70)
print("DATE AND TIME PROCESSING")
print("=" * 70)

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.month_name()
    df["Day_Name"] = df["Date"].dt.day_name()

    print("\nDate columns created successfully.")

if "Time" in df.columns:
    df["Time"] = pd.to_datetime(
        df["Time"],
        format="%H:%M",
        errors="coerce"
    )

    df["Hour"] = df["Time"].dt.hour

    print("Time columns created successfully.")


# ============================================================
# 10. CREATE TIME PERIOD
# ============================================================

if "Hour" in df.columns:

    def get_time_period(hour):
        if hour < 12:
            return "Morning"
        elif hour < 17:
            return "Afternoon"
        elif hour < 21:
            return "Evening"
        else:
            return "Night"

    df["Time_Period"] = df["Hour"].apply(get_time_period)

    print("\nTime period created.")


# ============================================================
# 11. TOTAL SALES
# ============================================================

print("\n" + "=" * 70)
print("TOTAL SALES ANALYSIS")
print("=" * 70)

if "Total" in df.columns:

    total_sales = df["Total"].sum()

    average_sales = df["Total"].mean()

    minimum_sales = df["Total"].min()

    maximum_sales = df["Total"].max()

    median_sales = df["Total"].median()

    print("\nTotal Sales:", round(total_sales, 2))
    print("Average Transaction:", round(average_sales, 2))
    print("Minimum Transaction:", round(minimum_sales, 2))
    print("Maximum Transaction:", round(maximum_sales, 2))
    print("Median Transaction:", round(median_sales, 2))


# ============================================================
# 12. TOTAL QUANTITY SOLD
# ============================================================

print("\n" + "=" * 70)
print("QUANTITY ANALYSIS")
print("=" * 70)

if "Quantity" in df.columns:

    total_quantity = df["Quantity"].sum()

    average_quantity = df["Quantity"].mean()

    print("\nTotal Quantity Sold:", total_quantity)

    print(
        "Average Quantity per Transaction:",
        round(average_quantity, 2)
    )


# ============================================================
# 13. BRANCH ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("BRANCH ANALYSIS")
print("=" * 70)

if "Branch" in df.columns:

    branch_sales = (
        df.groupby("Branch")["Total"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nSales by Branch:")
    print(branch_sales)

    branch_average = (
        df.groupby("Branch")["Total"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nAverage transaction by Branch:")
    print(branch_average)

    branch_transactions = df["Branch"].value_counts()

    print("\nTransactions by Branch:")
    print(branch_transactions)


# ============================================================
# 14. CITY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CITY ANALYSIS")
print("=" * 70)

if "City" in df.columns:

    city_sales = (
        df.groupby("City")["Total"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nSales by City:")
    print(city_sales)

    city_quantity = (
        df.groupby("City")["Quantity"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nQuantity sold by City:")
    print(city_quantity)


# ============================================================
# 15. PRODUCT LINE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PRODUCT LINE ANALYSIS")
print("=" * 70)

if "Product line" in df.columns:

    product_sales = (
        df.groupby("Product line")["Total"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nSales by Product Line:")
    print(product_sales)

    product_quantity = (
        df.groupby("Product line")["Quantity"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nQuantity by Product Line:")
    print(product_quantity)

    product_average = (
        df.groupby("Product line")["Total"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nAverage transaction by Product Line:")
    print(product_average)


# ============================================================
# 16. GENDER ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("GENDER ANALYSIS")
print("=" * 70)

if "Gender" in df.columns:

    gender_count = df["Gender"].value_counts()

    print("\nCustomer count by Gender:")
    print(gender_count)

    gender_sales = (
        df.groupby("Gender")["Total"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nSales by Gender:")
    print(gender_sales)

    gender_average = (
        df.groupby("Gender")["Total"]
        .mean()
    )

    print("\nAverage transaction by Gender:")
    print(gender_average)


# ============================================================
# 17. CUSTOMER TYPE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CUSTOMER TYPE ANALYSIS")
print("=" * 70)

if "Customer type" in df.columns:

    customer_count = df["Customer type"].value_counts()

    print("\nCustomer Type Count:")
    print(customer_count)

    customer_sales = (
        df.groupby("Customer type")["Total"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nSales by Customer Type:")
    print(customer_sales)

    customer_average = (
        df.groupby("Customer type")["Total"]
        .mean()
    )

    print("\nAverage Transaction:")
    print(customer_average)


# ============================================================
# 18. PAYMENT METHOD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PAYMENT METHOD ANALYSIS")
print("=" * 70)

if "Payment" in df.columns:

    payment_count = df["Payment"].value_counts()

    print("\nPayment Method Count:")
    print(payment_count)

    payment_sales = (
        df.groupby("Payment")["Total"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nSales by Payment Method:")
    print(payment_sales)


# ============================================================
# 19. CUSTOMER RATING ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CUSTOMER RATING ANALYSIS")
print("=" * 70)

if "Rating" in df.columns:

    print("\nAverage Rating:")
    print(round(df["Rating"].mean(), 2))

    print("\nMedian Rating:")
    print(round(df["Rating"].median(), 2))

    print("\nHighest Rating:")
    print(df["Rating"].max())

    print("\nLowest Rating:")
    print(df["Rating"].min())


# ============================================================
# 20. GROSS INCOME ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("GROSS INCOME ANALYSIS")
print("=" * 70)

if "gross income" in df.columns:

    total_income = df["gross income"].sum()

    average_income = df["gross income"].mean()

    print("\nTotal Gross Income:")
    print(round(total_income, 2))

    print("\nAverage Gross Income:")
    print(round(average_income, 2))

    if "Product line" in df.columns:

        income_by_product = (
            df.groupby("Product line")["gross income"]
            .sum()
            .sort_values(ascending=False)
        )

        print("\nGross Income by Product:")
        print(income_by_product)


# ============================================================
# 21. TAX ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("TAX ANALYSIS")
print("=" * 70)

if "Tax 5%" in df.columns:

    print("\nTotal Tax:")
    print(round(df["Tax 5%"].sum(), 2))

    print("\nAverage Tax:")
    print(round(df["Tax 5%"].mean(), 2))

    print("\nMaximum Tax:")
    print(round(df["Tax 5%"].max(), 2))


# ============================================================
# 22. UNIT PRICE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("UNIT PRICE ANALYSIS")
print("=" * 70)

if "Unit price" in df.columns:

    print("\nAverage Unit Price:")
    print(round(df["Unit price"].mean(), 2))

    print("\nMinimum Unit Price:")
    print(round(df["Unit price"].min(), 2))

    print("\nMaximum Unit Price:")
    print(round(df["Unit price"].max(), 2))

    print("\nMedian Unit Price:")
    print(round(df["Unit price"].median(), 2))


# ============================================================
# 23. DAILY SALES ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DAILY SALES ANALYSIS")
print("=" * 70)

if "Date" in df.columns:

    daily_sales = (
        df.groupby("Date")["Total"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nTop 10 Sales Days:")
    print(daily_sales.head(10))

    print("\nLowest 10 Sales Days:")
    print(daily_sales.tail(10))


# ============================================================
# 24. DAY OF WEEK ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DAY OF WEEK ANALYSIS")
print("=" * 70)

if "Day_Name" in df.columns:

    day_sales = (
        df.groupby("Day_Name")["Total"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nSales by Day:")
    print(day_sales)

    day_transactions = df["Day_Name"].value_counts()

    print("\nTransactions by Day:")
    print(day_transactions)


# ============================================================
# 25. MONTH ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("MONTH ANALYSIS")
print("=" * 70)

if "Month_Name" in df.columns:

    month_sales = (
        df.groupby("Month_Name")["Total"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nSales by Month:")
    print(month_sales)


# ============================================================
# 26. HOUR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("HOURLY ANALYSIS")
print("=" * 70)

if "Hour" in df.columns:

    hourly_transactions = df["Hour"].value_counts().sort_index()

    print("\nTransactions by Hour:")
    print(hourly_transactions)

    hourly_sales = (
        df.groupby("Hour")["Total"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nSales by Hour:")
    print(hourly_sales)


# ============================================================
# 27. TIME PERIOD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("TIME PERIOD ANALYSIS")
print("=" * 70)

if "Time_Period" in df.columns:

    period_sales = (
        df.groupby("Time_Period")["Total"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nSales by Time Period:")
    print(period_sales)

    period_count = df["Time_Period"].value_counts()

    print("\nTransactions by Time Period:")
    print(period_count)


# ============================================================
# 28. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include=np.number
).columns

correlation = df[numeric_columns].corr()

print("\nCorrelation Matrix:")
print(correlation)


# ============================================================
# 29. HIGHEST VALUE TRANSACTIONS
# ============================================================

print("\n" + "=" * 70)
print("HIGHEST VALUE TRANSACTIONS")
print("=" * 70)

if "Total" in df.columns:

    highest_transactions = (
        df.sort_values(
            by="Total",
            ascending=False
        )
        .head(10)
    )

    print("\nTop 10 Transactions:")
    print(highest_transactions)


# ============================================================
# 30. LOWEST VALUE TRANSACTIONS
# ============================================================

print("\n" + "=" * 70)
print("LOWEST VALUE TRANSACTIONS")
print("=" * 70)

if "Total" in df.columns:

    lowest_transactions = (
        df.sort_values(
            by="Total",
            ascending=True
        )
        .head(10)
    )

    print("\nBottom 10 Transactions:")
    print(lowest_transactions)


# ============================================================
# 31. OUTLIER ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("OUTLIER ANALYSIS")
print("=" * 70)

if "Total" in df.columns:

    Q1 = df["Total"].quantile(0.25)

    Q3 = df["Total"].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR

    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df["Total"] < lower_limit)
        |
        (df["Total"] > upper_limit)
    ]

    print("\nQ1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower Limit:", lower_limit)
    print("Upper Limit:", upper_limit)
    print("Number of Outliers:", len(outliers))


# ============================================================
# 32. VISUALIZATION - SALES BY BRANCH
# ============================================================

if "Branch" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=df,
        x="Branch",
        y="Total",
        estimator="sum"
    )

    plt.title("Total Sales by Branch")
    plt.xlabel("Branch")
    plt.ylabel("Total Sales")

    plt.tight_layout()

    plt.savefig(
        "sales_by_branch.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 33. VISUALIZATION - PRODUCT SALES
# ============================================================

if "Product line" in df.columns:

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=df,
        y="Product line",
        x="Total",
        estimator="sum",
        order=df.groupby(
            "Product line"
        )["Total"].sum().sort_values(
            ascending=False
        ).index
    )

    plt.title("Total Sales by Product Line")
    plt.xlabel("Total Sales")
    plt.ylabel("Product Line")

    plt.tight_layout()

    plt.savefig(
        "sales_by_product_line.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 34. VISUALIZATION - GENDER
# ============================================================

if "Gender" in df.columns:

    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="Gender"
    )

    plt.title("Customer Distribution by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Number of Customers")

    plt.tight_layout()

    plt.savefig(
        "customer_gender.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 35. VISUALIZATION - PAYMENT
# ============================================================

if "Payment" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Payment"
    )

    plt.title("Payment Method Distribution")
    plt.xlabel("Payment Method")
    plt.ylabel("Number of Transactions")

    plt.tight_layout()

    plt.savefig(
        "payment_methods.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 36. VISUALIZATION - CUSTOMER TYPE
# ============================================================

if "Customer type" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Customer type"
    )

    plt.title("Customer Type Distribution")
    plt.xlabel("Customer Type")
    plt.ylabel("Number of Customers")

    plt.tight_layout()

    plt.savefig(
        "customer_type.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 37. VISUALIZATION - SALES DISTRIBUTION
# ============================================================

if "Total" in df.columns:

    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="Total",
        bins=30,
        kde=True
    )

    plt.title("Distribution of Total Sales")
    plt.xlabel("Total Sales")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        "sales_distribution.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 38. VISUALIZATION - UNIT PRICE DISTRIBUTION
# ============================================================

if "Unit price" in df.columns:

    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="Unit price",
        bins=30,
        kde=True
    )

    plt.title("Unit Price Distribution")
    plt.xlabel("Unit Price")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        "unit_price_distribution.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 39. VISUALIZATION - QUANTITY DISTRIBUTION
# ============================================================

if "Quantity" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Quantity"
    )

    plt.title("Quantity Distribution")
    plt.xlabel("Quantity")
    plt.ylabel("Number of Transactions")

    plt.tight_layout()

    plt.savefig(
        "quantity_distribution.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 40. VISUALIZATION - CUSTOMER RATING
# ============================================================

if "Rating" in df.columns:

    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="Rating",
        bins=20,
        kde=True
    )

    plt.title("Customer Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        "rating_distribution.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 41. VISUALIZATION - SALES BY DAY
# ============================================================

if "Day_Name" in df.columns:

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    day_sales_plot = (
        df.groupby("Day_Name")["Total"]
        .sum()
        .reindex(day_order)
    )

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x=day_sales_plot.index,
        y=day_sales_plot.values
    )

    plt.title("Sales by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Total Sales")

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(
        "sales_by_day.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 42. VISUALIZATION - SALES BY HOUR
# ============================================================

if "Hour" in df.columns:

    hourly_sales_plot = (
        df.groupby("Hour")["Total"]
        .sum()
    )

    plt.figure(figsize=(10, 5))

    sns.lineplot(
        x=hourly_sales_plot.index,
        y=hourly_sales_plot.values,
        marker="o"
    )

    plt.title("Sales by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Total Sales")

    plt.tight_layout()

    plt.savefig(
        "sales_by_hour.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 43. VISUALIZATION - TIME PERIOD
# ============================================================

if "Time_Period" in df.columns:

    period_order = [
        "Morning",
        "Afternoon",
        "Evening",
        "Night"
    ]

    period_plot = (
        df.groupby("Time_Period")["Total"]
        .sum()
        .reindex(period_order)
    )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        x=period_plot.index,
        y=period_plot.values
    )

    plt.title("Sales by Time Period")
    plt.xlabel("Time Period")
    plt.ylabel("Total Sales")

    plt.tight_layout()

    plt.savefig(
        "sales_by_time_period.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 44. BOXPLOT - TOTAL SALES
# ============================================================

if "Total" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        x=df["Total"]
    )

    plt.title("Boxplot of Total Sales")
    plt.xlabel("Total Sales")

    plt.tight_layout()

    plt.savefig(
        "sales_boxplot.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 45. BOXPLOT - SALES BY PRODUCT
# ============================================================

if "Product line" in df.columns:

    plt.figure(figsize=(12, 7))

    sns.boxplot(
        data=df,
        x="Total",
        y="Product line"
    )

    plt.title("Sales Distribution by Product Line")
    plt.xlabel("Total Sales")
    plt.ylabel("Product Line")

    plt.tight_layout()

    plt.savefig(
        "product_sales_boxplot.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 46. SCATTER PLOT - QUANTITY VS TOTAL
# ============================================================

if "Quantity" in df.columns and "Total" in df.columns:

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x="Quantity",
        y="Total"
    )

    plt.title("Quantity vs Total Sales")
    plt.xlabel("Quantity")
    plt.ylabel("Total Sales")

    plt.tight_layout()

    plt.savefig(
        "quantity_vs_sales.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 47. SCATTER PLOT - UNIT PRICE VS TOTAL
# ============================================================

if "Unit price" in df.columns and "Total" in df.columns:

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x="Unit price",
        y="Total"
    )

    plt.title("Unit Price vs Total Sales")
    plt.xlabel("Unit Price")
    plt.ylabel("Total Sales")

    plt.tight_layout()

    plt.savefig(
        "unit_price_vs_sales.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 48. GENDER VS PRODUCT LINE
# ============================================================

if "Gender" in df.columns and "Product line" in df.columns:

    gender_product = pd.crosstab(
        df["Product line"],
        df["Gender"]
    )

    print("\n" + "=" * 70)
    print("PRODUCT LINE BY GENDER")
    print("=" * 70)

    print(gender_product)

    plt.figure(figsize=(12, 6))

    gender_product.plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title("Product Line Purchases by Gender")
    plt.xlabel("Product Line")
    plt.ylabel("Number of Transactions")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        "product_gender_analysis.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 49. BRANCH VS PRODUCT LINE
# ============================================================

if "Branch" in df.columns and "Product line" in df.columns:

    branch_product = pd.crosstab(
        df["Branch"],
        df["Product line"]
    )

    print("\n" + "=" * 70)
    print("BRANCH VS PRODUCT LINE")
    print("=" * 70)

    print(branch_product)


# ============================================================
# 50. PAYMENT VS GENDER
# ============================================================

if "Payment" in df.columns and "Gender" in df.columns:

    payment_gender = pd.crosstab(
        df["Payment"],
        df["Gender"]
    )

    print("\n" + "=" * 70)
    print("PAYMENT METHOD VS GENDER")
    print("=" * 70)

    print(payment_gender)

    plt.figure(figsize=(9, 6))

    payment_gender.plot(
        kind="bar",
        figsize=(9, 6)
    )

    plt.title("Payment Method by Gender")
    plt.xlabel("Payment Method")
    plt.ylabel("Number of Transactions")

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        "payment_gender.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 51. CUSTOMER TYPE VS BRANCH
# ============================================================

if "Customer type" in df.columns and "Branch" in df.columns:

    customer_branch = pd.crosstab(
        df["Branch"],
        df["Customer type"]
    )

    print("\n" + "=" * 70)
    print("CUSTOMER TYPE VS BRANCH")
    print("=" * 70)

    print(customer_branch)


# ============================================================
# 52. HEATMAP - CORRELATION
# ============================================================

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "correlation_heatmap.png",
    dpi=300
)

plt.show()


# ============================================================
# 53. TOP PRODUCT BY SALES
# ============================================================

if "Product line" in df.columns:

    top_product = (
        df.groupby("Product line")["Total"]
        .sum()
        .idxmax()
    )

    print("\n" + "=" * 70)
    print("TOP PRODUCT")
    print("=" * 70)

    print("\nHighest-selling product line:")
    print(top_product)


# ============================================================
# 54. TOP BRANCH BY SALES
# ============================================================

if "Branch" in df.columns:

    top_branch = (
        df.groupby("Branch")["Total"]
        .sum()
        .idxmax()
    )

    print("\n" + "=" * 70)
    print("TOP BRANCH")
    print("=" * 70)

    print("\nHighest-selling branch:")
    print(top_branch)


# ============================================================
# 55. TOP PAYMENT METHOD
# ============================================================

if "Payment" in df.columns:

    top_payment = df["Payment"].value_counts().idxmax()

    print("\n" + "=" * 70)
    print("MOST COMMON PAYMENT METHOD")
    print("=" * 70)

    print("\nMost commonly used payment method:")
    print(top_payment)


# ============================================================
# 56. TOP SALES DAY
# ============================================================

if "Day_Name" in df.columns:

    top_day = (
        df.groupby("Day_Name")["Total"]
        .sum()
        .idxmax()
    )

    print("\n" + "=" * 70)
    print("TOP SALES DAY")
    print("=" * 70)

    print("\nHighest-sales day:")
    print(top_day)


# ============================================================
# 57. TOP SALES HOUR
# ============================================================

if "Hour" in df.columns:

    top_hour = (
        df.groupby("Hour")["Total"]
        .sum()
        .idxmax()
    )

    print("\n" + "=" * 70)
    print("TOP SALES HOUR")
    print("=" * 70)

    print("\nHighest-sales hour:")
    print(top_hour)


# ============================================================
# 58. DATASET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

print("\nNumber of Records:", len(df))

print("Number of Features:", len(df.columns))

print(
    "Total Sales:",
    round(df["Total"].sum(), 2)
    if "Total" in df.columns
    else "N/A"
)

print(
    "Average Sales:",
    round(df["Total"].mean(), 2)
    if "Total" in df.columns
    else "N/A"
)

print(
    "Total Quantity:",
    df["Quantity"].sum()
    if "Quantity" in df.columns
    else "N/A"
)


# ============================================================
# 59. SAVE CLEANED DATA
# ============================================================

output_file = "data/cleaned_supermarket_sales.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nCleaned dataset saved to:")
print(output_file)


# ============================================================
# 60. FINAL PROJECT INSIGHTS
# ============================================================

print("\n" + "=" * 70)
print("KEY BUSINESS INSIGHTS")
print("=" * 70)

if "Branch" in df.columns:

    branch_sales_result = (
        df.groupby("Branch")["Total"]
        .sum()
        .sort_values(ascending=False)
    )

    print(
        "\n1. Branch with highest sales:",
        branch_sales_result.index[0]
    )


if "Product line" in df.columns:

    product_sales_result = (
        df.groupby("Product line")["Total"]
        .sum()
        .sort_values(ascending=False)
    )

    print(
        "2. Product line with highest sales:",
        product_sales_result.index[0]
    )


if "Payment" in df.columns:

    payment_result = (
        df["Payment"]
        .value_counts()
    )

    print(
        "3. Most commonly used payment method:",
        payment_result.index[0]
    )


if "Gender" in df.columns:

    gender_result = (
        df.groupby("Gender")["Total"]
        .sum()
        .sort_values(ascending=False)
    )

    print(
        "4. Gender with higher total spending:",
        gender_result.index[0]
    )


if "Customer type" in df.columns:

    customer_result = (
        df.groupby("Customer type")["Total"]
        .sum()
        .sort_values(ascending=False)
    )

    print(
        "5. Customer type with higher total sales:",
        customer_result.index[0]
    )


if "Day_Name" in df.columns:

    day_result = (
        df.groupby("Day_Name")["Total"]
        .sum()
        .sort_values(ascending=False)
    )

    print(
        "6. Day with highest sales:",
        day_result.index[0]
    )


if "Hour" in df.columns:

    hour_result = (
        df.groupby("Hour")["Total"]
        .sum()
        .sort_values(ascending=False)
    )

    print(
        "7. Hour with highest sales:",
        hour_result.index[0]
    )


# ============================================================
# END OF PROJECT
# ============================================================

print("\n" + "=" * 70)
print("EDA PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated analysis includes:")
print("- Data inspection")
print("- Data cleaning")
print("- Missing value analysis")
print("- Duplicate analysis")
print("- Statistical analysis")
print("- Branch analysis")
print("- City analysis")
print("- Product analysis")
print("- Gender analysis")
print("- Customer analysis")
print("- Payment analysis")
print("- Time analysis")
print("- Correlation analysis")
print("- Outlier analysis")
print("- 20+ visualizations")
print("- Business insights")

print("\nThank you!")
print("=" * 70)
