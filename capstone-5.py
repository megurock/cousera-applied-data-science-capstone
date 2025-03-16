import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./dataset_part_2.csv')
print(df.head())

sns.catplot(y="PayloadMass", x="FlightNumber", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Pay load Mass (kg)",fontsize=20)
plt.show()



# --------------------------------------------------------------------------------
# TASK 1: Visualize the relationship between Flight Number and Launch Site
# --------------------------------------------------------------------------------
sns.catplot(y="LaunchSite", x="FlightNumber", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Launch Site",fontsize=20)
plt.show()



# --------------------------------------------------------------------------------
# TASK 2: Visualize the relationship between Payload Mass and Launch Site
# --------------------------------------------------------------------------------
sns.catplot(y="LaunchSite", x="PayloadMass", hue="Class", data=df, aspect = 5)
plt.xlabel("Payload Mass (kg)",fontsize=20)
plt.ylabel("Launch Site",fontsize=20)
plt.show()



# --------------------------------------------------------------------------------
# TASK 3: Visualize the relationship between success rate of each orbit type
# --------------------------------------------------------------------------------
# 軌道ごとの成功率を計算
orbit_success_rate = df.groupby("Orbit")["Class"].mean().reset_index()

# 可視化
plt.figure(figsize=(12,6))
sns.barplot(x="Orbit", y="Class", hue="Orbit", data=orbit_success_rate, palette="viridis", legend=False)


# グラフのラベル
plt.xlabel("Orbit Type", fontsize=14)
plt.ylabel("Success Rate", fontsize=14)
plt.title("Success Rate by Orbit Type", fontsize=16)
plt.xticks(rotation=45)
plt.show()



# --------------------------------------------------------------------------------
# TASK 4: Visualize the relationship between FlightNumber and Orbit type
# --------------------------------------------------------------------------------
# 散布図を作成
plt.figure(figsize=(12,6))
sns.scatterplot(x="FlightNumber", y="Orbit", hue="Class", data=df, palette="viridis", s=100)

# 軸ラベルとタイトル
plt.xlabel("Flight Number", fontsize=14)
plt.ylabel("Orbit Type", fontsize=14)
plt.title("Relationship between Flight Number and Orbit Type", fontsize=16)
plt.legend(title="Class", loc="best")
plt.show()



# --------------------------------------------------------------------------------
# TASK 5: Visualize the relationship between Payload Mass and Orbit type
# --------------------------------------------------------------------------------
# 散布図の作成
plt.figure(figsize=(12,6))
sns.scatterplot(x="PayloadMass", y="Orbit", hue="Class", data=df, palette="viridis", s=100)

# 軸ラベルとタイトル
plt.xlabel("Payload Mass (kg)", fontsize=14)
plt.ylabel("Orbit Type", fontsize=14)
plt.title("Relationship between Payload Mass and Orbit Type", fontsize=16)
plt.legend(title="Class", loc="best")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()



# --------------------------------------------------------------------------------
# TASK 6: Visualize the launch success yearly trend
# --------------------------------------------------------------------------------
# A function to Extract years from the date
year=[]

def Extract_year():
  for i in df["Date"]:
    year.append(i.split("-")[0])
  return year

# 年を新しい列として追加
df["Year"] = Extract_year()

# 年ごとの成功率（Classの平均）を計算
yearly_success_rate = df.groupby("Year")["Class"].mean().reset_index()

# 年を整数型に変換（ソートのため）
yearly_success_rate["Year"] = yearly_success_rate["Year"].astype(int)

# 年ごとの成功率の推移を可視化
plt.figure(figsize=(12,6))
sns.lineplot(x="Year", y="Class", data=yearly_success_rate, marker="o", color="b")

# 軸ラベルとタイトル
plt.xlabel("Year", fontsize=14)
plt.ylabel("Average Success Rate", fontsize=14)
plt.title("Launch Success Yearly Trend", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()



# --------------------------------------------------------------------------------
# TASK 7: Create dummy variables to categorical columns
# --------------------------------------------------------------------------------
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
features.head()

# OneHotEncoder を適用するカテゴリカル列を指定
categorical_columns = ['Orbit', 'LaunchSite', 'LandingPad', 'Serial']

# get_dummies を使ってOneHotエンコーディングを適用
features_one_hot = pd.get_dummies(features, columns=categorical_columns)

# 結果を表示
print(features_one_hot.head())



# --------------------------------------------------------------------------------
# TASK 8: Cast all numeric columns to float64
# --------------------------------------------------------------------------------
# 全ての数値列を float64 にキャスト
features_one_hot = features_one_hot.astype('float64')

# CSVファイルとしてエクスポート
features_one_hot.to_csv('dataset_part_3.csv', index=False)

# 結果の確認
print(features_one_hot.head())