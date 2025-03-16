import csv, sqlite3
import prettytable
import pandas as pd

prettytable.DEFAULT = 'DEFAULT'
con = sqlite3.connect("my_data1.db")
cur = con.cursor()

# SPACEXTBL テーブルが存在すれば削除
cur.execute("DROP TABLE IF EXISTS SPACEXTBL;")

# CSV ファイルを読み込み、テーブルに保存
df = pd.read_csv("./Spacex.csv")
df.to_sql("SPACEXTBL", con, if_exists='replace', index=False, method="multi")



# -----------------------------------------------------------------------
# Task1: Display the names of the unique launch sites in the space mission
# -----------------------------------------------------------------------
cur.execute("SELECT DISTINCT Launch_Site FROM SPACEXTBL;")
launch_sites = cur.fetchall()

print("Unique Launch Sites:")
for site in launch_sites:
  print(site[0])
print('-----------------------------------')



# -----------------------------------------------------------------------
# Task2: Display 5 records where launch sites begin with the string 'CCA'
# -----------------------------------------------------------------------
cur.execute("SELECT * FROM SPACEXTBL WHERE Launch_Site LIKE 'CCA%' LIMIT 5;")
records = cur.fetchall()

print("Records where launch sites begin with 'CCA':")
for record in records:
  print(record)
print('-----------------------------------')



# -----------------------------------------------------------------------
# Task3: Display the total payload mass carried by boosters launched by NASA (CRS)
# -----------------------------------------------------------------------
cur.execute("""
  SELECT SUM(PAYLOAD_MASS__KG_)
  FROM SPACEXTBL
  WHERE Customer = 'NASA (CRS)';
""")
total_payload_mass = cur.fetchone()[0]

print(f"Total payload mass carried by boosters launched by NASA (CRS): {total_payload_mass} kg")
print('-----------------------------------')



# -----------------------------------------------------------------------
# Task4: Display average payload mass carried by booster version F9 v1.1
# -----------------------------------------------------------------------
cur.execute("""
  SELECT AVG(PAYLOAD_MASS__KG_)
  FROM SPACEXTBL
  WHERE Booster_Version = 'F9 v1.1';
""")
average_payload_mass = cur.fetchone()[0]

print(f"Average payload mass carried by booster version F9 v1.1: {average_payload_mass} kg")
print('-----------------------------------')



# -----------------------------------------------------------------------
# Task5: List the date when the first successful landing outcome
# in ground pad was archived.
# -----------------------------------------------------------------------
cur.execute("""
  SELECT MIN(Date)
  FROM SPACEXTBL
  WHERE Landing_Outcome = 'Success (ground pad)';
""")
first_successful_landing_date = cur.fetchone()[0]

print(f"The date when the first successful landing outcome in ground pad was archived: {first_successful_landing_date}")
print('-----------------------------------')



# -----------------------------------------------------------------------
# Task6: List the names of the boosters which have success in drone ship
# and have payload mass greater than 4000 but less than 6000
# -----------------------------------------------------------------------
cur.execute("""
  SELECT DISTINCT Booster_Version
  FROM SPACEXTBL
  WHERE Landing_Outcome = 'Success (drone ship)'
  AND PAYLOAD_MASS__KG_ > 4000
  AND PAYLOAD_MASS__KG_ < 6000;
""")
boosters = cur.fetchall()

print("Boosters with success in drone ship and payload mass between 4000 and 6000:")
for booster in boosters:
  print(booster[0])
print('-----------------------------------')



# -----------------------------------------------------------------------
# Task7: List the total number of successful and failure mission outcomes
# -----------------------------------------------------------------------
cur.execute("""
  SELECT Mission_Outcome, COUNT(*)
  FROM SPACEXTBL
  WHERE Mission_Outcome IN ('Success', 'Failure')
  GROUP BY Mission_Outcome;
""")
mission_outcomes = cur.fetchall()

print("Total number of successful and failure mission outcomes:")
for outcome in mission_outcomes:
  print(f"{outcome[0]}: {outcome[1]}")
print('-----------------------------------')



# -----------------------------------------------------------------------
# Task8: List the names of the booster_versions which have carried
# the maximum payload mass. Use a subquery
# -----------------------------------------------------------------------
cur.execute("""
  SELECT Booster_Version
  FROM SPACEXTBL
  WHERE PAYLOAD_MASS__KG_ = (
      SELECT MAX(PAYLOAD_MASS__KG_)
      FROM SPACEXTBL
  );
""")
boosters = cur.fetchall()

print("Booster versions which have carried the maximum payload mass:")
for booster in boosters:
  print(booster[0])
print('-----------------------------------')



# -----------------------------------------------------------------------
# Task9: List the records which will display the month names, failure landing_outcomes
# in drone ship ,booster versions, launch_site for the months in year 2015.
# Note: SQLLite does not support monthnames.
# So you need to use substr(Date, 6,2) as month to get the months
# and substr(Date,0,5)='2015' for year.
# -----------------------------------------------------------------------
cur.execute("""
  SELECT substr(Date, 6, 2) AS Month, Booster_Version, Launch_Site
  FROM SPACEXTBL
  WHERE Landing_Outcome = 'Failure (drone ship)'
  AND substr(Date, 1, 4) = '2015';
""")
records = cur.fetchall()

# 月番号を月名に変換する辞書
month_names = {
  '01': 'January', '02': 'February', '03': 'March', '04': 'April',
  '05': 'May', '06': 'June', '07': 'July', '08': 'August',
  '09': 'September', '10': 'October', '11': 'November', '12': 'December'
}

print("Records for failure landing outcomes in drone ship in 2015:")
for record in records:
  month = month_names.get(record[0], "Unknown")  # 月番号を月名に変換
  print(f"Month: {month}, Booster Version: {record[1]}, Launch Site: {record[2]}")
print('-----------------------------------')



# -----------------------------------------------------------------------
# Task10: Rank the count of landing outcomes (such as Failure (drone ship)
# or Success (ground pad)) between the date 2010-06-04 and 2017-03-20, in descending order.
# -----------------------------------------------------------------------
cur.execute("""
  SELECT Landing_Outcome, COUNT(*) AS Count,
          (SELECT COUNT(*) FROM SPACEXTBL AS t2
          WHERE t2.Landing_Outcome <= t1.Landing_Outcome
          AND Date BETWEEN '2010-06-04' AND '2017-03-20') AS Rank
  FROM SPACEXTBL AS t1
  WHERE Date BETWEEN '2010-06-04' AND '2017-03-20'
  GROUP BY Landing_Outcome
  ORDER BY Count DESC;
""")
records = cur.fetchall()

print("Landing Outcome Ranking (2010-06-04 ~ 2017-03-20):")
for record in records:
    print(f"Landing Outcome: {record[0]}, Count: {record[1]}, Rank: {record[2]}")
print('-----------------------------------')



# コミットして接続を閉じる
con.commit()
con.close()
