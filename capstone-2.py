import requests
from bs4 import BeautifulSoup
import unicodedata
import pandas as pd


# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------
def date_time(table_cells):
  """
  This function returns the data and time from the HTML  table cell
  Input: the  element of a table data cell extracts extra row
  """
  return [data_time.strip() for data_time in list(table_cells.strings)][0:2]

def booster_version(table_cells):
  """
  This function returns the booster version from the HTML  table cell
  Input: the  element of a table data cell extracts extra row
  """
  out=''.join([booster_version for i,booster_version in enumerate( table_cells.strings) if i%2==0][0:-1])
  return out

def landing_status(table_cells):
  """
  This function returns the landing status from the HTML table cell
  Input: the  element of a table data cell extracts extra row
  """
  out=[i for i in table_cells.strings][0]
  return out

def get_mass(table_cells):
  mass=unicodedata.normalize("NFKD", table_cells.text).strip()
  if mass:
    mass.find("kg")
    new_mass=mass[0:mass.find("kg")+2]
  else:
    new_mass=0
  return new_mass

def extract_column_from_header(row):
  """
  This function returns the landing status from the HTML table cell
  Input: the  element of a table data cell extracts extra row
  """
  if (row.br):
    row.br.extract()
  if row.a:
    row.a.extract()
  if row.sup:
    row.sup.extract()

  colunm_name = ' '.join(row.contents)

  # Filter the digit and empty names
  if not(colunm_name.strip().isdigit()):
    colunm_name = colunm_name.strip()
    return colunm_name



# -----------------------------------------------------------------------
# TASK 1: Request the Falcon9 Launch Wiki page from its URL
# -----------------------------------------------------------------------
static_url = "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"
response = requests.get(static_url)


# -----------------------------------------------------------------------
# TASK 2: Extract all column/variable names from the HTML table header
# -----------------------------------------------------------------------
soup = BeautifulSoup(response.content, 'html.parser')
print(soup.find('title').string)

html_tables = soup.find_all('table')
first_launch_table = html_tables[2]

column_names = []
for header in first_launch_table.find_all('th'):
  column_name = extract_column_from_header(header)
  if column_name:
    column_names.append(column_name)


# -----------------------------------------------------------------------
# TASK 3: Create a data frame by parsing the launch HTML tables
# -----------------------------------------------------------------------
# Create an empty dictionary
launch_dict = dict.fromkeys(column_names)

# Remove an irrelevant column
del launch_dict['Date and time ( )']

# Let's initial the launch_dict with each value to be an empty list
launch_dict['Flight No.'] = []
launch_dict['Launch site'] = []
launch_dict['Payload'] = []
launch_dict['Payload mass'] = []
launch_dict['Orbit'] = []
launch_dict['Customer'] = []
launch_dict['Launch outcome'] = []
# Added some new columns
launch_dict['Version Booster'] = []
launch_dict['Booster landing'] = []
launch_dict['Date'] = []
launch_dict['Time'] = []

print(launch_dict)

extracted_row = 0

# Extract each table
for table_number, table in enumerate(soup.find_all('table', "wikitable plainrowheaders collapsible")):
  # get table row
  for rows in table.find_all("tr"):
    # check to see if first table heading is a number corresponding to launch number
    if rows.th:
      if rows.th.string:
        flight_number = rows.th.string.strip()
        flag = flight_number.isdigit()
    else:
      flag = False

    # get table element
    row = rows.find_all('td')

    # if it is a valid row, save cells in the dictionary
    if flag:
      extracted_row += 1
      # Flight Number value
      launch_dict["Flight No."].append(flight_number)

      # Date value
      datatimelist = date_time(row[0])
      date = datatimelist[0].strip(',')
      launch_dict["Date"].append(date)

      # Time value
      time = datatimelist[1]
      launch_dict["Time"].append(time)

      # Booster version
      bv = booster_version(row[1])
      if not bv:
        bv = row[1].a.string
      launch_dict["Version Booster"].append(bv)

      # Launch Site
      launch_site = row[2].a.string
      launch_dict["Launch site"].append(launch_site)

      # Payload
      payload = row[3].a.string
      launch_dict["Payload"].append(payload)

      # Payload Mass
      payload_mass = get_mass(row[4])
      launch_dict["Payload mass"].append(payload_mass)

      # Orbit
      orbit = row[5].a.string
      launch_dict["Orbit"].append(orbit)

      # Customer
      customer = row[6].a.string if row[6].a else row[6].get_text(strip=True)
      launch_dict["Customer"].append(customer)

      # Launch outcome
      launch_outcome = list(row[7].strings)[0]
      launch_dict["Launch outcome"].append(launch_outcome)

      # Booster landing
      booster_landing = landing_status(row[8])
      launch_dict["Booster landing"].append(booster_landing)

df = pd.DataFrame({ key:pd.Series(value) for key, value in launch_dict.items() })
df.to_csv('spacex_web_scraped.csv', index=False)