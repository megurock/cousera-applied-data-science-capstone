import folium
import pandas as pd
from folium.plugins import MarkerCluster
from folium.plugins import MousePosition
from folium.features import DivIcon

# --------------------------------------------------------------------------------
# TASK 1: Mark all launch sites on a map
# --------------------------------------------------------------------------------
df = pd.read_csv('./spacex_launch_geo.csv')
print(df.head())

# 関連するサブカラムを選択
spacex_df = df[['Launch Site', 'Lat', 'Long', 'class']]

# 発射サイトごとに最初のデータを取得（グループ化して最初の行を選択）
launch_sites_df = spacex_df.groupby(['Launch Site'], as_index=False).first()

# 発射サイト、緯度、経度のみの列を残す
launch_sites_df = launch_sites_df[['Launch Site', 'Lat', 'Long']]
print(launch_sites_df)

# 地図の初期位置をNASA Johnson Space Centerに設定
nasa_coordinate = [29.559684888503615, -95.0830971930759]
site_map = folium.Map(location=nasa_coordinate, zoom_start=5)

# NASA Johnson Space Centerの座標に円を追加し、ポップアップラベルを表示
circle = folium.Circle(nasa_coordinate, radius=1000, color='#d35400', fill=True).add_child(folium.Popup('NASA Johnson Space Center'))

# NASA Johnson Space Centerの座標にマーカーを追加し、アイコンとしてテキストラベルを表示
marker = folium.map.Marker(
  nasa_coordinate,
  icon=DivIcon(
    icon_size=(20, 20),
    icon_anchor=(0, 0),
    html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % 'NASA JSC',
  )
)

site_map.add_child(circle)
site_map.add_child(marker)

# 発射サイトごとに円とマーカーを地図に追加
for _, row in launch_sites_df.iterrows():
  site_coordinate = [row['Lat'], row['Long']]
  site_name = row['Launch Site']

  # 発射サイトの座標に円を追加
  circle = folium.Circle(
    location=site_coordinate,
    radius=500,  # 円の半径
    color='blue',
    fill=True,
    fill_color='blue',
    fill_opacity=0.5
  ).add_child(folium.Popup(site_name))

  # 発射サイトの座標にマーカーを追加し、名前をポップアップラベルとして表示
  marker = folium.map.Marker(
    site_coordinate,
    icon=DivIcon(
      icon_size=(20, 20),
      icon_anchor=(0, 0),
      html='<div style="font-size: 12; color:#0000FF;"><b>%s</b></div>' % site_name,
    )
  )

  site_map.add_child(circle)
  site_map.add_child(marker)

# 地図をHTMLファイルとして保存
site_map.save("spacex_launch_sites_map.html")




# --------------------------------------------------------------------------------
# TASK 2: Mark the success/failed launches for each site on the map
# --------------------------------------------------------------------------------
print(spacex_df.tail(10))
marker_cluster = MarkerCluster()

# 新しい列 `marker_color` を作成し、class に基づいて色を設定
# class=1 (成功) は緑、class=0 (失敗) は赤に設定
spacex_df_copy = spacex_df.copy()
spacex_df_copy['marker_color'] = spacex_df_copy['class'].apply(lambda x: 'green' if x == 1 else 'red')



# 地図の初期位置を設定（例えば、NASA Johnson Space Center）
nasa_coordinate = [29.559684888503615, -95.0830971930759]
site_map = folium.Map(location=nasa_coordinate, zoom_start=5)


# マーカークラスタを作成
marker_cluster = MarkerCluster()

# spacex_dfの各行に対してマーカーを作成
for index, record in spacex_df_copy.iterrows():
  # 発射サイトの緯度経度
  site_coordinate = [record['Lat'], record['Long']]
  # 発射結果に基づく色
  marker_color = record['marker_color']

  # マーカーを作成
  marker = folium.Marker(
    location=site_coordinate,
    icon=folium.Icon(color=marker_color, icon='info-sign'),
    popup=f"Launch Site: {record['Launch Site']}\nStatus: {'Success' if record['class'] == 1 else 'Failure'}"
  )

  # マーカークラスタに追加
  marker_cluster.add_child(marker)

# マーカークラスタを地図に追加
site_map.add_child(marker_cluster)

# 地図を表示
site_map.save("spacex_launch_status_map.html")


# --------------------------------------------------------------------------------
# TASK 3: Calculate the distances between a launch site to its proximities
# --------------------------------------------------------------------------------
# Add Mouse Position to get the coordinate (Lat, Long) for a mouse over on the map
formatter = "function(num) {return L.Util.formatNum(num, 5);};"
mouse_position = MousePosition(
  position='topright',
  separator=' Long: ',
  empty_string='NaN',
  lng_first=False,
  num_digits=20,
  prefix='Lat:',
  lat_formatter=formatter,
  lng_formatter=formatter,
)

site_map.add_child(mouse_position)
site_map

from math import sin, cos, sqrt, atan2, radians

def calculate_distance(lat1, lon1, lat2, lon2):
  # approximate radius of earth in km
  R = 6373.0

  lat1 = radians(lat1)
  lon1 = radians(lon1)
  lat2 = radians(lat2)
  lon2 = radians(lon2)

  dlon = lon2 - lon1
  dlat = lat2 - lat1

  a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
  c = 2 * atan2(sqrt(a), sqrt(1 - a))

  distance = R * c
  return distance


# TODO: Mark down a point on the closest coastline using MousePosition and calculate the distance between the coastline point and the launch site.
# find coordinate of the closet coastline
# e.g.,: Lat: 28.56367  Lon: -80.57163
launch_site_lat, launch_site_lon = 28.563197, -80.576820  # 例: SpaceX Cape Canaveral SLC-40
coastline_lat, coastline_lon = 28.56367, -80.57163  # 例: 最寄りの海岸線の座標

# 2. 海岸線との距離を計算
distance_coastline = calculate_distance(launch_site_lat, launch_site_lon, coastline_lat, coastline_lon)


# Create and add a folium.Marker on your selected closest coastline point on the map
# Display the distance between coastline point and launch site using the icon property
# for example
# 3. 距離を表示するマーカーを作成
distance_marker = folium.Marker(
  location=[coastline_lat, coastline_lon],
  icon=DivIcon(
    icon_size=(20, 20),
    icon_anchor=(0, 0),
    html='<div style="font-size: 12px; color:#d35400;"><b>{:.2f} KM</b></div>'.format(distance_coastline),
  )
)


# TODO: Draw a PolyLine between a launch site to the selected coastline point
# Create a `folium.PolyLine` object using the coastline coordinates and launch site coordinate
# 4. ポリラインを描画
lines = folium.PolyLine(
  locations=[[launch_site_lat, launch_site_lon], [coastline_lat, coastline_lon]],
  weight=2,
  color='blue'
)
site_map.add_child(lines)

# Create a marker with distance to a closest city, railway, highway, etc.
# Draw a line between the marker to the launch site
site_map.add_child(distance_marker)
site_map.add_child(lines)

# 6. 最寄りの都市・鉄道・高速道路の地点を追加（例: 高速道路）
highway_lat, highway_lon = 28.57205, -80.58528  # 例: 最寄りの高速道路
distance_highway = calculate_distance(launch_site_lat, launch_site_lon, highway_lat, highway_lon)

# 高速道路の距離マーカー
highway_marker = folium.Marker(
  location=[highway_lat, highway_lon],
  icon=DivIcon(
    icon_size=(20, 20),
    icon_anchor=(0, 0),
    html='<div style="font-size: 12px; color:#d35400;"><b>{:.2f} KM</b></div>'.format(distance_highway),
  )
)

# 高速道路とのポリライン
highway_line = folium.PolyLine(
  locations=[[launch_site_lat, launch_site_lon], [highway_lat, highway_lon]],
  weight=2,
  color='red'
)

# 地図に追加
site_map.add_child(highway_marker)
site_map.add_child(highway_line)

# 地図を表示
site_map.save("spacex_launch_map.html")