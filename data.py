import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.max_columns = 50
pd.options.display.max_colwidth  = 200
import os
from IPython.display import display

from dataclasses import dataclass
import colorama
from colorama import Fore, Back, Style
import folium
import json
import geopandas as gpd

import re
import pyproj
from pyproj import Proj, transform

from shapely.ops import cascaded_union
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
plt.style.use('fivethirtyeight')
import seaborn as sns # visualization
import warnings # Supress warnings 
warnings.filterwarnings('ignore')

import plotly.graph_objs as go
from PIL import Image

from tqdm import tqdm

metadata_path = '../metadata/'
train_path = '../train/'
test_path = '../test/'
sub = pd.read_csv('../sample_submission.csv')


y_ = Fore.YELLOW
r_ = Fore.RED
g_ = Fore.GREEN
b_ = Fore.BLUE
m_ = Fore.MAGENTA
c_ = Fore.CYAN
sr_ = Style.RESET_ALL

color_dict = {'site': c_, 'floor': y_, 'path': b_}

test_structure = {test_path: ['path_1.txt','path_2.txt','path_3.txt','...', 'path_n.txt']}

metadata_structure = {metadata_path: 
                               {'site_1': {'floor_1': ['geojson_map.json', 'floor_info.json', 'floor_image.png'],
                                           'floor_2': ['geojson_map.json', 'floor_info.json', 'floor_image.png']},
                                'site_2': {'basement': ['geojson_map.json', 'floor_info.json', 'floor_image.png'],
                                           'floor_1': ['geojson_map.json', 'floor_info.json', 'floor_image.png']},
                               }
                     }

train_structure = {train_path: 
                               {'site_1': {'floor_1': ['path_1.txt', 'path_2.txt'],
                                           'floor_2': ['path_1.txt', 'path_2.txt', 'path_3.txt']},
                                'site_2': {'basement': ['path_1.txt'],
                                           'floor_1': ['path_1.txt', 'path_2.txt']},
                               }
                     }

def pretty(d, indent=0, max_enum = 10):
    for enum, (key, value) in enumerate(d.items()):
        if enum < max_enum:
            if ((len(str(key)) < 5) or (any(x in str(key) for x in ['floor', 'basement']))) and ('site' not in str(key)):
                print('\t'*indent, color_dict['floor'] + str(key)) 
            
            elif ((len(str(key)) > 5)):
                print('\t'*indent, color_dict['site'] + str(key)) 
            
            else:
                print('\t' * indent + str(key))
            if isinstance(value, dict):
                pretty(value, indent+1)
            else:
                if (len(value)>0) & (any(x in str(value) for x in ['.json', '.txt', '.png'])):
                    print("""{0}{1}{2}""".format('\t'*(indent+1), color_dict['path'], str(value)))
                else: 
                    print('\t' * (indent+1) + str(value))
        print(Style.RESET_ALL)
                    
def create_dict(metadata_path, max_enum = 1000, files_enum = None):
    
    metadata_dict = {}
    sites = os.listdir(metadata_path)
    metadata_dict[metadata_path] = sites
    sites_path = list(map(lambda x: os.path.join(metadata_path, x), sites))
    sites_dict = {}
    for sites_enum, site_path in enumerate(sites_path):
        
        if sites_enum<max_enum:
            
            site_floors = os.listdir(site_path)
            floors_path = list(map(lambda x: os.path.join(site_path, x), site_floors)) 
            
            floor_dict = {}
            for floor_enum, floor in enumerate(floors_path): 
                if floor_enum<max_enum:
                    if files_enum:
                        floor_dict[site_floors[floor_enum]] = len(os.listdir(floor)[:files_enum])
                    else:
                        floor_dict[site_floors[floor_enum]] = len(os.listdir(floor))
                        
            sites_dict[sites[sites_enum]] = floor_dict
                    
                    
    return {metadata_path: sites_dict}
                    
# copy from https://github.com/location-competition/indoor-location-competition-20/blob/master/io_f.py

@dataclass
class ReadData:
    acce: np.ndarray
    acce_uncali: np.ndarray
    gyro: np.ndarray
    gyro_uncali: np.ndarray
    magn: np.ndarray
    magn_uncali: np.ndarray
    ahrs: np.ndarray
    wifi: np.ndarray
    ibeacon: np.ndarray
    waypoint: np.ndarray


def read_data_file(data_filename):
    acce = []
    acce_uncali = []
    gyro = []
    gyro_uncali = []
    magn = []
    magn_uncali = []
    ahrs = []
    wifi = []
    ibeacon = []
    waypoint = []

    with open(data_filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line_data in lines:
        line_data = line_data.strip()
        if not line_data or line_data[0] == '#':
            continue

        line_data = line_data.split('\t')

        if line_data[1] == 'TYPE_WAYPOINT':
            waypoint.append([int(line_data[0]), float(line_data[2]), float(line_data[3])])
            continue
       
        if line_data[1] == 'TYPE_ACCELEROMETER':
            acce.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue
        
        if line_data[1] == 'TYPE_ACCELEROMETER_UNCALIBRATED':
            acce_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue
        
        if line_data[1] == 'TYPE_GYROSCOPE':
            gyro.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_GYROSCOPE_UNCALIBRATED':
            gyro_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue
        
        if line_data[1] == 'TYPE_MAGNETIC_FIELD':
            magn.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_MAGNETIC_FIELD_UNCALIBRATED':
            magn_uncali.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_ROTATION_VECTOR':
            #ahrs.append([int(line_data[0]), float(line_data[2]), float(line_data[3]), float(line_data[4])])
            continue

        if line_data[1] == 'TYPE_WIFI':
            sys_ts = line_data[0]
            ssid = line_data[2]
            bssid = line_data[3]
            rssi = line_data[4]
            lastseen_ts = line_data[6]
            wifi_data = [sys_ts, ssid, bssid, rssi, lastseen_ts]
            wifi.append(wifi_data)
            continue

        if line_data[1] == 'TYPE_BEACON':
            ts = line_data[0]
            uuid = line_data[2]
            major = line_data[3]
            minor = line_data[4]
            rssi = line_data[6]
            ibeacon_data = [ts, '_'.join([uuid, major, minor]), rssi]
            ibeacon.append(ibeacon_data)
            continue
        
    
    acce = np.array(acce)
    acce_uncali = np.array(acce_uncali)
    gyro = np.array(gyro)
    gyro_uncali = np.array(gyro_uncali)
    magn = np.array(magn)
    magn_uncali = np.array(magn_uncali)
    ahrs = np.array(ahrs)
    wifi = np.array(wifi)
    ibeacon = np.array(ibeacon)
    waypoint = np.array(waypoint)
    #print(waypoint.shape[0]) # amount of waypoints

    
    #print('Acce shape:', acce.shape)
    #print('acce_uncali shape:', acce_uncali.shape)
    #print('gyro shape:', gyro.shape)
    #print('gyro_uncali shape:', gyro_uncali.shape)
    #print('magn shape:', magn.shape)
    #print('magn_uncali shape:', magn_uncali.shape)
    #print('ahrs shape:', ahrs.shape)
    #print('wifi shape:', wifi.shape)
    #print('ibeacon shape:', ibeacon.shape)
    #print('Waypoint shape:', waypoint.shape)
    
    return ReadData(acce, acce_uncali, gyro, gyro_uncali, magn, magn_uncali, ahrs, wifi, ibeacon, waypoint)

def visualize_trajectory(trajectory, floor_plan_filename, width_meter, 
                         height_meter, title=None, mode='lines + markers + text', show=True):
    """
    Copied from from https://github.com/location-competition/indoor-location-competition-20/blob/master/visualize_f.py

    """
    fig = go.Figure()

    # add trajectory
    size_list = [6] * trajectory.shape[0]
    size_list[0] = 10
    size_list[-1] = 10

    color_list = ['rgba(4, 174, 4, 0.5)'] * trajectory.shape[0]
    color_list[0] = 'rgba(12, 5, 235, 1)'
    color_list[-1] = 'rgba(235, 5, 5, 1)'

    position_count = {}
    text_list = []
    for i in range(trajectory.shape[0]):
        if str(trajectory[i]) in position_count:
            position_count[str(trajectory[i])] += 1
        else:
            position_count[str(trajectory[i])] = 0
        text_list.append('        ' * position_count[str(trajectory[i])] + f'{i}')
    text_list[0] = 'Start 0'
    text_list[-1] = f'End {trajectory.shape[0] - 1}'

    fig.add_trace(
        go.Scattergl(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            mode=mode,
            marker=dict(size=size_list, color=color_list),
            line=dict(shape='linear', color='lightgrey', width=3, dash='dash'),
            text=text_list,
            textposition="top center",
            name='trajectory',
        ))

    # add floor plan
    floor_plan = Image.open(floor_plan_filename)
    fig.update_layout(images=[
        go.layout.Image(
            source=floor_plan,
            xref="x",
            yref="y",
            x=0,
            y=height_meter,
            sizex=width_meter,
            sizey=height_meter,
            sizing="contain",
            opacity=1,
            layer="below",
        )
    ])

    # configure
    fig.update_xaxes(autorange=False, range=[0, width_meter])
    fig.update_yaxes(autorange=False, range=[0, height_meter], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=go.layout.Title(
            text=title or "No title.",
            xref="paper",
            x=0,
        ),
        autosize=True,
        width=800,
        height=  800 * height_meter / width_meter,
        template="plotly_white",
    )

    if show:
        fig.show()

    return fig

def visualize_train_trajectory(path):
    """
    Edited from 
    https://www.kaggle.com/ihelon/indoor-location-exploratory-data-analysis
    """
    _id, floor = path.split("/")[:2]
    
    train_floor_data = read_data_file(f"../train/{path}")
    with open(f"../metadata/{_id}/{floor}/floor_info.json") as f:
        train_floor_info = json.load(f)

    return visualize_trajectory(
        train_floor_data.waypoint[:, 1:3], 
        f"../metadata/{_id}/{floor}/floor_image.png",
        train_floor_info["map_info"]["width"], 
        train_floor_info["map_info"]["height"],
        f"Visualization of {path}"
    )

geo_dfs = []
geo_cols = ["geometry","Vr","category","name","code","floor_num", 'sid',
            "type","id","version","display","point","points","doors", "site_name"]

problematic_sites = []
for site in os.listdir(metadata_path):
    site_path = os.path.join(metadata_path, site)
    for floor in os.listdir(site_path):
        floor_path = os.path.join(site_path, floor)
        try:
            geo_df = (gpd.GeoDataFrame.from_features(
                        pd.read_json(os.path.join(floor_path, 'geojson_map.json'))['features'])
                     .assign(site_name=site))
        except:
            problematic_sites+=[site]
        geo_dfs.append(geo_df)
problematic_sites=list(set(problematic_sites))
full_geo_df = pd.concat(geo_dfs, axis = 0, ignore_index = True)

full_geo_df[['geometry', 'point', 'site_name']].sample()

################


def get_lat_lon(point, proj = pyproj.Transformer.from_crs(3857, 4326, always_xy=True)):
    try:
        x1, y1 = point[0], point[1]
        lon, lat = proj.transform(x1, y1)
        return lat, lon
    except:
        return np.nan

def get_point(x, i=0):
    try:
        return x[i]
    except:
        return np.nan
    
full_geo_df_sample = full_geo_df.sample(500).reset_index(drop = True)
full_geo_df_sample['lat_lon'] = full_geo_df_sample.point.apply(get_lat_lon)
full_geo_df_sample['lat'] = full_geo_df_sample['lat_lon'].apply(lambda x: get_point(x,0))
full_geo_df_sample['lon'] = full_geo_df_sample['lat_lon'].apply(lambda x: get_point(x,1))

##################

train_dict = create_dict(train_path)[train_path]
train_path_df = pd.DataFrame.from_dict(train_dict, orient = 'index')

assert train_path_df[train_path_df == 0].sum().sum() == 0, "Floor present in Site, but no path available"

train_path_df['number_of_floors'] = train_path_df.apply(lambda x: ~x.isna()).sum(axis = 1)

train_path_df = (train_path_df.reset_index(drop= False).rename(columns = {'index': 'site'})
 .melt(ignore_index = 'False', id_vars = ['site', 'number_of_floors'], var_name = 'floor',
      value_name = 'number_of_paths'))

train_path_df = train_path_df.loc[~train_path_df.number_of_paths.isna()].reset_index(drop = True)

#################

floor_meta_info = (full_geo_df.loc[~full_geo_df.floor_num.isna()]
                   [['site_name', 'name', 'floor_num']].reset_index(drop = True))
train_path_df_plus_meta = (train_path_df.merge(floor_meta_info, 
                           left_on = ['site', 'floor'], right_on = ['site_name', 'name']))

################

height = 17
width = 20

font = {
    'family' : 'DejaVu Sans',
    'weight' : 'bold',
    'size'   : 22
}

matplotlib.rc('font', **font)

fig_floors, ax_floors = plt.subplots( nrows=1, ncols=1, figsize = (width, height) )
fig_total_path, ax_total_path = plt.subplots( nrows=1, ncols=1, figsize = (width, height) )
fig_avg_path, ax_avg_path = plt.subplots( nrows=1, ncols=1, figsize = (width, height) )



fig, axes = plt.subplots(2, 2, figsize = (20, 12))
ax = axes.ravel()
plot_df = train_path_df[['site', 'number_of_floors']].drop_duplicates(ignore_index = True)
ax[0]= plt.subplot2grid((2, 2), (0, 0), colspan=1)
plot_df.number_of_floors.hist(ax = ax[0], bins = 50, color = '#2695f0')
ax[0].set_title('Number of Floors per Site distribution')

################


floor_numbers = np.array([x for x in train_path_df_plus_meta.floor_num])

num_of_floors = (train_path_df_plus_meta.groupby('floor_num').agg({'floor_num': ['count']}).reset_index())
num_of_floors.columns = ['floor_num', 'amount']

ax[1]= plt.subplot2grid((2, 2), (0, 1))
#plt.hist(floor_numbers, bins = 30, color = '#2695f0')
num_of_floors.plot(kind = 'bar', x = 'floor_num', y = 'amount', ax = ax_floors, color = '#2695f0', legend=None)
ax_floors.set_title('Distribution of Standard Numbered Floors')

#ax[1]= plt.subplot2grid((2, 2), (0, 1))
#train_path_df.number_of_paths.hist(ax = ax[1], bins = 30, color = '#2695f0')
#ax[1].set_title('Number of Paths per Floor and Site distribution')


##############




ax[2] = plt.subplot2grid((2, 2), (1, 0), colspan=1)

plot_df_3 = (train_path_df_plus_meta.groupby('floor_num').agg({'number_of_paths': ['sum', 'mean']}).reset_index())
plot_df_3.columns = ['floor_num', 'total_paths', 'avg_paths']
plot_df_3['avg_paths'] = round(plot_df_3['avg_paths'], 3)
#plot_df_3 = plot_df_3.melt(id_vars = 'floor_num')

plot_df_3.plot(kind = 'bar', x = 'floor_num', y = 'total_paths', ax = ax_total_path, color = '#f0b326')
#plot_df_3.plot(kind = 'bar', x = 'floor_num', y = 'avg_paths', ax = ax[2])
ax_total_path.set_title('Total Number of Paths per Floor Number')

ax[3] = plt.subplot2grid((2, 2), (1, 1), colspan=1)
plot_df_3.plot(kind = 'bar', x = 'floor_num', y = 'avg_paths', ax = ax_avg_path, color = '#f0b326')
ax_avg_path.set_title('Average Number of Paths per Floor Number')

ax[0].set_xlabel('N_floors')
ax_floors.set_xlabel('Floor number')
ax_floors.set_ylabel('Number of Floors')
ax_total_path.set_xlabel('Floor number')
ax_total_path.set_ylabel('Total Number of Paths')
ax_avg_path.set_xlabel('Floor number')
ax_avg_path.set_ylabel('Average Number of Paths')

ax_total_path.get_legend().remove()
ax_avg_path.get_legend().remove()

#fig.savefig("allInfo.png")
#plt.show()



fig_floors.savefig("datadistribution1.png")
fig_total_path.savefig("datadistribution2.png")
fig_avg_path.savefig("datadistribution3.png")

##################
##################
##################
##################

def count_all_waypoint_on_floor(path):
    total_waypoints = 0
    try:
        paths = os.listdir(path)
    except:
        return 0

    paths = list(map(lambda x: os.path.join(path, x), paths))
    
    for current_path in paths:
        info = read_data_file(current_path)
        total_waypoints = total_waypoints + info.waypoint.shape[0]
        #print(current_path + ' ' + str(info.waypoint.shape[0]))

    #print(path + ' ' + str(total_waypoints))
    return total_waypoints

#print(train_path_df_plus_meta)

def get_floors():
    floors = []
    for index, row in floor_meta_info.iterrows():
        sites = os.path.join(train_path, row['site_name'])
        floor = os.path.join(sites, row['name'])
        floors.append(floor)

    return floors

waypoint_count = []
#with open("waypoint_count", "a") as f:
#    for floor in get_floors():
#        count = count_all_waypoint_on_floor(floor)
#        f.write(str(count)+"\n")
#        waypoint_count.append(count)
#        print(count)

with open("waypoint_count", "r") as f:
    content = f.readlines()

waypoint_count = [int(x.strip()) for x in content] 

floor_meta_info['waypoint_amount'] = waypoint_count


######################



plot_df_waypoint = (floor_meta_info.groupby('floor_num').agg({'waypoint_amount': ['sum', 'mean']}).reset_index())
plot_df_waypoint.columns = ['floor number', 'total_waypoints', 'avg_waypoints']
plot_df_waypoint['avg_waypoints'] = round(plot_df_waypoint['avg_waypoints'], 3)

fig_way_total, ax_total = plt.subplots( nrows=1, ncols=1, figsize = (width, height) )

plot_df_waypoint.plot(kind = 'bar', x = 'floor number', y = 'total_waypoints', ax=ax_total, color = '#ea4c46', legend=None)
ax_total.set_title('Total Number of Waypoints per Floor Number')
ax_total.set_ylabel('Total Number of Waypoints')

fig_way_total.savefig("datadistribution4.png")


fig_way_avg, ax_avg = plt.subplots( nrows=1, ncols=1, figsize = (width, height) )

plot_df_waypoint.plot(kind = 'bar', x = 'floor number', y = 'avg_waypoints', ax=ax_avg, color = '#ea4c46', legend=None)
ax_avg.set_title('Average Number of Waypoints per Floor Number')
ax_avg.set_ylabel('Average Number of Waypoints')

fig_way_avg.savefig("datadistribution5.png")
#plt.show()

#####################################################################

sub[['site', 'path', 'timestamp']] = sub['site_path_timestamp'].str.split('_', expand=True)

#print(num_of_floors)
#display(sub)

plot_df_sub = (sub.groupby('site').agg({'path': ['count']}).reset_index())

display(plot_df_sub)

total_paths = 0

#print(waypoint_count)
#print(train_path_df_plus_meta)
#print(floor_meta_info)
#print(plot_df_waypoint)
##################################################################
#print(count_all_waypoint_on_floor('5a0546857ecc773753327266/B1'))
##################################################################

