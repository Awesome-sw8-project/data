import os
import re
import json

#original amount    26925
#meta-data          20943
#meta + self        26921
#above + special    26925

# Different paths
metadata_path = '../metadata/'
train_path = '../train/'
test_path = '../test/'
save_location = './train/'

# Retrieve the floor number of those not defined in meta data and which is easier to convert
def get_floor_manually(floor):

    # Checks basement levels
    if re.match(r"^(B|LG)[0-9]+|[0-9]+B$",floor):
        floor = int(''.join(filter(str.isdigit, floor))) * -1
        return str(floor)
    
    # Checks floor levels above 
    if re.match(r"^(F|L)[0-9]|[0-9]F$",floor):
        floor = int(''.join(filter(str.isdigit, floor))) - 1
        return str(floor)

    # Hardcoded to handle the specific scenario of converting P1-level
    if floor == 'P1':
        return -3

    return "U"

# Retrieve the floor number from the meta data
def get_floor_num(data_in):
    with open(metadata_path + data_in + "/geojson_map.json") as json_file:
        data = json.loads(json_file.read())
        try:
            num = data['features'][0]['properties']['floor_num']
            if num > 0:
                num = num - 1
            return str(num)        
        except:
            return get_floor_manually(data_in.split('/')[1])

# Retrieve the relevant data from the comments in the original files        
def decode_comment(line):
    if 'SiteName' in line:
        if 'FloorId' not in line:
            return 'n'
        name = line[line.find('SiteName:') + 9 : line.rfind('FloorId:') - 1]
        return 'n' + name

    if 'startTime' in line: 
        time = line[line.find('startTime:') + 10 :]
        return 't' + time

    if 'endTime' in line: 
        end = line[line.find('endTime:') + 8 :]
        return 'e' + end


# Read the original data files and write the relevant info to a new file
def read_data_file(data_filename):
    path = data_filename.split('/')

    siteID = path[2]
    siteName = ""
    floorLevel = path[3]
    pathID = path[4].split('.')[0]
    startTime = ""
    endTime = ""
    sensorData = []
    waypoint = ['TYPE_WAYPOINT']

    floorLevel = get_floor_num(siteID + "/" + floorLevel)

    if floorLevel == "U": # If no floor is defined in the meta-data the new file will not be created
        return

    # Start reading the old file
    with open(data_filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line_data in lines:
        line_data = line_data.strip()
        if not line_data or line_data[0] == '#':

            comment = str(decode_comment(line_data))
            
            if comment[0] == 'n':
                siteName = comment[1:]
                continue

            if comment[0] == 't':
                startTime = comment[1:]
                continue

            if comment[0] == 'e':
                endTime = comment[1:]
                continue

            continue

        data = line_data.split('\t')

        if data[1] == 'TYPE_WAYPOINT':
            waypoint.append(data[0]+','+floorLevel+','+data[2]+','+data[3])
            continue

        sensorData.append(line_data)

    # Start writing to the new file
    with open(save_location + siteID + '_' + pathID + '.txt', 'w', encoding='utf-8') as file:
        file.write("#\t" + siteID + "\n")
        file.write("#\t" + siteName + "\n")
        file.write("#\t" + pathID + "\n")

        file.write(startTime)
        for point in waypoint:
            file.write("\t" + point)
        file.write("\n")

        for line in sensorData:
            file.write(line + "\n")

        file.write("#\t" + endTime)


if __name__ == "__main__":
    # Make folder to save data in
    if os.path.exists(save_location):
        os.mkdir(save_location)

    # Find all the paths
    for site in os.listdir(train_path):
        site_path = os.path.join(train_path, site)
        for floor in os.listdir(site_path):
            floors = os.path.join(site_path, floor)
            for path in os.listdir(floors):
                paths = os.path.join(floors, path)
                read_data_file(paths)