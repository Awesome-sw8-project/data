import os
import re
import json
import pandas as pd


# Returns Dataframe with information from the sample_submission.csv
def reformat_sample_submission(sample_submission_path, reformatted_ss):
    sample_submission = pd.read_csv(sample_submission_path)

    data = pd.DataFrame(columns = ['Site', 'Path', 'Timestamp'])

    for index, row in sample_submission.iterrows():
        split = str(row['site_path_timestamp']).split('_')
        data.loc[len(data.index)] = [split[0], split[1], str(split[2])]  
    
    data.to_csv(reformatted_ss)

# Retrieve reformatted sample submission data
def load_reformatted_ss_data (reformatted_ss):
    if os.path.exists(reformatted_ss):
        data = pd.read_csv(reformatted_ss, dtype={'Timestamp':str})

    return data


# Retrieve the relevant data from the comments in the original test files        
def decode_comment(line):
    if 'SiteID' in line:
        siteid = line[line.find('SiteID:') + 7 : line.rfind('SiteName:') - 1]
        name = line[line.find('SiteName:') + 9 : ]
        return 'n' + siteid + '/' + name

    if 'startTime' in line: 
        time = line[line.find('startTime:') + 13 :]
        return 't' + time

    if 'endTime' in line: 
        end = line[line.find('endTime:') + 11 :]
        return 'e' + end

# Create empty waypoints from sample_submission timestamps
def create_waypoints(sample_submission, pathID):
    waypoint = ['TYPE_WAYPOINT']
    pathdef = sample_submission.loc[sample_submission['Path'] == pathID]
        
    for index, row in pathdef.iterrows():
        waypoint.append(str(row['Timestamp']) + ',' + ',' + ',')
    
    return waypoint

# Read the original data files and write the relevant info to a new file
def read_data_file(data_filename, sample_submission, save_location):
    path = data_filename.split('/')
    pathID = path[2].split('.')[0]
    sensorData = []
    waypoints = create_waypoints(sample_submission, pathID)

    # Start reading the old file
    with open(data_filename, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()

    for line_data in lines:
        line_data = line_data.strip()
        
        if not line_data or line_data[0] == '#':
            comment = str(decode_comment(line_data))
            
            if comment[0] == 'n':
                divided_comment = comment.split('/')
                siteID = str(divided_comment[0])[1:]
                siteName = str(divided_comment[1])
                continue

            if comment[0] == 't':
                startTime = comment[1:]
                continue

            if comment[0] == 'e':
                endTime = comment[1:]
                continue

            continue

        data = line_data.split('\t')

        sensorData.append(line_data)

    # Start writing to the new file
    with open(save_location + siteID + '_' + pathID + '.txt', 'w', encoding='utf-8') as file:
        file.write("#\t" + siteID + "\n")
        file.write("#\t" + siteName + "\n")
        file.write("#\t" + pathID + "\n")

        file.write(startTime)
        for point in waypoints:
            file.write("\t" + point)
        file.write("\n")

        for line in sensorData:
            file.write(line + "\n")

        file.write("#\t" + endTime)


if __name__ == "__main__":
    # Different paths
    test_folder = '../test/'
    sample_submission = '../sample_submission.csv'
    reformatted_test = './test_reformatted/'
    reformatted_ss = 'sample_submission_reformatted.csv'

    # Reformat sample_submission.csv to divide the Ids into Site, Path and Timestamp
    if not os.path.exists(reformatted_ss):
        reformat_sample_submission(sample_submission, reformatted_ss)
    
    # Load reformatted sample_submission file
    sample_submission_data = load_reformatted_ss_data(reformatted_ss)

    # Create folder to save new test data in
    if not os.path.exists(reformatted_test):
        os.mkdir(reformatted_test)

    # Find all the paths
    for path in os.listdir(test_folder):
        if not path == '.DS_Store':
            paths = os.path.join(test_folder, path)
            read_data_file(paths, sample_submission_data, reformatted_test)