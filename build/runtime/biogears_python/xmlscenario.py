import xml.etree.ElementTree as ET
import xml.dom.minidom
import pandas as pd # type: ignore

def segments_to_xml(segments):
    # Define namespaces
    namespaces = {
        'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
    }
    
    # Create the root element with the namespace
    scenario = ET.Element('Scenario', attrib={
        'xmlns': 'uri:/mil/tatrc/physiology/datamodel',
        'xmlns:xsi': namespaces['xsi'],
        'contentVersion': 'BioGears_6.3.0-beta',
        'xsi:schemaLocation': ''
    })
    
    # Add Name and Description elements
    name = ET.SubElement(scenario, 'Name')
    name.text = 'Dynamic Scenario'
    
    description = ET.SubElement(scenario, 'Description')
    description.text = 'Generated scenario based on segments data.'
    
    # Add InitialParameters element
    initial_parameters = ET.SubElement(scenario, 'InitialParameters')
    patient_file = ET.SubElement(initial_parameters, 'PatientFile')
    patient_file.text = 'StandardMale.xml'
    
    # Add DataRequests element
    data_requests = ET.SubElement(scenario, 'DataRequests')
    data_request_params = [
        {'Name': 'HeartRate', 'Unit': '1/min'},
        #{'Name': 'CardiacOutput', 'Unit': 'mL/min'},
        #{'Name': 'MeanArterialPressure', 'Unit': 'mmHg'},
        #{'Name': 'SystolicArterialPressure', 'Unit': 'mmHg'},
        #{'Name': 'DiastolicArterialPressure', 'Unit': 'mmHg'},
        #{'Name': 'TotalMetabolicRate', 'Unit': 'kcal/day'},
        {'Name': 'CoreTemperature', 'Unit': 'degC'},
        {'Name': 'SkinTemperature', 'Unit': 'degC'},
        #{'Name': 'RespirationRate', 'Unit': '1/min'},
        #{'Name': 'AchievedExerciseLevel'},
        #{'Name': 'FatigueLevel'},
        #{'Name': 'TotalWorkRate', 'Unit': 'W'},
        #{'Name': 'TotalMetabolicRate', 'Unit': 'W'},
        #{'Name': 'TotalWorkRateLevel'}
    ]
    
    for param in data_request_params:
        ET.SubElement(data_requests, 'DataRequest', attrib={'xsi:type': 'PhysiologyDataRequestData', **param})

    # Add actions to the XML directly under the Scenario element
    for i in range(len(segments['time'])):
        # Add EnvironmentChangeData action
        environment_action = ET.SubElement(scenario, 'Action', attrib={'xsi:type': 'EnvironmentChangeData'})
        conditions = ET.SubElement(environment_action, 'Conditions')
        ambient_temperature = ET.SubElement(conditions, 'AmbientTemperature', attrib={'value': str(segments['atemp_c'][i]), 'unit': 'degC'})
        relative_humidity = ET.SubElement(conditions, 'RelativeHumidity', attrib={'value': str(segments['rh_pct'][i] / 100.0)})  # Convert percentage to fraction
        
        # Add ExerciseData action
        exercise_action = ET.SubElement(scenario, 'Action', attrib={'xsi:type': 'ExerciseData'})
        intensity = ET.SubElement(exercise_action, 'Intensity', attrib={'value': str(segments['intensity'][i])})
        
        # Add AdvanceTimeData action
        advance_time_action = ET.SubElement(scenario, 'Action', attrib={'xsi:type': 'AdvanceTimeData'})
        time = ET.SubElement(advance_time_action, 'Time', attrib={'value': str(segments['time'][i]), 'unit': 'min'})
    
    # Convert to string
    xml_string = ET.tostring(scenario, encoding='unicode')
    
    # Add XML declaration
    xml_string = '<?xml version="1.0"?>\n' + xml_string
    
    return xml_string

def save_xml_to_file(xml_string, file_path):
    with open(file_path, 'w') as file:
        file.write(xml_string)

def pretty_print_xml(xml_string):
    dom = xml.dom.minidom.parseString(xml_string)
    pretty_xml_as_string = dom.toprettyxml()
    print(pretty_xml_as_string)