from pymatgen.command_line.critic2_caller import Critic2Caller
import json

#this function is here because pmg has a dumb bug
def get_zpsp(path: str) -> dict:
    '''Gets valence value from POTCAR'''
    zpsp_ints = []
    zpsp_strs = []
    with open(path, 'r') as f:
        for line in f:
            if 'POMASS' in line:
                zpsp_ints.append(int(line.split()[5].split('.')[0]))
            if 'TITEL' in line:
                label = line.split()[3]
                label = label.split('_')[0]
                zpsp_strs.append(label)

    zpsp = dict(zip(zpsp_strs, zpsp_ints))

    return zpsp

#potcar_path has to be here until they approve my pull request
def get_critical_points(path: str, potcar_path: str):
    '''Gets data from critic2 output'''
    zpsp: dict = get_zpsp(potcar_path)
    critic2_data = Critic2Caller.from_path(path, zpsp=zpsp)
    data: dict = critic2_data._cp_report

    return data

def write_critical_points_to_json(path: str, potcar_path: str):
    '''Writes critic2 output to json file'''
    zpsp: dict = get_zpsp(potcar_path)
    data: dict = get_critical_points(path, potcar_path) 

    with open('critic2_data.json', 'w') as f:
        json.dump(data, f, indent=4)

# Example usage:
# write_critical_points_to_json("/Users/wladerer/github/qtaim_generator/tests/test_files/solidstate", "/Users/wladerer/github/qtaim_generator/tests/test_files/solidstate/pot")

