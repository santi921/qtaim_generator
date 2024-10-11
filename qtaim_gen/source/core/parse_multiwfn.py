def parse_charge_doc(charge_out_txt):
    """
    Method to parse the charge out from multiwfn
    Takes: 
        charge_out_txt: str, path to the charge out file
    Returns: 
        charge_dict_overall: dict, dictionary with the charges from different methods
        atomic_dipole_dict_overall: dict, dictionary with the atomic dipoles from different methods
        dipole_info: dict, dictionary with the total dipoles from different methods
    """

    charge_key_1 = "Atom        Charge"
    charge_key_2 = "Center       Charge"
    charge_key_3 = "Final atomic charges:"
    atomic_dipole_key = "Atomic dipole moments (a.u.):"

    dipole_cm5_xyz_key = "X/Y/Z of dipole moment from CM5 charges"
    dipole_adch_xyz_key = "X/Y/Z of dipole moment from the charge (a.u.)"
    dipole_cm5_key = "Total dipole moment from CM5 charges"
    dipole_adch_key = "Total dipole from ADC charges (a.u.)"
    dipole_key = "Total dipole moment from atomic charges:"
    dipole_xyz_key = "X/Y/Z of dipole moment vector:"

    charge_ordering = ["hirshfeld", "vdd", "becke", "adch", "chelpg", "mk", "cm5", "resp", "peoe", "mbis"]
    dipole_order = ["hirshfeld", "vdd", "becke", "hirshfeld" , "hirshfeld"]
    atomic_dipole_order = ['becke', 'adch']

    charge_dict_overall = {}
    atomic_dipole_dict_overall = {}
    dipole_info = {"cm5": {}, "adch": {}, "hirshfeld": {}, "vdd": {}, "becke": {}}

    # iterate over lines of the file 
    with open(charge_out_txt, 'r') as f:
        
        charge_dict_index = 0
        dipole_index = 0
        atomic_dipole_index = 0 
        trigger, trigger2=False, False
        trigger_dipole = False
        for line in f:

            if line == "\n" or len(line)<3:
                if trigger or trigger2:    
                        trigger, trigger2=False, False
                        charge_dict_overall[charge_ordering[charge_dict_index]] = charge_dict
                        charge_dict_index += 1

                elif trigger_dipole:                 
                        trigger_dipole = False
                        atomic_dipole_dict_overall[atomic_dipole_order[atomic_dipole_index]] = atomic_dipole_dict
                        atomic_dipole_index += 1


            if trigger or trigger2:
                if line.split()[0] == "Atom":
                    ind, element = line.split()[1].split("(")
                elif line.strip()[0].isnumeric():        
                    ind, element = line.split()[0].split("(")
                
                value = float(line.split()[-1])
                charge_dict[ind + "_" + element] = value
                
            if trigger_dipole: 
                #print(len(line) < 3)
                if line.strip().startswith("Atom"):
                    #print(line)
                    ind, element = line.split()[1].split("(")
                    dipole = line.split()[5: 8]
                    #print(x, y, z)
                    atomic_dipole_dict[ind + "_" + element] = [float(i) for i in dipole]

            if dipole_cm5_key in line:
                float_dipole_cm5 = float(line.split()[-2])
                dipole_info["cm5"]["mag"] = float_dipole_cm5
            
            if dipole_cm5_xyz_key in line:
                str_nums = line.split()[7:-1]
                float_cm5_xyz = [float(num) for num in str_nums]
                dipole_info["cm5"]["xyz"] = float_cm5_xyz

            if dipole_adch_key in line:
                float_dipole_adch = float(line.split()[-3])
                dipole_info["adch"]["mag"] = float_dipole_adch

            if dipole_adch_xyz_key in line:
                str_nums = line.split()[8:-1]
                float_adch_xyz = [float(num) for num in str_nums]
                dipole_info["adch"]["xyz"] = float_adch_xyz
            
            # logic here needs to be added to 
            if dipole_index < 3:
                if dipole_key in line:
                    float_dipole_temp = float(line.split()[-2])
                    dipole_info[dipole_order[dipole_index]]["mag"] = float_dipole_temp
                if dipole_xyz_key in line:
                    str_nums = line.split()[5:-1]
                    float_diple_xyz_temp = [float(num) for num in str_nums]
                    dipole_info[dipole_order[dipole_index]]["xyz"] = float_diple_xyz_temp
                    dipole_index += 1
                
            if charge_key_1 in line:
                trigger=True
                charge_dict = {}
            
            if charge_key_2 in line:
                trigger2=True
                charge_dict = {}
            
            if charge_key_3 in line:
                trigger2=True
                charge_dict = {}

            if atomic_dipole_key in line:
                trigger_dipole = True
                atomic_dipole_dict = {}

    return charge_dict_overall, atomic_dipole_dict_overall, dipole_info