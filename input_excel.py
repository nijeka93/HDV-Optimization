import pandas as pd
import numpy as np
from carculator_truck import *
from classes import VehicleScope, Battery_Scope, Pack, Cell, StandardComponent
from data_manager import add_component, get_component, update_component_variable

def calc_lifetime_hours(driving_cycle: str, size: str, lifetime_kilometers: int) -> float: # bei allen functionen einfügen
    driving_cycle = get_driving_cycle(size=size, name=driving_cycle)
    counter = 0
    avg_speed = 0
    for speed in driving_cycle:
        if speed != np.nan:
            counter += 1
            avg_speed += speed[0]
    avg_speed = avg_speed / counter
    print(f"avg speed: {avg_speed}")
    lifetime_hours = lifetime_kilometers / avg_speed
    print(f"lifetime hours: {lifetime_hours}")
    
    return lifetime_hours

def load_parameters_from_excel(file_path):
    # Load the Excel file
    df = pd.read_excel(file_path, sheet_name="Input Optimization")

    # Parse the necessary parameters from the specified columns
    parameters_df = df.iloc[3:, [3, 5]]  # Columns "D" and "F", starting at row 5
    parameters_df.columns = ['Parameter', 'Value']

    # Convert the dataframe into a dictionary
    parsed_parameters = parameters_df.set_index("Parameter")["Value"].dropna().to_dict()
    
    
    powered_hours = calc_lifetime_hours(
        driving_cycle=parsed_parameters.get("Driving Cycle", ""),
        size=parsed_parameters.get("Vehicle Size", ""),
        lifetime_kilometers=int(parsed_parameters.get("Lifetime Milage", 0)))
    print(f"powered hours: {powered_hours}")

    vehicle_scope = VehicleScope(
        powertrain = parsed_parameters.get("Powertrain", ""), # [-]; either: BEV, PHEV, FCEV
        size = parsed_parameters.get("Vehicle Size", ""), # [-]; either: 7.5t, 12t, 18t, 26t, 32t, 40t, 60t
        year = int(parsed_parameters.get("Production Year", 0)), # [a]; between 2020 and 2050
        lifetime_kilometers = int(parsed_parameters.get("Lifetime Milage", 0)), # [km]; integer
        driving_cycle = parsed_parameters.get("Driving Cycle", ""), # [-]; either Long haul, Regional delivery, Urban delivery
        target_range = int(parsed_parameters.get("Target Range", 0)), # [km]; integer
        payload = int(parsed_parameters.get("Avg. Payload", 0)), # [kg]; integer
        power = int(parsed_parameters.get("System Target Power", 0)), # [kW]
        country_of_production = parsed_parameters.get("Electricity Mix Production", 0), # [xy] (2-letter country code)
        country_of_use = parsed_parameters.get("Electricity Mix Usage", 0), # [xy] (2-letter country code)
        powered_hours = powered_hours
    )
    add_component("vehicle_scope", vehicle_scope)
    
    # Extract cell-to-cell clearance values into a list
    cell_to_cell_clearance = []
    for i in range(0, 5):  # Opt0 to Opt4
        clearance_key = f"Opt{i}: Cell-to-Cell Clearance"
        if clearance_key in parsed_parameters:
            try:
                clearance_value = float(parsed_parameters[clearance_key])
                cell_to_cell_clearance.append(clearance_value)
            except ValueError:
                print(f"Warning: Invalid float value for {clearance_key}. Skipping.")
    print(f"cell-to-cell clearances: {cell_to_cell_clearance}")
    battery_scope = Battery_Scope(
        target_voltage = int(parsed_parameters.get("System Target Voltage", 0)), # [V]; integer
        target_energy = int(parsed_parameters.get("System Target Energy", 0)), # [kWh]; integer
        target_power = int(parsed_parameters.get("System Target Power", 0)), # [kW]; integer
        number_of_packs = int(parsed_parameters.get("Number of Packs", 0)), # [#]; integer
        max_volume_per_pack = int(parsed_parameters.get("Available Volume per Pack", 0)), # [ccm]; integer
        replaceable_mounting_add = float(parsed_parameters.get("Replaceable Mounting: Added Mass", 0)), # [%]
        packing_efficiency= float(parsed_parameters.get("Volumentric Packing Efficiency", 0)), # [%]
        cell_to_cell_clearance = cell_to_cell_clearance,
        initial_battery_replacements = 0, # [#/vehicle lifetime]; dummy value which is replaced by calculation later on
    )
    add_component("battery_scope", battery_scope)
    
    pack = Pack(
        target_voltage = int(parsed_parameters.get("Pack Target Voltage", 0)), # [#]; integer
        casing_material = parsed_parameters.get("Pack Casing (Main) Material", ""), # [-]; either: Steel, Aluminium, , Sttel/Aluminium, Aluminium/Plastics
        casing_wall_thickness = float(parsed_parameters.get("Pack Casing Wall Thickness", 0)), # [mm]; float
        cooling_system_mass = int(parsed_parameters.get("Pack Cooling System Mass", 0)), # [kg]; float        
    )
    add_component("pack", pack)
    
    pack_bms = StandardComponent(
        model = parsed_parameters.get("Pack BMS Model", 0), # [#]
        quantity = 1,
        length = float(parsed_parameters.get("Pack BMS Length", 0)), # [cm]; float
        width = float(parsed_parameters.get("Pack BMS Width", 0)), # [cm]; float
        height = float(parsed_parameters.get("Pack BMS Height", 0)), # [cm]; float
        mass = float(parsed_parameters.get("Pack BMS Mass", 0)), # [kg]; float
        failure_rate = 0.1500, # [fpmh]; float
    )
    add_component("pack_bms", pack_bms)
        
    pack_pcb = StandardComponent(
        model = "default", # [#]
        quantity = 1,
        length = 15, # [cm]; float
        width = 10, # [cm]; float
        height = 0.2, # [cm]; float
        mass = 0.05, # [kg]; float
        failure_rate = 0.2786, # [fpmh]; float
    )
    add_component("pack_pcb", pack_pcb)

    pack_fuse = StandardComponent(
        model = parsed_parameters.get("Pack Fuse Model", 0), # [#]
        quantity = int(parsed_parameters.get("Number of Fuses per Pack", 0)), # [#]; integer
        length = float(parsed_parameters.get("Pack Fuse Length", 0)), # [cm]; float
        width = float(parsed_parameters.get("Pack Fuse Width", 0)), # [cm]; float
        height = float(parsed_parameters.get("Pack Fuse Height", 0)), # [cm]; float
        mass = float(parsed_parameters.get("Pack Fuse Mass", 0)), # [kg]; float
        failure_rate = 0.0010, # [fpmh]; float
    )
    add_component("pack_fuse", pack_fuse)
    
    pack_relay = StandardComponent(
        model = parsed_parameters.get("Pack Relay Model", 0), # [#]
        quantity = int(parsed_parameters.get("Number of Relays per Pack", 0)), # [#]; integer
        length = float(parsed_parameters.get("Pack Relay Length", 0)), # [cm]; float
        width = float(parsed_parameters.get("Pack Relay Width", 0)), # [cm]; float
        height = float(parsed_parameters.get("Pack Relay Height", 0)), # [cm]; float
        mass = float(parsed_parameters.get("Pack Relay Weight", 0)), # [kg]; float
        failure_rate = 0.3300, # [fpmh]; float
    )
    add_component("pack_relay", pack_relay)

    pack_signal_connector = StandardComponent(
        model = "default", # [#]
        quantity = 1,
        length = 3, # [cm]; float
        width = 1.5, # [cm]; float
        height = 0.5, # [cm]; float
        mass = 0.01, # [kg]; float
        failure_rate = 0.4389, # [fpmh]; float        
    )
    add_component("pack_signal_connector", pack_signal_connector)
    
    pack_power_connector = StandardComponent(
        model = "default", # [#]
        quantity = 1,
        length = 2, # [cm]; float
        width = 2, # [cm]; float
        height = 3, # [cm]; float
        mass = 0.03, # [kg]; float
        failure_rate = 0.0533, # [fpmh]; float        
    )
    add_component("pack_power_connector", pack_power_connector)
    
    pack_busbar = StandardComponent(
        model = "default", # [#]
        quantity = 1,
        length = 1, # [cm]; float
        width = 2, # [cm]; float
        height = 2, # [cm]; float
        mass = 0.03, # [kg]; float
        failure_rate = 0.0533, # [fpmh]; float        
    )
    add_component("pack_busbar", pack_busbar)
      
    pack_current_sensor = StandardComponent(
        model = parsed_parameters.get("Pack Current Sensor Model", 0), # [#]
        quantity = int(parsed_parameters.get("Number of Current Sensors per Pack", 0)), # [#]; integer
        length = float(parsed_parameters.get("Pack Current Sensor Length", 0)), # [cm]; float
        width = float(parsed_parameters.get("Pack Current Sensor Width", 0)), # [cm]; float
        height = float(parsed_parameters.get("Pack Current Sensor Height", 0)), # [cm]; float
        mass = float(parsed_parameters.get("Pack Current Sensor Weight", 0)), # [kg]; float
        failure_rate = 0.001, # [fpmh]; float         
    )
    add_component("pack_current_sensor", pack_current_sensor)
    
    pack_voltage_sensor = StandardComponent(
        model = parsed_parameters.get("Pack Voltage Sensor Model", 0), # [#]
        quantity = int(parsed_parameters.get("Number of Voltage Sensors per Pack", 0)), # [#]; integer
        length = float(parsed_parameters.get("Pack Voltage Sensor Length", 0)), # [cm]; float
        width = float(parsed_parameters.get("Pack Voltage Sensor Width", 0)), # [cm]; float
        height = float(parsed_parameters.get("Pack Voltage Sensor Height", 0)), # [cm]; float
        mass = float(parsed_parameters.get("Pack Voltage Sensor Weight", 0)), # [kg]; float
        failure_rate = 0.1613, # [fpmh]; float 
    )
    add_component("pack_voltage_sensor", pack_voltage_sensor)
    
    module_bms = StandardComponent(
        model = parsed_parameters.get("Module BMS Model", 0), # [#]
        quantity = 1,
        length = float(parsed_parameters.get("Module BMS Length", 0)), # [cm]; float
        width = float(parsed_parameters.get("Module BMS Width", 0)), # [cm]; float
        height = float(parsed_parameters.get("Module BMS Height", 0)), # [cm]; float
        mass = float(parsed_parameters.get("Module BMS Weight", 0)), # [kg]; float
        failure_rate = 0.0520, # [fpmh]; float          
    )
    add_component("module_bms", module_bms)
    
    module_pcb = StandardComponent(
        model = "default", # [#]
        quantity = 1,
        length = 15, # [cm]; float
        width = 10, # [cm]; float
        height = 0.2, # [cm]; float
        mass = 0.05, # [kg]; float
        failure_rate = 0.2786, # [fpmh]; float        
    )
    add_component("module_pcb", module_pcb)
    
    module_signal_connector = StandardComponent(
        model = "default", # [#]
        quantity = 1,
        length = 3, # [cm]; float
        width = 1.5, # [cm]; float
        height = 0.5, # [cm]; float
        mass = 0.01, # [kg]; float
        failure_rate = 0.4389, # [fpmh]; float ToDo: ins excel übertragen
    )
    add_component("module_signal_connector", module_signal_connector)
    
    module_voltage_sensor = StandardComponent(
        model = parsed_parameters.get("Module Voltage Sensor Model", 0), # [#]
        quantity = int(parsed_parameters.get("Number of Voltage Sensors per Grouping", 0)), # [#]; integer
        length = float(parsed_parameters.get("Module Voltage Sensor Length", 0)), # [cm]; float
        width = float(parsed_parameters.get("Module Voltage Sensor Width", 0)), # [cm]; float
        height = float(parsed_parameters.get("Module Voltage Sensor Height", 0)), # [cm]; float
        mass = float(parsed_parameters.get("Module Voltage Sensor Weight", 0)), # [kg]; float
        failure_rate = 0.1613, # [fpmh]; float 
    )
    add_component("module_voltage_sensor", module_voltage_sensor)
    
    module_temperature_sensor = StandardComponent(
        model = parsed_parameters.get("Module Temperature Sensor Model", 0), # [#]
        quantity = int(parsed_parameters.get("Number of Temperature Sensors per Module", 0)), # [#]; integer
        length = float(parsed_parameters.get("Module Temperature Sensor Length", 0)), # [cm]; float
        width = float(parsed_parameters.get("Module Temperature Sensor Width", 0)), # [cm]; float
        height = float(parsed_parameters.get("Module Temperature Sensor Height", 0)), # [cm]; float
        mass = float(parsed_parameters.get("Module Temperature Sensor Weight", 0)), # [kg]; float
        failure_rate = 0.0645, # [fpmh]; float 
    )
    add_component("module_temperature_sensor", module_temperature_sensor)
    
    cell_failure_rate = 0.01
    if parsed_parameters.get("Cell Chemistry", "") == "high": # = 10 failures per billion hours (FIT) = 0,01 failures per million hours (fpmh)
        cell_failure_rate = 0.01
    elif parsed_parameters.get("Cell Chemistry", "") == "mid": # 5 failures per billion hours (FIT) = 0,005 failures per million hours (fpmh)
        cell_failure_rate = 0.005
    elif parsed_parameters.get("Cell Chemistry", "") == "low": # 1 failures per billion hours (FIT) = 0,001 failures per million hours (fpmh)
        cell_failure_rate = 0.001

    cell = Cell(
        model = parsed_parameters.get("Cell Model", ""), # [#]
        quantity = 1, # [#]; TODO: calc before initialization how many cells per pack are required
        chemistry = parsed_parameters.get("Cell Chemistry", ""),  # [-]; NMC811, NMC622, NMC532, NMC111, LFP, NCA, LTO
        geometry = parsed_parameters.get("Cell Geometry", ""), # [#]
        length = int(parsed_parameters.get("Cell Length", 0)), # [cm]
        width = int(parsed_parameters.get("Cell Width", 0)), # [cm]
        height = int(parsed_parameters.get("Cell Height", 0)), # [cm]
        mass = float(parsed_parameters.get("Cell Mass", 0)), # [kg]
        voltage = float(parsed_parameters.get("Cell Nominal Voltage", 0)), # [V]
        currency = float(parsed_parameters.get("Cell Nominal Currency", 0)), # [A]
        capacity = float(parsed_parameters.get("Cell Nominal Capacity", 0)), # [Ah]
        energy = float(parsed_parameters.get("Cell Nominal Energy", 0)), # [Wh] 
        gravimetric_energy = int(parsed_parameters.get("Cell Gravimetric Energy", 0)), # [Wh/kg]
        cycle_life = int(parsed_parameters.get("Cell Nominal Cycle Life", 0)), # [#]
        failure_rate = cell_failure_rate, # [fpmh]; float
    )
    add_component("cell", cell)
    
    energy_storage_carculator = {
        "electric": {
            (parsed_parameters.get("Powertrain", ""), parsed_parameters.get("Vehicle Size", ""), int(parsed_parameters.get("Production Year", 0))): parsed_parameters.get("Cell Chemistry", "")
            },
        "origin": parsed_parameters.get("Electricity Mix Production", 0)  # Country code for battery origin
    }
    
    payload_carculator = {
        (parsed_parameters.get("Powertrain", ""), parsed_parameters.get("Vehicle Size", ""), int(parsed_parameters.get("Production Year", 0))): int(parsed_parameters.get("Avg. Payload", 0))
        }
    print(f"payload: {payload_carculator}")
    
    power_carculator = {
        (parsed_parameters.get("Powertrain", ""), parsed_parameters.get("Vehicle Size", ""), int(parsed_parameters.get("Production Year", 0))): int(parsed_parameters.get("System Target Power", 0))
        }
    print(f"power: {power_carculator}")
    
    carculator_scope = {
        "powertrain": [parsed_parameters.get("Powertrain", "")], # [-]; either: BEV, PHEV, FCEV
        "size": [parsed_parameters.get("Vehicle Size", "")], # [-]; either: 7.5t, 12t, 18t, 26t, 32t, 40t, 60t
        "year": [int(parsed_parameters.get("Production Year", 0))], # [a]; between 2020 and 2050
    }
    
    # Load the default vehicle parameters from carculator
    tip = TruckInputParameters()
    tip.static()
    
    # Fill the xarray with the chosen input parameters
    dcts, array = fill_xarray_from_input_parameters(tip, scope=carculator_scope)
    
    lifetime_kilometers = int(parsed_parameters.get("Lifetime Milage", 0))
    array.loc[dict(parameter="lifetime kilometers")] = lifetime_kilometers
    carculator_scope["lifetime kilometers"] = lifetime_kilometers
    
    # Initialize TruckModel based on user inputs
    print(f"driving cycle: {parsed_parameters.get('Driving Cycle', '')} [-]")
    print(f"target range: {parsed_parameters.get('Target Range', 0)} [km]")
    print(f"country: {parsed_parameters.get('Electricity Mix Use', 0)} [xy]")
    print(f"payload: {parsed_parameters.get('Avg. Payload', 0)} [kg]")
    print(f"power: {parsed_parameters.get('System Target Power', 0)} [kW]")
    print(f"energy storage: {energy_storage_carculator} [-]")
    
    tm = TruckModel(
        array,
        cycle = parsed_parameters.get("Driving Cycle", ""), # [-]; either Long haul, Regional delivery, Urban delivery
        target_range = int(parsed_parameters.get("Target Range", 0)), # [km]; integer
        country = parsed_parameters.get("Electricity Mix Use", 0), # [xy] (2-letter country code)
        payload = payload_carculator, # [kg]; integer in dict
        power = power_carculator, # [kW]; integer in dict
        energy_storage = energy_storage_carculator
    )
    
    print("Initial TruckModel is defined")
    
    initial_battery_replacements = tm.array.sel(
        powertrain=parsed_parameters.get("Powertrain", ""),
        year=int(parsed_parameters.get("Production Year", 0)),
        parameter='battery lifetime replacements',
        value=0)
    print(f"initial battery replacements: {initial_battery_replacements.values[0]} [#]")
    
    # Set and return the truck model
    tm.set_all()
    
    # Fetch the optimization target and impact thresholds set in the excel
    scope = parsed_parameters.get("Scope")
    target_function = parsed_parameters.get("Target Function")
    impact_thresholds = {
        "climate change": parsed_parameters.get("climate change"),
        "ozone depletion": parsed_parameters.get("ozone depletion"),
        "eutrophication: marine": parsed_parameters.get("eutrophication: marine"),
        "eutrophication: freshwater": parsed_parameters.get("eutrophication: freshwater"),
        "acidification: terrestrial": parsed_parameters.get("acidification: terrestrial"),
        "land use": parsed_parameters.get("land use"),
        "water use": parsed_parameters.get("water use"),
        "photochemical oxidant formation: human health": parsed_parameters.get("photochemical oxidant formation: human health"),
        "human toxicity: carcinogenic": parsed_parameters.get("human toxicity: carcinogenic"),
        "human toxicity: non-carcinogenic": parsed_parameters.get("human toxicity: non-carcinogenic"),
        "ecotoxicity: freshwater": parsed_parameters.get("ecotoxicity: freshwater"),
        "ionising radiation": parsed_parameters.get("ionising radiation"),
        "energy resources depletion: non-renewable": parsed_parameters.get("energy resources depletion: non-renewable"),
        "material resources: metals/minerals": parsed_parameters.get("material resources: metals/minerals"),
    }
    if target_function != "Total Environmental Impact (distance-to-target)":
        num_objectives = 1
    else:
        num_objectives = len(impact_thresholds)

    return vehicle_scope, battery_scope, carculator_scope, energy_storage_carculator, scope, target_function, num_objectives, impact_thresholds