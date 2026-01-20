import math
import numpy as np
import sys
from data_manager import get_data_point, list_components, get_component, get_component_volume, get_cell_volume

module_components = ["cell", "module_voltage_sensor", "module_temperature_sensor", "module_pcb", "module_bms", "module_signal_connector"]
pack_components = ["pack_voltage_sensor", "pack_current_sensor", "pack_pcb", "pack_bms", "pack_signal_connector", "pack_power_connector", "pack_busbar", "pack_relay", "pack_fuse"]

def update_number_of_components(cells_per_module: float, modules_in_parallel: int, cells_in_parallel: int):
    """
    Returns a list each for the quantities of module and pack components. Module_components_quantitites is derived per module, whereas pack_components_quantitites
    is derived per pack.
    
    Parameters:
    param:: cells_per_module: float
    param:: modules_in_parallel: int
    param:: cells_in_parallel: int
    
    Returns:
    list[int]: module_component_quantities [#]
    list[int]: pack_component_quantities [#]
    """
    
    module_components_quantities = []
    pack_components_quantities = []
    i = 0
    
    for component in module_components:
        if component == "cell" or component == "module_signal_connector":
            module_components_quantities.append(cells_per_module)
        elif component == "module_bms":
            module_components_quantities.append(1)
        elif component == "module_voltage_sensor":
            try:
                per_string = int(get_data_point(component, "quantity"))
            except Exception:
                per_string = 1
            qty = int(math.ceil(float(cells_in_parallel) / float(modules_in_parallel))) * per_string
            module_components_quantities.append(qty)
        elif component == "module_temperature_sensor":
            try:
                sensors_per_cell = float(get_data_point(component, "quantity"))
            except Exception:
                sensors_per_cell = 0.0
            if sensors_per_cell and sensors_per_cell > 0:
                qty = int(math.ceil(sensors_per_cell * float(cells_per_module) + 2))
                print(f"computed module_temperature_sensor qty = ceil({sensors_per_cell} * {cells_per_module} + 2) = {qty}")
            else:
                qty = int(math.ceil(0.2 * float(cells_per_module) + 2)) # 0.2 temp sensors per cell is appropriate for the 2.8kg prismatic cell
                print(f"no quantity_per_cell provided; using fixed quantity = {qty}")
            module_components_quantities.append(qty)
        else:
            module_components_quantities.append(get_data_point(component, "quantity"))
    print(f"module component list: {module_components}")
    print(f"module component quantities: {module_components_quantities}")

            
    for component in pack_components:
        if component == "pack_voltage_sensor":
            pack_components_quantities.append(modules_in_parallel)
        elif component == "pack_bms":
            pack_components_quantities.append(1)
        else:
            pack_components_quantities.append(get_data_point(component, "quantity"))
    
    return module_components_quantities, pack_components_quantities


def calculate_module(module_components: list[str], modules_in_parallel: int, module_design: int, cell_to_cell_clearance: list[float], cells_per_module: float, cells_in_parallel: int, replaceable_mounting_add: float, packing_efficiency: float, replaceability: list[bool]):
    """
    Returns the mass of the module casing in g. The mass is calculated via the volume of the module casing, derived via the internal volume required (+ some margin)
    Assumptions:
    Wall thickness sealed: 0.3 + 0.5 cm (Casing + Isolation)
    Wall thickness non-sealed: 0.2 + 0.4 cm (Casing + Isolation)
    # Aluminum density = 2.7 g/cm³
    # HDPE density = 0.94 g/cm3
    # Isolation (PE foam) density = 0.2 g/cm³
    
    Parameters:
    module_components (list[str]): list of all the specific module components (e.g., '(cylindrical, small)', etc.)
    modules_in_parallel (int): number of modules in parallel per pack
    cells_in_parallel (int): number of cells in parallel per pack
    module_design (int): optimization variable; integers 0-4 represent certain module designs
    geometry (str): cell type
    cell_to_cell_clearance (list[float]): distance between cells [mm]
    replaceable_mounting_add (float): additional material to install components replaceable instead of permanently fixed (e.g., 1.03; representing a 3% mass penalty)
    packing_efficiency (float): volumentric packing efficiency (e.g., 0,97, meaning that 97% of volume is occupied)
    replaceability (list[bool]): identical list of module_components, but containing bools representing the component's replaceability ('0' = permanent; '1' = replaceable)
    
    Returns:
    int: module_casing_weight [g].
    """
    
    internal_volume = 0
    module_mass = 0
    cell_mass = 0
    module_components_quantities, pack_components_quantities = update_number_of_components(modules_in_parallel=modules_in_parallel, cells_per_module=cells_per_module, cells_in_parallel=cells_in_parallel)
    print(f"module_components_quantities: {module_components_quantities}")

    for index, component in enumerate(module_components):
        if "cell" in component: # fetching cell volume from get_cell_properties
            if module_design < 3 or replaceability[index] == 0: # replaceability == 0 --> not replaceable; module_design <= 2 --> with foam filling --> components not replaceable 
                internal_volume += get_cell_volume(cell_to_cell_clearance) * module_components_quantities[index]
                cell_mass = get_data_point(component, "mass")
                module_mass += cell_mass * module_components_quantities[index]
                print(f"total cell mass per module: {cell_mass} [kg] * {cells_per_module} [#] = {cell_mass * cells_per_module} [kg]")
            else: # is replaceable
                internal_volume += get_cell_volume(cell_to_cell_clearance) * replaceable_mounting_add * module_components_quantities[index]
                cell_mass = get_data_point(component, "mass")
                module_mass += cell_mass * replaceable_mounting_add * module_components_quantities[index]
                print(f"total cell mass per module: {cell_mass} [kg] * {cells_per_module} [#] = {cell_mass * cells_per_module} [kg]")
        else: # fetching component volume from class function
            if module_design < 3 or replaceability[index] == 0:
                internal_volume += get_component_volume(component) * module_components_quantities[index]
                module_mass += get_data_point(component, "mass") * module_components_quantities[index]
                #print(f"updated module mass: {module_mass} [kg]")
            else: # is replaceable
                internal_volume += get_component_volume(component) * replaceable_mounting_add * module_components_quantities[index]
                module_mass += get_data_point(component, "mass") * replaceable_mounting_add * module_components_quantities[index]
                #print(f"updated module mass: {module_mass} [kg]")
        print(f"updated module mass: {module_mass} [kg] = old module_mass + ({get_data_point(component, 'mass')} * {module_components_quantities[index]})")
    internal_volume = internal_volume / packing_efficiency
    print(f"number of cells per module: {module_components_quantities[index]}")
    print(f"module internal volume: {internal_volume} [ccm]")
    
    mounting_volume = internal_volume ** (2/3) * 2 # Assuming top- & bottom-mounting a 1 mm thickness
    print(f"module mounting volume: {mounting_volume} [ccm]")
    
    isolation_volume = (internal_volume ** (1/3) + 0.3) ** 3 - internal_volume # Assuming 3mm isolation
    print(f"module isolation volume: {isolation_volume} [ccm]")
    
    isolation_volume_sealed = (internal_volume ** (1/3) + 0.35) ** 3 - internal_volume # Assuming 3.5mm isolation
    print(f"sealed module isolation volume: {isolation_volume} [ccm]")
    
    casing_volume = (internal_volume ** (1/3) + 0.3 + 0.2) ** 3 - (internal_volume ** (1/3) + 0.3) ** 3 # Assuming 2mm wall thickness
    print(f"module casing volume: {casing_volume} [ccm]")

    casing_volume_sealed = (internal_volume ** (1/3) + 0.35 + 0.2) ** 3 - (internal_volume ** (1/3) + 0.4) ** 3 # Assuming 2mm wall thickness
    print(f"sealed module casing volume: {casing_volume} [ccm]")
    
    if module_design == 0: # Closed Casing (not sealed; HDPE) with foam filling
        module_volume = internal_volume + mounting_volume + isolation_volume
        module_casing_mass = (mounting_volume + isolation_volume) * 0.94 # HDPE density: 0.94 g/cm3
        print(f"module design 0: volume = internal ({internal_volume}) mounting ({mounting_volume}) + isolation ({isolation_volume}) = {module_volume} [ccm]")
    elif module_design == 1: # Open Casing (HDPE) with foam filling
        module_volume = internal_volume + mounting_volume + isolation_volume
        module_casing_mass = (mounting_volume + isolation_volume) * 0.94 * 0.8 # HDPE density: 0.940 g/cm3; # Assuming open design uses 80% of material compared to closed design
        print(f"module design 1: volume = internal ({internal_volume}) mounting ({mounting_volume}) + isolation ({isolation_volume}) * 0,8 = {module_volume} [ccm]")
    elif module_design == 2: # Closed Casing (sealed; aluminium) without filling
        module_volume = internal_volume + mounting_volume + isolation_volume_sealed + casing_volume_sealed
        module_casing_mass = mounting_volume * 2.7 + isolation_volume_sealed * 0.2 + casing_volume_sealed * 2.7 # Aluminium density: 2.7 g/cm3; PE foam density: 0,2 g/cm3; HDPE density: 0.94 g/cm3
        print(f"module design 2: volume = internal ({internal_volume}) mounting ({mounting_volume}) + isolation ({isolation_volume}) + casing ({casing_volume_sealed})= {module_volume} [ccm]")
    elif module_design == 3: # Closed Casing (not sealed; aluminium) without filling
        module_volume = internal_volume + mounting_volume + isolation_volume + casing_volume
        module_casing_mass = mounting_volume * 2.7 + isolation_volume * 0.2 + casing_volume * 2.7 # Aluminium density: 2.7 g/cm3; PE foam density: 0,2 g/cm3; HDPE density: 0.94 g/cm3
        print(f"module design 3: volume = internal ({internal_volume}) mounting ({mounting_volume}) + isolation ({isolation_volume}) + casing ({casing_volume})= {module_volume} [ccm]")
    elif module_design == 4: # Open Casing (aluminium) without filling
        module_volume = internal_volume + mounting_volume + isolation_volume
        module_casing_mass = mounting_volume * 2.7 + isolation_volume * 2.7 * 0.8 # # Aluminium density: 2.7 g/cm3; Assuming open design uses 80% of material compared to closed design;
        print(f"module design 4: volume = internal ({internal_volume}) mounting ({mounting_volume}) + isolation ({isolation_volume}) = {module_volume} [ccm]")
    else:
        raise ValueError("module design not registered")
    module_casing_mass = module_casing_mass / 1000 # g --> kg
    print(f"module casing mass: {module_casing_mass} [kg]")
    
    module_mass += module_casing_mass
    print(f"total module mass: {module_mass} [kg]")
    print(f"total module volume: {module_volume} [ccm]")

    return float(module_casing_mass), int(module_volume), cell_mass, float(module_mass)

def calculate_pack(pack_components: list[str], cells_per_module: float, module_volume: int, module_mass: int, modules_per_pack: int, modules_in_parallel:int, cells_in_parallel:int, replaceable_mounting_add: float, packing_efficiency: float, replaceability: list[bool], cooling_system_volume: float, cooling_system_mass: float):
    """
    Returns the mass of the module casing in g. The mass is calculated via the volume of the module casing, derived via the internal volume required (+ some margin)
    Assumptions:
    Wall thickness: 0.4 + 1.5 cm (Steel + Isolation)
    # Steel density = 7.8 g/cm³
    # Isolation (PE foam) density = 0.2 g/cm³
    
    Parameters:
    pack_components (list[str]): list of all the specific pack components (e.g., 'power connector', etc.)
    cells_per_module (float): number of cells per module
    module_mass (int): mass of the module incl. all module components [kg]
    modules_per_pack (int): number of modules per pack (e.g., 10)
    cells_in_parallel (int): number of cells in parallel per pack
    modules_in_parallel (int): number of modules connected in parallel
    replaceable_mounting_add (float): additional material to install components replaceable instead of permanently fixed (e.g., 1.03; representing a 3% mass penalty)
    packing_efficiency (float): volumentric packing efficiency (e.g., 0,97, meaning that 97% of volume is occupied)
    replaceability (list[bool]): identical list of pack_components, but containing bools representing the component's replaceability ('0' = permanent; '1' = replaceable)
    cooling_system_volume (float): volume of the cooling system [ccm]
    cooling_system_mass (float): mass of the cooling system [kg]
    
    Returns:
    float: pack_mass [kg]
    float: pack_volume [ccm]
    """
    
    internal_volume = 0
    pack_mass = 0
    position = 0
    module_components_quantities, pack_components_quantities = update_number_of_components(modules_in_parallel=modules_per_pack, cells_per_module=cells_per_module, cells_in_parallel=cells_in_parallel)
    print(f"pack_components_quantities: {pack_components_quantities}")

    for index, component in enumerate(pack_components):
        if replaceability[position] == 0: # = not replaceable
            internal_volume += get_component_volume(component) * pack_components_quantities[index]
            pack_mass += get_data_point(component, "mass") * pack_components_quantities[index]
            position += 1
        else: # is replaceable
            internal_volume += get_component_volume(component) * replaceable_mounting_add * pack_components_quantities[index]
            pack_mass += get_data_point(component, "mass") * replaceable_mounting_add * pack_components_quantities[index]
            position += 1
        print(f"updated pack mass: {pack_mass} [kg] = old pack_mass + ({get_data_point(component, 'mass')} * {pack_components_quantities[index]})")

    # adding module volumes and cooling plate volume to the internal pack volume
    print(f"pack components volume (w/o modules or cooling system): {internal_volume} [ccm]")
    print(f"... + modules {module_volume} [ccm] * {modules_per_pack} [#] / {packing_efficiency} [%]")
    print(f"... + cooling system {cooling_system_volume * 1000} [ccm]")
    
    internal_volume += module_volume * modules_per_pack / packing_efficiency
    internal_volume += cooling_system_volume
    print(f"pack components volume (with modules and cooling system): {internal_volume} [ccm]")

    mounting_volume = internal_volume ** (2/3) * 0.5 * 4 # assuming 4 installation levels a 5 mm
    mounting_mass = mounting_volume * 2.7 / 1000 # Aluminium density: 2.7 g/cm3; g --> kg
    
    print(f"... + mounting volume {mounting_volume}") # TODO: implement in input excel
    print(f"mounting mass: {mounting_mass}")

    internal_volume += mounting_volume
    
    # adding module masses and cooling plate volume to the internal pack mass
    pack_mass += module_mass * modules_per_pack
    pack_mass += cooling_system_mass
    pack_mass += mounting_mass
    
    print(f"= pack internal volume: {internal_volume}")
    print(f"number of modules per pack: {modules_per_pack} [#]")
    
    isolation_volume = (internal_volume ** (1/3) + 1.5) ** 3 - internal_volume # Assuming 15mm isolation and mounting
    print(f"pack isolation volume: {isolation_volume} [ccm]")
    
    casing_volume = (internal_volume ** (1/3) + 1.5 + 0.6) ** 3 - (internal_volume ** (1/3) + 1.5) ** 3 # Assuming 5 mm wall thickness
    print(f"pack casing volume (incl. mounting level): {casing_volume + mounting_mass} [ccm]")

    pack_volume = (internal_volume ** (1/3) + 1.5 + 0.6) ** 3 / packing_efficiency # [ccm]
    print(f"pack volume: {pack_volume} [ccm]")
    
    if get_data_point("pack", "casing_material") == "Aluminium":
        pack_casing_mass = isolation_volume * 0.4 + casing_volume * 2.7 # Assuming isolation and mounting material density of 0.4 g/ccm
    elif get_data_point("pack", "casing_material") == "Plastics":
        pack_casing_mass = isolation_volume * 0.4 + casing_volume * 0.9
    elif get_data_point("pack", "casing_material") == "Steel":
        pack_casing_mass = isolation_volume * 0.4 + casing_volume * 7.8
    else:
        raise ValueError("pack casing material not registered")
    
    pack_casing_mass = pack_casing_mass / 1000 # g --> kg   
    print(f"pack enclosure mass: {pack_casing_mass} [kg]") 
    print(f"pack enclosure mass (incl. installation levels): {pack_casing_mass + mounting_mass} [kg]")
    
    pack_mass += pack_casing_mass
     
    return float(pack_mass), int(pack_volume)


# Function to calculate MTBF and reliability
def calc_mtbf_and_reliability(fpmh: float, powered_hours: int):
    if fpmh <= 0:
        print(f"Warning: fpmh is {fpmh}. Setting it to 0.001 to avoid division by zero.")
        fpmh = 0.001
    mtbf = 1000000 / fpmh
    reliability = math.exp(-powered_hours / mtbf)
    return mtbf, reliability


# Step 1: Define the design space
def create_design_space(target_voltage: int, target_energy: int, number_of_packs: int):
    cell_voltage = get_data_point("cell", "voltage")
    cell_energy = get_data_point("cell", "energy")
    mass_per_cell = get_data_point("cell", "mass") # cell replaceability not set yet, hence permanent mass is assumed
    print(f"mass per cell [kg]: {mass_per_cell}")
    
    # Cells per row to achieve target voltage
    cells_in_series = math.ceil(target_voltage / cell_voltage)
    print(f"cells in series = {cells_in_series} = target voltage ({target_voltage}) [V] / cell voltage ({cell_voltage} [V])")
    
    # Cells per column to achieve target capacity
    cells_in_parallel = round(target_energy / cell_energy / cells_in_series * 1000, 0)
    print(f"cells in parallel = {cells_in_parallel} = target energy {target_energy} [kWh] / cell energy {cell_energy} [Wh] / {cells_in_series} * 1000")
        
    # Cells needed in total (per pack)
    cells_per_pack = int(cells_in_series) * int(cells_in_parallel) / number_of_packs
    print(f"cells per pack: {cells_per_pack} = {cells_in_series} * {cells_in_parallel} / {number_of_packs}")
    
    return cells_per_pack, cells_in_series, cells_in_parallel, mass_per_cell

# Step 2: Define the objective functions (to maximize the value retention and battery efficiency)
# First objective: Minimize failure rates (translated into battery swaps)
def objective_value_retention(initial_battery_replacements: int, cells_per_module: int, cells_in_parallel: int, modules_in_parallel: int, modules_per_pack: int, replaceable_mounting_add: float, replaceability_module: list[int], replaceability_pack: list[int], module_mass: int, pack_mass: int, powered_hours: int):
    """
    Objective function to minimize the system's failure rate.

    Parameters:
    geometry (str): contains the cell type, e.g., 'cell (prismatic small)'
    chemistry (str): chemistry of the cell (e.g., 'NMC', etc.)
    cell_mass_share (float): the mass share of cells to the overall pack (e.g., 0.68)
    cells_per_module (float): number of cell per module (e.g., 100 #)
    cells_in_parallel (int): number of cells in parallel per pack
    modules_per_pack (int): number of modules per pack (e.g., 10 #)
    replaceable_mounting_add (float): additional material to install components replaceable instead of permanently fixed (e.g., 1.03; representing 1 3% mass penalty )
    replaceability_module (list[int]): list of module_components containing bools representing the component's replaceability ('0' = permanent; '1' = replaceable)
    replaceability_pack (list[int]): list of pack_components containing bools representing the component's replaceability ('0' = permanent; '1' = replaceable)
    powered_hours (int): the number of hours the vehicle is powered during its whole lifetime
    lifetime_kilometers (int): the number of kilometers the vehicle is driven during its whole lifetime
    target_capacity (int): the capacity each traction battery pack is supposed to have (e.g., 125 kWh)

    Returns:
    battery_replacements (float): the amount battery equivalents replaced during the hdv use phase
    """

    # --- Module components ---
    module_component_mtbfs = []
    pack_component_mtbfs = []
    replacement_masses_per_pack = []
    replacements_per_pack = []
    module_components_quantities, pack_components_quantities = update_number_of_components(modules_in_parallel=modules_in_parallel, cells_per_module=cells_per_module, cells_in_parallel=cells_in_parallel)
    # module_components = ["cell", "module_voltage_sensor", "module_temperature_sensor", "module_pcb" "module_bms", "module_signal_connector"]
    # pack_component = [, "pack_voltage_sensor", "pack_current_sensor", "pack_pcb", "pack_bms", "pack_signal_connector", "pack_power_connector", "pack_busbar", "pack_relay", "pack_fuse"]

    # 1) Calculate Mean Time Between Failures (MTBF; in hours) for each module component
    for index, component in enumerate(module_components):
                
        # Calculate MTBF and replacements for each module component
        component_fpmh = get_data_point(component, "failure_rate") * module_components_quantities[index]
        component_mtbf, component_reliability = calc_mtbf_and_reliability(fpmh=component_fpmh, powered_hours=powered_hours)
        print(f"{component} mtbf: {component_mtbf}")
        module_component_mtbfs.append(component_mtbf)
        # module_component_reliabilities.append(component_reliability)
        
    print(f"module mtbfs: {module_component_mtbfs}")
    
    # 2) Identify the module component type causing to replace the whole module most often -> will determine module replacement timing
    module_replacer_mtbf = sys.maxsize # just a placeholder
    for i, mtbf in enumerate(module_component_mtbfs):
    # Set correct cell mtbf by deriving if component aging or defect is dominant
        if mtbf >= (powered_hours / (initial_battery_replacements + 0.001)): # + 0.001 to avoid division by zero
            component_mtbf = initial_battery_replacements
            print(f"Initial MTBF based on battery aging: {powered_hours / (initial_battery_replacements + 0.001)}")
        if not replaceability_module[i] and mtbf < module_replacer_mtbf:
            module_replacer_mtbf = mtbf
            print(f"Lowest MTBF of non-replaceable component: {module_components[i]} with {module_replacer_mtbf}")
                
    # 3) Calculate replacement weights for each module component
    for index, component in enumerate(module_components):    
        # Calculate replaced mass per pack and vehicle lifetime (module level components)
        # a) Set correct component weight
        if replaceability_module[index] == 1:
            mass = get_data_point(component, "mass") * replaceable_mounting_add
        else:
            mass = get_data_point(component, "mass")
            
        # c) Add replacement weights per pack accordingly:
        # Case 1: component replaceable and more prone to defects than worst permanently-fixed component -> weight of component is replaced
        if replaceability_module[index] == 1 and module_component_mtbfs[index] <= module_replacer_mtbf:
            replacement_masses_per_pack.append(mass * modules_per_pack)
            print(f"module component {component} is replaceable -> adding {mass * modules_per_pack} to replacement mass per pack and vehicle lifetime")
        # Case 2: component is less prone to defects than worst permanently-fixed component -> component is replaced before damage by other defect and no additional replacement occurs
        elif module_component_mtbfs[index] > module_replacer_mtbf:
            replacement_masses_per_pack.append(0)
            print(f"module component {component} is replaced before damage by other defect -> adding 0 to replacement mass per pack and vehicle lifetime")
        # Case 3: component not replaceable but damaged most often; module replaceable -> weight of module is replaced
        elif replaceability_module[index] == 0 and module_component_mtbfs[index] <= module_replacer_mtbf and replaceability_pack[0] == 1:
            replacement_masses_per_pack.append(module_mass)
            print(f"module component {component} is NOT replaceable but most prone to defect -> adding {module_mass} to replacement mass per pack and vehicle lifetime")
        # Case 4: component and module not replaceable; component type damaged most often -> weight of pack is replaced
        elif replaceability_module[index] == 0 and module_component_mtbfs[index] <= module_replacer_mtbf and replaceability_pack[0] == 0:
            replacement_masses_per_pack.append(pack_mass)
            print(f"module component {component} (most prone to defect) and module are NOT replaceable -> adding {pack_mass} to replacement mass per pack and vehicle lifetime")
        else:
            raise Exception(f"Battery module component replacement weights could not be calculated! Error occured for component: {component} at index: {index}")
    
    # 4) Calculate Mean Time Between Failures (MTBF; in hours), replacement weights for each pack component
    component_index = 0
    for index, component in enumerate(pack_components):
        
        # Calculate MTBF and replacements for each pack component
        pack_component_fpmh = get_data_point(component, "failure_rate") * pack_components_quantities[index] # Multiply failure rate by amount to get chance of one failing
        pack_component_mtbf, component_reliability = calc_mtbf_and_reliability(fpmh=pack_component_fpmh, powered_hours=powered_hours) 
        pack_component_mtbfs.append(pack_component_mtbf)
        # module_component_reliabilities.append(component_reliability)
        
        # Calculate replaced mass per pack and vehicle lifetime (module level components)
        # a) Set correct component weight
        if replaceability_pack[component_index] == 1:
            mass = get_data_point(component, "mass") + 0.02
        else:    
            mass = get_data_point(component, "mass")
            
        # b) Add replacement weights per pack accordingly:
        # Case 1: component replaceable -> weight of component is replaced
        if replaceability_pack[component_index] == 1:
            replacement_masses_per_pack.append(mass)
            print(f"pack component {component} is replaceable -> adding {mass} to replacement mass per pack and vehicle lifetime")
        # Case 2: component not replaceable -> weight of module is replaced
        elif replaceability_pack[component_index] == 0:
            replacement_masses_per_pack.append(pack_mass)
            print(f"pack component {component} is NOT replaceable -> adding {pack_mass} to replacement mass per pack and vehicle lifetime")
        else:
            raise Exception(f"Battery pack component replacement weight could not be calculated! Error occured for component: {component} at index: {index}")   
        component_index += 1
    
    # 3: Calculated the pack equivalent replacements (which is assumed to be equal to the battery equivalent replacements):
    interim_replacements = 0
    for mtbf in (module_component_mtbfs + pack_component_mtbfs):
        replacements_per_pack.append(powered_hours / mtbf)
    print(f"replacements per pack: {replacements_per_pack}")
    
    for mass, replacements in zip(replacement_masses_per_pack, replacements_per_pack):
        interim_replacements += mass * replacements
        print(f"mass: {mass}, replacements: {float(replacements)}")
    battery_replacements = interim_replacements / pack_mass * get_data_point("battery_scope", "number_of_packs")
    
    print(f"battery replacements per hdv lifetime: {float(battery_replacements)}")        
    
    return float(battery_replacements)


# Second objective: Maximize Battery Efficiency (translated into gravimetric energy density)
def objective_efficiency_increase(module_design: int, cell_to_cell_clearance: int, replaceability_module: dict, replaceability_pack: dict, cells_per_module: float, cells_per_pack: int, modules_per_pack: int, modules_in_parallel: int, cells_in_parallel: int, replaceable_mounting_add: float, packing_efficiency: float):
    """
    Objective function to compute the battery cell energy density and cell mass share, 
    considering efficiency drawbacks due to the fixation methods.

    Parameters:
    chemistry (str): Type of active material (i.e., 'NMC', 'NCA', 'LFP', 'LTO')
    geometry (str): Type of the battery (e.g., 'cell (cylindrical, small)', 'cell (prismatic, mid)', etc.)
    component_removability (dict): Dctionary of all battery components, stating which are permanent and which are removable
    cells_per_module (float): Number of cells per module (can be real number, if the total number of cells is not dividable by the selected number of modules).
    cells_in_parallel (int): number of cells in paralle per pack
    cells_per_pack (int): Total number of cells.
    modules_per_pack (int): Total number of modules per pack.
    replaceable_mounting_add (float): additional material to install components replaceable instead of permanently fixed (e.g., 1.03; representing 1 3% mass penalty )
    packing_efficiency (float): volumentric packing efficiency (e.g., 0,97, meaning that 97% of volume is occupied)
    
    Returns:
    dict: battery_efficiencies{"cell mass share": float, "total cells mass": int, "module mass": int, "module volume": int, "pack mass": int, "pack volume": int}
    """
        
    # Adjust cell mass share based on the efficiency drawbacks
    battery_efficiencies = {}
    initial_cell_mass_share = 0.6
    module_casing_mass, module_volume, cell_mass, module_mass = calculate_module(
        module_components=module_components,
        modules_in_parallel=modules_in_parallel,
        module_design=module_design,
        cell_to_cell_clearance=cell_to_cell_clearance,
        cells_per_module=cells_per_module,
        cells_in_parallel=cells_in_parallel,
        replaceable_mounting_add=replaceable_mounting_add,
        packing_efficiency=packing_efficiency,
        replaceability=replaceability_module
        )
    
    cooling_system_mass = get_data_point("pack", "cooling_system_mass")
    print(f"cooling system mass: {cooling_system_mass}")
    cooling_system_volume = cooling_system_mass / 2.7 # assuming cooling system is made of aluminium (with a density of 2.7 g/ccm)
        
    pack_mass, pack_volume = calculate_pack(
        pack_components=pack_components,
        cells_per_module=cells_per_module,
        module_volume=module_volume,
        module_mass=module_mass,
        modules_per_pack=modules_per_pack,
        modules_in_parallel=modules_in_parallel,
        cells_in_parallel=cells_in_parallel,
        replaceable_mounting_add=replaceable_mounting_add,
        packing_efficiency=packing_efficiency,
        replaceability=replaceability_pack,
        cooling_system_mass=cooling_system_mass,
        cooling_system_volume=cooling_system_volume
        )
    # print(f"pack mass: {pack_mass}")
    # print(f"pack_volume: {pack_volume}")
    cells_per_pack_mass = cell_mass * cells_per_pack
    adjusted_cell_mass_share = (cell_mass * cells_per_pack) / (pack_mass)
    if adjusted_cell_mass_share < 0.5 or adjusted_cell_mass_share > 0.9:
        adjusted_cell_mass_share = initial_cell_mass_share
        print(f"Adjusted cell_mass_share was infeasible, corrected to {adjusted_cell_mass_share}")
    else:
        print(f"Adjusted cell_mass_share = {adjusted_cell_mass_share} [-] = {cell_mass} * {cells_per_pack} [kg] total cells mass per pack / {pack_mass} [kg] pack mass")
        print(f"Original cell_mass_share of {initial_cell_mass_share} adjusted to {adjusted_cell_mass_share}")
    print(f"Total cells mass = {cells_per_pack_mass} [kg] = cell per pack ({cells_per_pack}) [#] * cell mass ({cell_mass}) [kg]")
    
    battery_efficiencies["cell mass share"] = float(adjusted_cell_mass_share) # [%]
    battery_efficiencies["cells per pack mass"] = int(cells_per_pack_mass) # [kg]
    battery_efficiencies["module mass"] = int(module_mass) # [kg]
    battery_efficiencies["module volume"] = int(module_volume) # [ccm]
    battery_efficiencies["pack mass"] = int(pack_mass) # [kg]
    battery_efficiencies["pack volume"] = int(pack_volume) # [ccm]

    # Return the calculated values
    return battery_efficiencies