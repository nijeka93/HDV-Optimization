class VehicleScope:
    def __init__(self, powertrain, size, year, lifetime_kilometers,
                 driving_cycle, target_range, payload, power, country_of_production, country_of_use, powered_hours):
        """
        Initialize a vehicle scope object.

        :param powertrain: [-]; either: BEV, PHEV, FCEV
        :param size: [-]; either: 7.5t, 12t, 18t, 26t, 32t, 40t, 60t
        :param year: [a]; between 2020 and 2050
        :param life_kilometers: [km]; integer
        :param driving_cycle: [-]; either "Long haul", "Regional delivery", "Urban delivery"
        :param target_range: [km]; integer
        :param payload: [kg]; integer
        :param power: # [kW]
        :param country_of_production: # [xy] (2-letter country code)
        :param country_of_use: # [xy] (2-letter country code)
        :param powered_hours: [h]
        """
        self.powertrain = powertrain
        self.size = size
        self.year = year
        self.lifetime_kilometers = lifetime_kilometers
        self.driving_cycle = driving_cycle
        self.target_range = target_range
        self.payload = payload
        self.power = power
        self.country_of_production = country_of_production
        self.country_of_use = country_of_use
        self.powered_hours = powered_hours

class Battery_Scope:
    def __init__(self, target_voltage, target_energy, target_power, number_of_packs, max_volume_per_pack, replaceable_mounting_add, packing_efficiency, cell_to_cell_clearance, initial_battery_replacements):
        """
        Initialize a battery scope object.

        :param target voltage: [V]; integer
        :param target energy: [kWh]; integer
        :param target power: [kW]; integer
        :param number_of_packs: [#]; integer
        :param max_volume_per_pack: [ccm]; integer
        :param replaceable_mounting_add: [%]; float
        :param packing_efficiency: [mm]; float
        :param cell_to_cell_clearance: [mm]; float
        :param initial_battery_replacements: [#/vehicle lifetime]; dummy value which is replaced by calculation later on
        """
        self.target_voltage = target_voltage
        self.target_energy = target_energy
        self.target_power = target_power
        self.number_of_packs = number_of_packs
        self.max_volume_per_pack = max_volume_per_pack
        self.replaceable_mounting_add = replaceable_mounting_add
        self.packing_efficiency = packing_efficiency
        self.cell_to_cell_clearance = cell_to_cell_clearance
        self.initial_battery_replacements = initial_battery_replacements
    

class Pack:
    def __init__(self, target_voltage, casing_material, casing_wall_thickness, cooling_system_mass):
        """
        Initialize a battery scope object.

        :param target voltage: [V]; integer
        :param target material: [-]; string
        :param target casing wall thickness: [mm]; float
        :param cooling_system_mass: [kg]; float
        """
        self.target_voltage = target_voltage
        self.casing_material = casing_material
        self.casing_wall_thickness = casing_wall_thickness
        self.cooling_system_mass = cooling_system_mass

class Cell:
    def __init__(self, model, quantity, length, width, height, mass, failure_rate, chemistry, geometry, voltage, currency,
                 capacity, energy, gravimetric_energy, cycle_life):
        """
        Initialize a cell object.

        :param model: model of the component; string
        :param quantity: quantity of this particular compoent per battery level or grouping [#]; integer
        :param length: length of the component [cm]; float
        :param width: width of the component [cm]; float
        :param height: height of the component [cm]; float
        :param mass: mass of the component [kg]; float
        :param failure_rate: Failure rate of the component [fpmh]; float
        :param chemistry: [-]; # [-]; NMC811, NMC622, NMC532, NMC111, LFP, NCA, LTO
        :param geometry: [-]; "cylindric", "prismatic", "pouch"
        :param voltage: [V]; float
        :param currency: [A]; float
        :param capacity: [Ah]; float
        :param energy: [Wh]; float
        :param gravimetriy_energy: [Wh/kg]; integer
        :param cycle_life: [#]; integer
        """
        self.model = model
        self.quantity = quantity
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass
        self.failure_rate = failure_rate
        self.chemistry = chemistry
        self.geometry = geometry
        self.voltage = voltage
        self.currency = currency
        self.capacity = capacity
        self.energy = energy
        self.gravimetric_energy = gravimetric_energy
        self.cycle_life = cycle_life
        
    def __getitem__(self, key):
        """
        Allow subscriptable access to attributes.
        :param key: The name of the attribute to access (string).
        :raises KeyError: If the key is not valid.
        """
        if not hasattr(self, key):
            raise KeyError(f"{key} is not a valid attribute of StandardComponent.")
        return getattr(self, key)


class StandardComponent:
    def __init__(self, model="default", quantity=1, length=0.0, width=0.0, height=0.0, mass=0.0, failure_rate=0.0001):
        """
        Initialize a NormalBatteryComponent object.
        :param model: model of the component; string
        :param quantity: quantity of this particular compoent per battery level or grouping; mostly integer, sometimes float
        :param length: length of the component [cm]; float
        :param width: width of the component [cm]; float
        :param height: height of the component [cm]; float
        :param mass: mass of the component [kg]; float
        :param failure_rate: Failure rate of the component [fpmh]; float
        """
        self.model = model
        self.quantity = quantity
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass
        self.failure_rate = failure_rate

    def __getitem__(self, key):
        """
        Allow subscriptable access to attributes.
        :param key: The name of the attribute to access (string).
        :raises KeyError: If the key is not valid.
        """
        if not hasattr(self, key):
            raise KeyError(f"{key} is not a valid attribute of StandardComponent.")
        return getattr(self, key)


