from classes import VehicleScope, Battery_Scope, Pack, Cell, StandardComponent

# A global dictionary to store battery component instances
components = {}

def add_component(name, component_data):
    """
    Add a component's data to the global dictionary.

    :param name: Name or identifier for the component.
    :param component_data: Must be an instance of a known class.
    """
    if not isinstance(component_data, (VehicleScope, Battery_Scope, Pack, Cell, StandardComponent)):
        raise ValueError("Only registered instances can be added.")
    components[name] = component_data

def get_component(name):
    """
    Retrieve a component by its name.

    :param name: Name or identifier of the component.
    :return: The component instance, or None if not found.
    """
    if name not in components:
        raise ValueError(f"Component '{name}' not found.")
    return components[name]

def get_component_volume(name):
    """
    Retrieve a component's volume.

    :param name: Name or identifier of the component.
    :return: The volume of the component in cm³.
    :raises ValueError: If the component is not found or lacks volume attributes.
    """
    component = get_component(name)
    
    # Ensure the component has the required attributes for volume calculation
    if not all(hasattr(component, attr) for attr in ["length", "width", "height"]):
        raise ValueError(f"Component '{name}' lacks attributes required for volume calculation (length, width, height).")
    
    return component.length * component.width * component.height

def get_cell_volume(cell_to_cell_clearance: float):
    """
    Calculate the volume of a cell with additional clearance.

    :param cell_to_cell_clearance: Clearance to add to cell dimensions (in mm).
    :return: The calculated volume of the cell in cm³.
    """
    if "cell" not in components:
        raise ValueError("Component 'cell' not found.")
    
    cell = components["cell"]
    if not all(hasattr(cell, attr) for attr in ["length", "width", "height"]):
        raise ValueError("Component 'cell' lacks attributes required for volume calculation (length, width, height).")
    
    # Adjust dimensions by adding half of the clearance to each side
    return (cell.length + cell_to_cell_clearance / 10 / 2) * \
           (cell.width + cell_to_cell_clearance / 10 / 2) * \
           cell.height

def get_data_point(name, variable):
    """
    Get a specific variable of a component.

    :param name: Name or identifier of the component.
    :param variable: The name of the variable to fetch (string).
    :raises ValueError: If the component or variable doesn't exist.
    """
    component = get_component(name)
    
    if not hasattr(component, variable):
        raise ValueError(f"Variable '{variable}' does not exist in component '{name}'.")
    
    return getattr(component, variable)

def list_components():
    """
    List all stored components.
    """
    return list(components.keys())

def update_component_variable(name, variable, value):
    """
    Update a specific variable of a component.

    :param name: Name or identifier of the component.
    :param variable: The name of the variable to update (string).
    :param value: The new value to set for the variable.
    :raises ValueError: If the component or variable doesn't exist.
    """
    component = get_component(name)
    
    if not hasattr(component, variable):
        raise ValueError(f"Variable '{variable}' does not exist in component '{name}'.")

    setattr(component, variable, value)
    print(f"Updated '{variable}' of '{name}' to {value}.")