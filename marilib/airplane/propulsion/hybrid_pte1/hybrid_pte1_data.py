#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:22:21 2019

@author: DRUOT Thierry : original Scilab implementation
         PETEILH Nicolas : portage to Python
"""

#-------------------------------------------------------------------------


class PowerElectricChain(object):
    """
    Electric chain data
    """

    def __init__(self, mto=None,
                 mcn=None,
                 mcl=None,
                 mcr=None,
                 fid=None,
                 max_power=None,
                 max_power_rating=None,
                 overall_efficiency=None,
                 generator_pw_density=None,
                 rectifier_pw_density=None,
                 wiring_pw_density=None,
                 cooling_pw_density=None,
                 mass=None,
                 c_g=None):
        self.INFO = {
            "mto": {"unit": "uc", "om": 1.e0, "txt": "Take off power, mto<1: turbofan shaft power off take ratio, mto>1: e-fan motor power"},
            "mcn": {"unit": "uc", "om": 1.e0, "txt": "Maxi continuous power, mcn<1: turbofan shaft power off take ratio, mcn>1: e-fan motor power"},
            "mcl": {"unit": "uc", "om": 1.e0, "txt": "Max climb power, mcl<1: turbofan shaft power off take ratio, mcl>1: e-fan motor power"},
            "mcr": {"unit": "uc", "om": 1.e0, "txt": "Max cruise power, mcr<1: turbofan shaft power off take ratio, mcr>1: e-fan motor power"},
            "fid": {"unit": "uc", "om": 1.e0, "txt": "Flight idle power, fid<1: turbofan shaft power off take ratio, fid>1: e-fan motor power"},
            "max_power": {"unit": "kW", "om": 1.e4, "txt": "E-fan motor maximum power"},
            "max_power_rating": {"unit": "int", "om": 1.e0, "txt": "Engine rating of e-fan motor maximum power"},
            "overall_efficiency": {"unit": "no_dim", "om": 1.e0, "txt": "Power efficiency of the electric chain"},
            "generator_pw_density": {"unit": "kW/kg", "om": 1.e0, "txt": "Power density of electric generation"},
            "rectifier_pw_density": {"unit": "kW/kg", "om": 1.e0, "txt": "Power density of rectifiers"},
            "wiring_pw_density": {"unit": "kW/kg", "om": 1.e0, "txt": "Power density of wiring"},
            "cooling_pw_density": {"unit": "kW/kg", "om": 1.e0, "txt": "Power density of cooling system"},
            "mass": {"unit": "kg", "om": 1.e2, "txt": "Mass of the electric chain (generator, rectifier, wires, cooling)"},
            "c_g": {"unit": "m", "om": 1.e1, "txt": "Longitudinal position of the CG of the electric chain"}
        }
        self.mto = mto
        self.mcn = mcn
        self.mcl = mcl
        self.mcr = mcr
        self.fid = fid
        self.max_power = max_power
        self.max_power_rating = max_power_rating
        self.overall_efficiency = overall_efficiency
        self.generator_pw_density = generator_pw_density
        self.rectifier_pw_density = rectifier_pw_density
        self.wiring_pw_density = wiring_pw_density
        self.cooling_pw_density = cooling_pw_density
        self.mass = mass
        self.c_g = c_g

#-------------------------------------------------------------------------


class ElectricNacelle(object):
    """
    Electric nacelle data
    """

    def __init__(self, width=None,
                 length=None,
                 x_axe=None,
                 y_axe=None,
                 z_axe=None,
                 net_wetted_area=None,
                 efficiency_fan=None,
                 efficiency_prop=None,
                 motor_efficiency=None,
                 controller_efficiency=None,
                 controller_pw_density=None,
                 motor_pw_density=None,
                 nacelle_pw_density=None,
                 hub_width=None,
                 fan_width=None,
                 nozzle_width=None,
                 nozzle_area=None,
                 body_length=None,
                 bnd_layer=None,
                 mass=None,
                 c_g=None):
        self.INFO = {
            "width": {"unit": "m", "om": 1.e0, "txt": "Maximum width of the electric fan cowl"},
            "length": {"unit": "m", "om": 1.e0, "txt": "Length of the electric fan cowl"},
            "x_axe": {"unit": "m", "om": 1.e1, "txt": "Longitudinal position of the center of the electric nacelle air inlet"},
            "y_axe": {"unit": "m", "om": 1.e1, "txt": "Span wise position of the center of the electric nacelle air inlet"},
            "z_axe": {"unit": "m", "om": 1.e0, "txt": "Vertical position of the center of the electric nacelle air inlet"},
            "net_wetted_area": {"unit": "m2", "om": 1.e1, "txt": "Total net wetted area of the electric fan nacelle (fan cowl)"},
            "efficiency_fan": {"unit": "no_dim", "om": 1.e0, "txt": "Fan efficiency for turbofan (capability to turn shaft power into kinetic energy)"},
            "efficiency_prop": {"unit": "no_dim", "om": 1.e0, "txt": "Propeller like Fan+Cowl efficiency for turbofan (FanThrust.Vair)/(Shaft power)"},
            "motor_efficiency": {"unit": "no_dim", "om": 1.e0, "txt": "Motor efficiency"},
            "controller_efficiency": {"unit": "no_dim", "om": 1.e0, "txt": "Controller electric efficiency"},
            "controller_pw_density": {"unit": "kW/kg", "om": 1.e0, "txt": "Power density of controller"},
            "motor_pw_density": {"unit": "kW/kg", "om": 1.e0, "txt": "Power density of electric motor"},
            "nacelle_pw_density": {"unit": "kW/kg", "om": 1.e0, "txt": "Power density of e-fan nacelle and mountings"},
            "hub_width": {"unit": "m", "om": 1.e0, "txt": "Diameter of the hub of the electric nacelle"},
            "fan_width": {"unit": "m", "om": 1.e0, "txt": "Diameter of the fan of the electric nacelle"},
            "nozzle_width": {"unit": "m", "om": 1.e0, "txt": "Diameter of the nozzle of the electric nacelle"},
            "nozzle_area": {"unit": "m2", "om": 1.e0, "txt": "Exhaust nozzle area of the electric nacelle"},
            "body_length": {"unit": "m", "om": 1.e0, "txt": "Length of the body behind the electric nacelle"},
            "bnd_layer": {"unit": "structure", "om": 1.e0, "txt": "Boundary layer thickness law in front of the e-fan, 2d array"},
            "mass": {"unit": "kg", "om": 1.e2, "txt": "Equipped mass of the nacelle of the electric fan (including the controller, motor and nacelle)"},
            "c_g": {"unit": "m", "om": 1.e1, "txt": "Longitudinal position of the CG of the electric nacelle"}
        }
        self.width = width
        self.length = length
        self.x_axe = x_axe
        self.y_axe = y_axe
        self.z_axe = z_axe
        self.net_wetted_area = net_wetted_area
        self.efficiency_fan = efficiency_fan
        self.efficiency_prop = efficiency_prop
        self.motor_efficiency = motor_efficiency
        self.controller_efficiency = controller_efficiency
        self.controller_pw_density = controller_pw_density
        self.motor_pw_density = motor_pw_density
        self.nacelle_pw_density = nacelle_pw_density
        self.hub_width = hub_width
        self.fan_width = fan_width
        self.nozzle_width = nozzle_width
        self.nozzle_area = nozzle_area
        self.body_length = body_length
        self.bnd_layer = bnd_layer
        self.mass = mass
        self.c_g = c_g


#-------------------------------------------------------------------------
class ElectricEngine(object):
    """
    Electric motor rating power in given conditions
    """

    def __init__(self, mto_e_power_ratio=None,
                 mto_e_shaft_power=None,
                 mto_e_fan_thrust=None,
                 mcn_e_power_ratio=None,
                 mcn_e_shaft_power=None,
                 mcn_e_fan_thrust=None,
                 mcl_e_power_ratio=None,
                 mcl_e_shaft_power=None,
                 mcl_e_fan_thrust=None,
                 mcr_e_power_ratio=None,
                 mcr_e_shaft_power=None,
                 mcr_e_fan_thrust=None,
                 fid_e_power_ratio=None,
                 fid_e_shaft_power=None,
                 fid_e_fan_thrust=None,
                 flight_data=None):
        self.INFO = {
            "mto_e_power_ratio": {"unit": "no_dim", "om": 1.e0, "txt": "Turbofan off take power ratio in take off rating (one engine), Sea Level, ISA+15, Mach 0,25"},
            "mto_e_shaft_power": {"unit": "kW", "om": 1.e3, "txt": "E-fan shaft power in take off rating (one engine), Sea Level, ISA+15, Mach 0,25"},
            "mto_e_fan_thrust": {"unit": "daN", "om": 1.e3, "txt": "E-fan thrust in take off rating (one engine), Sea Level, ISA+15, Mach 0,25"},
            "mcn_e_power_ratio": {"unit": "no_dim", "om": 1.e0, "txt": "Turbofan off take power ratio in maxi continuous rating (one engine), required ceiling altitude, ISA, half cruise Mach"},
            "mcn_e_shaft_power": {"unit": "kW", "om": 1.e3, "txt": "E-fan shaft power in maxi continuous rating (one engine), required ceiling altitude, ISA, cruise Mach"},
            "mcn_e_fan_thrust": {"unit": "daN", "om": 1.e3, "txt": "E-fan thrust in maxi continuous rating (one engine), required ceiling altitude, ISA, cruise Mach"},
            "mcl_e_power_ratio": {"unit": "no_dim", "om": 1.e0, "txt": "Turbofan off take power ratio in max climb rating (one engine), required Top of Climb altitude, ISA, cruise Mach"},
            "mcl_e_shaft_power": {"unit": "kW", "om": 1.e3, "txt": "E-fan shaft power in max climb rating (one engine), required Top of Climb altitude, ISA, cruise Mach"},
            "mcl_e_fan_thrust": {"unit": "daN", "om": 1.e3, "txt": "E-fan thrust in max climb rating (one engine), required Top of Climb altitude, ISA, cruise Mach"},
            "mcr_e_power_ratio": {"unit": "no_dim", "om": 1.e0, "txt": "Turbofan off take power ratio in max cruise rating (one engine), reference cruise altitude, ISA, cruise Mach"},
            "mcr_e_shaft_power": {"unit": "kW", "om": 1.e3, "txt": "E-fan shaft power in max cruise rating (one engine), reference cruise altitude, ISA, cruise Mach"},
            "mcr_e_fan_thrust": {"unit": "daN", "om": 1.e3, "txt": "E-fan thrust in max cruise rating (one engine), reference cruise altitude, ISA, cruise Mach"},
            "fid_e_power_ratio": {"unit": "no_dim", "om": 1.e0, "txt": "Turbofan off take power ratio in flight idle rating (one engine), reference cruise altitude, ISA, cruise Mach"},
            "fid_e_shaft_power": {"unit": "kW", "om": 1.e3, "txt": "E-fan shaft power in flight idle rating (one engine), reference cruise altitude, ISA, cruise Mach"},
            "fid_e_fan_thrust": {"unit": "daN", "om": 1.e3, "txt": "E-fan thrust in flight idle rating (one engine), reference cruise altitude, ISA, cruise Mach"},
            "flight_data": {"unit": "dict", "txt": "Dictionary of flying conditions for each rating {'disa':array, 'altp':array, 'mach':array, 'nei':array}"}
        }
        self.mto_e_power_ratio = mto_e_power_ratio
        self.mto_e_shaft_power = mto_e_shaft_power
        self.mto_e_fan_thrust = mto_e_fan_thrust
        self.mcn_e_power_ratio = mcn_e_power_ratio
        self.mcn_e_shaft_power = mcn_e_shaft_power
        self.mcn_e_fan_thrust = mcn_e_fan_thrust
        self.mcl_e_power_ratio = mcl_e_power_ratio
        self.mcl_e_shaft_power = mcl_e_shaft_power
        self.mcl_e_fan_thrust = mcl_e_fan_thrust
        self.mcr_e_power_ratio = mcr_e_power_ratio
        self.mcr_e_shaft_power = mcr_e_shaft_power
        self.mcr_e_fan_thrust = mcr_e_fan_thrust
        self.fid_e_power_ratio = fid_e_power_ratio
        self.fid_e_shaft_power = fid_e_shaft_power
        self.fid_e_fan_thrust = fid_e_fan_thrust
        self.flight_data = flight_data

#-------------------------------------------------------------------------


class Battery(object):
    """
    Battery data
    """

    def __init__(self, strategy=None,
                 power_feed=None,
                 time_feed=None,
                 energy_cruise=None,
                 energy_density=None,
                 power_density=None,
                 mass=None,
                 c_g=None):
        self.INFO = {
            "strategy": {"unit": "int", "om": 1.e0, "txt": "Battery sizing strategy, 1: power_feed & energy_cruise driven, 2: battery mass driven"},
            "power_feed": {"unit": "kW", "om": 1.e4, "txt": "Power delivered to e-fan(s) at take off and(or) climb during a total of time_feed"},
            "time_feed": {"unit": "min", "om": 1.e1, "txt": "Maximum duration of the power_feed delivered to e-fan(s)"},
            "energy_cruise": {"unit": "kWh", "om": 1.e1, "txt": "Total battery energy dedicated to cruise"},
            "energy_density": {"unit": "kWh/kg", "om": 1.e0, "txt": "Battery energy density"},
            "power_density": {"unit": "kW/kg", "om": 1.e0, "txt": "Battery power density (capability to release power per mass unit"},
            "mass": {"unit": "kg", "om": 1.e3, "txt": "Total battery mass"},
            "c_g": {"unit": "m", "om": 1.e1, "txt": "Global CG of batteries"}
        }
        self.strategy = strategy
        self.power_feed = power_feed
        self.time_feed = time_feed
        self.energy_cruise = energy_cruise
        self.energy_density = energy_density
        self.power_density = power_density
        self.mass = mass
        self.c_g = c_g
