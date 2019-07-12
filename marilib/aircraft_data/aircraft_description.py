#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:22:21 2019

@author: DRUOT Thierry, DELMIRO Thales, GALLARD Francois

"""
from collections import OrderedDict
from datetime import datetime
import itertools

from configobj import ConfigObj

from marilib.tools import units as unit

from numpy import max, ceil, log10, floor, float64, arange, abs, array, ndarray

from marilib.aircraft_data.operational_performances \
    import DesignDriver, LowSpeed, HighSpeed, MaxPayloadMission, \
    NominalMission, MaxFuelMission, ZeroPayloadMission, \
    CostMission, Economics, Environmental_Impact

from marilib.aircraft_data.physical_performances \
    import Aerodynamics, Propulsion, CharacteristicWeight, CenterOfGravity

from marilib.airplane.airframe.airframe_data \
    import Cabin, Payload, Fuselage, Wing, Tanks, LandingGears, \
    Systems, HorizontalTail, VerticalTail

from marilib.airplane.propulsion.turbofan.turbofan_data \
    import TurbofanPylon, TurbofanNacelle, TurbofanEngine

from marilib.airplane.propulsion.hybrid_pte1.hybrid_pte1_data \
    import PowerElectricChain, Battery, ElectricNacelle, ElectricEngine

STANDARD_FORMAT = 4

#-------------------------------------------------------------------------


class Aircraft(object):
    """
    Assembling all aircraft data branches
    """

    def __init__(self, name=None):
        """
            Data structure branches, no ramification
        """
        self.name = name
        self.design_driver = DesignDriver()
        self.low_speed = LowSpeed()
        self.high_speed = HighSpeed()
        self.max_payload_mission = MaxPayloadMission()
        self.nominal_mission = NominalMission()
        self.max_fuel_mission = MaxFuelMission()
        self.zero_payload_mission = ZeroPayloadMission()
        self.cost_mission = CostMission()
        self.economics = Economics()
        self.environmental_impact = Environmental_Impact()

        self.aerodynamics = Aerodynamics()
        self.propulsion = Propulsion()
        self.weights = CharacteristicWeight()
        self.center_of_gravity = CenterOfGravity()

        self.cabin = Cabin()
        self.payload = Payload()
        self.fuselage = Fuselage()
        self.wing = Wing()
        self.landing_gears = LandingGears()
        self.horizontal_tail = HorizontalTail()
        self.vertical_tail = VerticalTail()
        self.tanks = Tanks()
        self.systems = Systems()

        self.turbofan_pylon = TurbofanPylon()
        self.turbofan_nacelle = TurbofanNacelle()
        self.turbofan_engine = TurbofanEngine()

        self.power_elec_chain = PowerElectricChain()
        self.electric_nacelle = ElectricNacelle()
        self.electric_engine = ElectricEngine()
        self.battery = Battery()

    def import_from_file(
            self, filename="Aircraft.ini", has_custom_units=False):

        in_parser = ConfigObj(filename, indent_type="    ")
        data_dict = in_parser["Aircraft"]
        set_ac_data(data_dict, self, has_custom_units)

    def export_to_file(self, filename="Aircraft.ini", def_order=True,
                       user_format=STANDARD_FORMAT, write_unit=False,
                       write_om=False, write_detail=False):
        """
        Build  ini file :
            Data tree file
            :param filename: Output file path - default value "Aircraft.ini"
            :param def_order: parameters' order - default value True for class definition order
                                                  alternative value False for alphabetical order
            :param user_format: parameters' value format - default value True or 4 for 4 decimals format
                                                           disable option with False or -1
            :param write_unit: Boolean to write the unit after the variable's value - default value False
            :param write_om: Boolean to write the order of magnitude as a inline comment - default value False
            :param write_detail: Boolean to write the varaible's description as a inline comment - default value False
        """

        out_parser = ConfigObj(indent_type="    ")
        out_parser.filename = filename

        if def_order:  # class definition order
            data_dict = self.get_ordered_data_dict()
            write_ordered_data_dict(data_dict, out_parser,
                                    user_format, None, write_unit, write_om, write_detail)

        else:  # alphabetical order
            data_dict = self.get_data_dict()
            write_data_dict(data_dict, out_parser,
                            user_format, None, write_unit, write_om, write_detail)

        timenow = datetime.now()
        date_hour = str(timenow.strftime("%d-%m-%Y %H:%M:%S"))
        out_parser.initial_comment = ["MARILib + GEMS scenario",
                                      "Created in " + date_hour]

        out_parser.write()

    def get_data_dict(self):
        return get_data_dict(self, "Aircraft", {})

    def get_ordered_data_dict(self):
        return get_ordered_data_dict(self, "Aircraft", OrderedDict())


#------------------------------------------------------------------------------


def get_proper_value(value, obj, key, custom_unit=None):
    is_negative = False
    isnumber = False
    if isinstance(value, str):
        if value[0] is '-':
            value = value.replace('-', '', 1)
            is_negative = True
        if value.isdigit():  # check int
            value = int(value)
            if is_negative:
                value *= -1
            isnumber = True
        elif value.replace('.', '', 1).isdigit():  # check float
            value = float(value)
            if is_negative:
                value *= -1
            isnumber = True
        elif value == "None":
            return None
        else:
            return value
    if isnumber:
        if custom_unit is None:
            if "info" in obj.__dict__:
                value_unit = obj.info[key]['unit']
            else:
                return value
        correct_value = unit.convert_from(value_unit, value)
        return correct_value
    else:
        raise NotImplementedError

#------------------------------------------------------------------------------


def set_ac_data(data_dict, obj, has_custom_units):
    for attr_path, attr_val in data_dict.iteritems():
        if hasattr(attr_val, "__dict__"):
            sub_attr = getattr(obj, attr_path)
            set_ac_data(attr_val, sub_attr, has_custom_units)
        else:
            data_line = attr_val.rsplit(None, 1)
            value_sequence = [data_line[0]]
            custom_unit = None
            if has_custom_units:
                custom_unit = data_line[-1]
            if value_sequence[0][0] == '[':
                value_sequence = data_line[0][1:-1].replace(",", "").split()
            attr_val = []
            for v in value_sequence:
                v = get_proper_value(v, obj, attr_path, custom_unit)
                attr_val.append(v)
            if len(attr_val) is 1:
                attr_val = attr_val[0]
            else:
                attr_val = tuple(attr_val)
                elif initial_char == '{':
                    attr_val = dict(itertools.izip(k, attr_val))
            setattr(obj, attr_path, attr_val)

#------------------------------------------------------------------------------


def write_data_line(value, key, out_parser, user_format, info_dict,
                    write_unit, write_om, write_detail):
    unit_str = ""
    comment_line = ""
    comment_inline = False
    if info_dict is not None:
        if write_unit and 'unit' in info_dict[key]:
            unit_str = info_dict[key]['unit']
            value = unit.convert_to(unit_str, value)
            unit_str = " " + unit_str
        if any((write_om, write_detail)):
            if write_om and 'om' in info_dict[key]:
                comment_line = "om: " + \
                    "{:.0e}".format(info_dict[key]['om'])
            if write_detail and 'txt' in info_dict[key]:
                comment_line += " " + info_dict[key]['txt']
            comment_inline = True
    if not user_format or user_format is -1:
        out_parser[key] = "{0}{1}".format(value,
                                          unit_str)
    elif user_format:
        if user_format is True:
            user_format = STANDARD_FORMAT
        out_parser[key] = "{0}{1}".format(to_user_format(value, user_format),
                                          unit_str)
    if comment_inline:
        out_parser.inline_comments[key] = comment_line


def write_data_dict(data_dict, out_parser,
                    user_format, info_dict, write_unit, write_om, write_detail):
    if any((write_unit, write_om, write_detail)) is False:
        info_dict = None
        data_dict.pop("info", None)
    elif info_dict is None:
        info_dict = data_dict.pop("info", None)
    for key in sorted(data_dict.keys()):
        value = data_dict[key]

        if isinstance(value, dict):
            out_parser[key] = {}
            write_data_dict(value,
                            out_parser[key],
                            user_format,
                            info_dict,
                            write_unit,
                            write_om,
                            write_detail)

        else:
            write_data_line(value,
                            key,
                            out_parser,
                            user_format,
                            info_dict,
                            write_unit,
                            write_om,
                            write_detail)


#------------------------------------------------------------------------------

def write_ordered_data_dict(data_dict, out_parser,
                            user_format, info_dict, write_unit, write_om, write_detail):
    if any((write_unit, write_om, write_detail)) is False:
        info_dict = None
        data_dict.pop("info", None)
    elif info_dict is None:
        info_dict = data_dict.pop("info", None)
    for key in data_dict:
        value = data_dict[key]

        if isinstance(value, OrderedDict):
            out_parser[key] = OrderedDict()
            write_ordered_data_dict(value,
                                    out_parser[key],
                                    user_format,
                                    info_dict,
                                    write_unit,
                                    write_om,
                                    write_detail)

        else:
            write_data_line(value,
                            key,
                            out_parser,
                            user_format,
                            info_dict,
                            write_unit,
                            write_om,
                            write_detail)

#-------------------------------------------------------------------------


def isNaN(num):
    return num != num


def convert_to_orig_type(lst, orig_seq):

    if isinstance(orig_seq, tuple):
        return tuple(lst)
    elif isinstance(orig_seq, ndarray):
        return array(lst)
    else:
        return lst


#-------------------------------------------------------------------------
def to_user_format(value, dec_format):

    if isNaN(value):
        return value
    elif isinstance(value, (tuple, list, ndarray)):
        lst = list(value)
        for i in arange(len(lst)):
            lst[i] = to_user_format(lst[i], dec_format)
        return str(convert_to_orig_type(lst, value)).replace("'", "")
    elif isinstance(value, (float, float64)):
        if value == 0. or value == -0.:
            return format(value, "".join((".", str(dec_format), "f")))
        else:
            V = abs(value)
            if abs(value) > 1:
                correction_factor = 1e-4  # to correct 10^n values format
                nb_dec = int(
                    max((0, (dec_format + 1) - ceil(log10(V + correction_factor)))))
            else:
                nb_dec = int((dec_format - 1) - floor(log10(V)))
            return format(value, "".join((".", str(nb_dec), "f")))
    else:
        return value


#-------------------------------------------------------------------------
def is_basetype(obj):

    if obj is None or not hasattr(obj, "__dict__"):
        return True

    return False


#-------------------------------------------------------------------------

def get_data_dict(obj, obj_name, data_dict):

    curr_data_d = {}
    data_dict[obj_name] = curr_data_d

    if not hasattr(obj, "__dict__"):
        return

    for attr_name in obj.__dict__.keys():
        attribute = getattr(obj, attr_name)

        if is_basetype(attribute):
            curr_data_d[attr_name] = attribute

        else:
            get_data_dict(attribute, attr_name, curr_data_d)

    return data_dict

#-------------------------------------------------------------------------


def get_ordered_data_dict(obj, obj_name, ord_dict):
    from inspect import getsource
    ord_dict_d = OrderedDict()
    ord_dict[obj_name] = ord_dict_d
    if not hasattr(obj, "__dict__"):
        return
    info = getsource(obj.__class__)
    i = info.find("__init__")
    while i <= len(info):
        occur = info.find("self.", i)
        if occur == -1:  # No attribute or end of class
            return ord_dict
        else:
            equal = info.find("=", occur)
            eol = info.find("\n", occur)
            if equal == -1 or equal >= eol:  # no "=" inline
                return ord_dict
            else:
                spc = info.find(" ", occur, equal)
                if spc != -1:
                    attr_name = info[occur + 5:spc]
                else:
                    attr_name = info[occur + 5:equal]
                if attr_name in obj.__dict__.keys():
                    attribute = getattr(obj, attr_name)
                    if is_basetype(attribute):
                        ord_dict_d[attr_name] = attribute
                    else:
                        get_ordered_data_dict(attribute, attr_name, ord_dict_d)
                i = eol + 1
    return ord_dict
