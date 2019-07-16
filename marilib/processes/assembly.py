#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:22:21 2019

@author: DRUOT Thierry

Changed name from "design.py" to "assembly.py" on 21:05:2019
"""

from marilib import numpy
# non backward compatible modules: SR1, NonlinearConstraint,BFGS
from scipy.optimize import minimize

from marilib.aircraft_model.airplane import airplane_design as airplane, regulation as regul
from marilib.aircraft_model.operations import handling_qualities as h_q, \
    mission as perfo
from marilib.airplane.airframe import airframe_design as airframe
from marilib.airplane.propulsion import propulsion_design as propulsion
from marilib.earth import environment as earth
from marilib.processes import component as sub_proc, initialization as init
from marilib.tools import units as unit
from marilib.tools.math import newton_solve


#=========================================================================
def aircraft_initialize(
        aircraft, n_pax_ref, design_range, cruise_mach, propu_config, n_engine):
    """
    Initialize a generic aircraft
    """

    aircraft.propulsion.architecture = propu_config

    aircraft.propulsion.fuel_type = init.fuel_type()

    aircraft.name = "my_test_airplane"
    aircraft.design_driver.design_range = design_range        # TLR
    aircraft.design_driver.cruise_mach = cruise_mach          # TLR
    aircraft.cabin.n_pax_ref = n_pax_ref                      # TLR

    aircraft.design_driver.ref_cruise_altp = init.ref_cruise_altp(
        propu_config)        # TLR
    aircraft.design_driver.top_of_climb_altp = init.top_of_climb_altp(
        propu_config)    # TLR

    aircraft.aerodynamics.hld_conf_clean = init.hld_conf_clean()
    aircraft.aerodynamics.hld_conf_ld = init.hld_conf_ld()

    aircraft.low_speed.altp_tofl = init.altp_tofl()
    aircraft.low_speed.disa_tofl = init.disa_tofl()
    # Regulation
    aircraft.low_speed.kvs1g_tofl = regul.kvs1g_min_take_off()
    aircraft.low_speed.req_tofl = init.req_tofl(design_range)        # TLR

    aircraft.low_speed.altp_app_speed = init.altp_app_speed()
    aircraft.low_speed.disa_app_speed = init.disa_app_speed()
    # Regulation
    aircraft.low_speed.kvs1g_app_speed = regul.kvs1g_min_landing()
    aircraft.low_speed.req_app_speed = init.req_app_speed(
        n_pax_ref)          # TLR

    aircraft.low_speed.disa_oei = init.disa_oei()
    aircraft.low_speed.req_oei_path = regul.ceil_oei_min_path(
        n_engine)     # Regulation
    aircraft.low_speed.req_oei_altp = init.req_oei_altp(
        propu_config)       # TLR

    aircraft.high_speed.disa_climb = init.disa_climb()
    aircraft.high_speed.req_vz_climb = init.req_vz_climb()           # TLR
    aircraft.high_speed.req_vz_cruise = init.req_vz_cruise()         # TLR
    aircraft.high_speed.req_toc_altp = init.top_of_climb_altp(propu_config)
    aircraft.high_speed.cas1_ttc = init.cas1_ttc(cruise_mach)
    aircraft.high_speed.cas2_ttc = init.cas2_ttc(cruise_mach)
    aircraft.high_speed.req_ttc = init.req_ttc()                     # TLR

    aircraft.cost_mission.disa = init.cost_mission_disa()
    aircraft.cost_mission.range = init.cost_mission_range(design_range)

    aircraft.economics.fuel_price = init.fuel_price()
    aircraft.economics.elec_price = init.elec_price()
    aircraft.economics.battery_mass_price = init.battery_mass_price()
    aircraft.economics.labor_cost = init.labor_cost()
    aircraft.economics.irp = init.irp()
    aircraft.economics.period = init.period()
    aircraft.economics.interest_rate = init.interest_rate()
    aircraft.economics.utilisation = init.utilisation(design_range)

    aircraft.environmental_impact.CO2_index = earth.emission_index("CO2")
    aircraft.environmental_impact.H2O_index = earth.emission_index("H2O")
    aircraft.environmental_impact.SO2_index = earth.emission_index("SO2")
    aircraft.environmental_impact.NOx_index = earth.emission_index("NOx")
    aircraft.environmental_impact.CO_index = earth.emission_index("CO")
    aircraft.environmental_impact.HC_index = earth.emission_index("HC")
    aircraft.environmental_impact.sulfuric_acid_index = earth.emission_index(
        "sulfuric_acid")
    aircraft.environmental_impact.nitrous_acid_index = earth.emission_index(
        "nitrous_acid")
    aircraft.environmental_impact.nitric_acid_index = earth.emission_index(
        "nitric_acid")
    aircraft.environmental_impact.soot_index = earth.emission_index("soot")

    aircraft.weights.mzfw = init.mzfw(n_pax_ref, design_range)
    aircraft.weights.mtow = init.mtow(n_pax_ref, design_range)
    aircraft.weights.mlw = init.mlw(
        n_pax_ref,
        aircraft.weights.mtow,
        aircraft.weights.mzfw)

    aircraft.cabin.n_pax_front = init.n_pax_front(n_pax_ref)
    aircraft.cabin.n_aisle = init.n_aisle(aircraft.cabin.n_pax_front)
    aircraft.fuselage.width = init.fuselage_width(
        aircraft.cabin.n_pax_front,
        aircraft.cabin.n_aisle)

    aircraft.payload.m_pax_nominal = init.m_pax_nominal(
        design_range)      # TLR
    aircraft.payload.m_pax_max = init.m_pax_max(
        design_range)              # TLR

    aircraft.center_of_gravity.cg_range_optimization = init.cg_range_optimization()

    aircraft.wing.attachment = init.wing_attachment()
    aircraft.wing.morphing = init.wing_morphing()
    aircraft.wing.hld_type = init.hld_type(n_pax_ref)

    # Main design variable
    aircraft.wing.area = init.wing_area(n_pax_ref, design_range)
    aircraft.wing.aspect_ratio = init.wing_aspect_ratio()
    aircraft.wing.span = init.wing_span(
        aircraft.wing.area,
        aircraft.wing.aspect_ratio)
    aircraft.wing.x_root = init.wing_x_root(
        aircraft.wing.aspect_ratio,
        aircraft.wing.span)

    aircraft.horizontal_tail.area = init.htp_area(aircraft.wing.area)
    aircraft.vertical_tail.area = init.vtp_area(aircraft.wing.area)

    #-------------------------------------------------------------------------
    aircraft.turbofan_engine.n_engine = n_engine
    aircraft.turbofan_engine.bpr = init.bpr(n_pax_ref)
    aircraft.turbofan_engine.reference_thrust = init.reference_thrust(
        n_pax_ref,
        design_range,
        n_engine)                                            # Main design variable

    aircraft.turbofan_engine.core_thrust_ratio = init.core_thrust_ratio()
    aircraft.turbofan_engine.core_width_ratio = init.core_width_ratio()
    aircraft.turbofan_engine.core_weight_ratio = init.core_weight_ratio()

    aircraft.turbofan_nacelle.attachment = init.nacelle_attachment(n_pax_ref)
    aircraft.turbofan_nacelle.efficiency_fan = init.efficiency_fan()
    aircraft.turbofan_nacelle.efficiency_prop = init.efficiency_prop()
    aircraft.turbofan_nacelle.width = init.nacelle_width(aircraft.turbofan_engine.bpr,
                                                         aircraft.turbofan_engine.reference_thrust)
    aircraft.turbofan_nacelle.y_ext = init.nacelle_y_ext(aircraft.turbofan_nacelle.attachment,
                                                         aircraft.fuselage.width,
                                                         aircraft.turbofan_nacelle.width)

    #-------------------------------------------------------------------------
    aircraft.rear_electric_nacelle.efficiency_fan = init.efficiency_fan()
    aircraft.rear_electric_nacelle.efficiency_prop = init.efficiency_prop()
    aircraft.rear_electric_nacelle.motor_efficiency = init.e_motor_efficiency()
    aircraft.rear_electric_nacelle.controller_efficiency = init.controller_efficiency()
    aircraft.rear_electric_nacelle.controller_pw_density = init.controller_pw_density()
    aircraft.rear_electric_nacelle.motor_pw_density = init.e_motor_pw_density()
    aircraft.rear_electric_nacelle.nacelle_pw_density = init.e_nacelle_pw_density()

    # Watts, electric motor power
    aircraft.pte1_power_elec_chain.mto = init.electric_shaft_power()
    # Watts, electric motor power
    aircraft.pte1_power_elec_chain.mcn = init.electric_shaft_power()
    # Watts, electric motor power
    aircraft.pte1_power_elec_chain.mcl = init.electric_shaft_power()
    # Watts, electric motor power
    aircraft.pte1_power_elec_chain.mcr = init.electric_shaft_power()
    aircraft.pte1_power_elec_chain.fid = 0.01

    aircraft.pte1_power_elec_chain.overall_efficiency = init.e_chain_efficiency()
    aircraft.pte1_power_elec_chain.generator_pw_density = init.generator_power_density()
    aircraft.pte1_power_elec_chain.rectifier_pw_density = init.rectifier_pw_density()
    aircraft.pte1_power_elec_chain.wiring_pw_density = init.wiring_pw_density()
    aircraft.pte1_power_elec_chain.cooling_pw_density = init.cooling_pw_density()

    aircraft.pte1_battery.strategy = init.battery_strategy()
    aircraft.pte1_battery.power_feed = init.battery_power_feed()
    aircraft.pte1_battery.time_feed = init.battery_time_feed()
    aircraft.pte1_battery.energy_cruise = init.battery_energy_cruise()
    aircraft.pte1_battery.energy_density = init.battery_energy_density()
    aircraft.pte1_battery.power_density = init.battery_power_density()

    #-------------------------------------------------------------------------
    aircraft.electrofan_engine.n_engine = n_engine
    aircraft.electrofan_engine.reference_thrust = init.reference_thrust(
        n_pax_ref,
        design_range,
        n_engine)                                            # Main design variable

    aircraft.electrofan_nacelle.attachment = init.nacelle_attachment(n_pax_ref)
    aircraft.electrofan_nacelle.efficiency_fan = init.efficiency_fan()
    aircraft.electrofan_nacelle.efficiency_prop = init.efficiency_prop()
    aircraft.electrofan_nacelle.width = init.nacelle_width(
        9, aircraft.electrofan_engine.reference_thrust)
    aircraft.turbofan_nacelle.y_ext = init.nacelle_y_ext(aircraft.electrofan_nacelle.attachment,
                                                         aircraft.fuselage.width,
                                                         aircraft.electrofan_nacelle.width)

    aircraft.ef1_power_elec_chain.overall_efficiency = init.e_chain_efficiency()
    aircraft.ef1_power_elec_chain.generator_pw_density = init.generator_power_density()
    aircraft.ef1_power_elec_chain.rectifier_pw_density = init.rectifier_pw_density()
    aircraft.ef1_power_elec_chain.wiring_pw_density = init.wiring_pw_density()
    aircraft.ef1_power_elec_chain.cooling_pw_density = init.cooling_pw_density()

    aircraft.ef1_battery.energy_density = init.battery_energy_density()
    aircraft.ef1_battery.power_density = init.battery_power_density()
    aircraft.ef1_battery.density = init.battery_density()
    aircraft.ef1_battery.fill_factor = init.battery_fill_factor()

    if (propu_config == "TF" or propu_config == "PTE1"):

        aircraft.horizontal_tail.attachment = init.htp_attachment(
            aircraft.turbofan_nacelle.attachment)

    elif (propu_config == "EF1"):

        aircraft.horizontal_tail.attachment = init.htp_attachment(
            aircraft.electrofan_nacelle.attachment)

    else:

        raise Exception("Propulsion architecture type is unknown")

    #-------------------------------------------------------------------------
    aircraft.propulsion.bli_effect = init.boundary_layer_effect()

    aircraft.propulsion.rating_code = init.rating_code()

    return


#=========================================================================
def eval_aircraft_geom_analysis(aircraft):
    """
    Perform geometrical analysis
    Do not solve any coupling
    """

    airframe.eval_cabin_design(aircraft)

    airframe.eval_fuselage_design(aircraft)

    airframe.eval_wing_design(aircraft)

    airframe.eval_vtp_design(aircraft)

    airframe.eval_vtp_statistical_sizing(aircraft)

    airframe.eval_htp_design(aircraft)

    airframe.eval_htp_statistical_sizing(aircraft)

    propulsion.eval_propulsion_design(aircraft)

    airplane.eval_aerodynamics_design(aircraft)

    return


#=========================================================================
def eval_tail_statistical_sizing(aircraft):
    """
    Perform tail area sizing according to statistical rules
    Do not manage the coupling carried by nacelle geometry
    """

    airframe.eval_cabin_design(aircraft)
    airframe.eval_fuselage_design(aircraft)

    #=========================================================================
    def fct_aircraft_pre_design(x_in, aircraft):

        ac = aircraft

        # Coupling variable
        ac.horizontal_tail.area = x_in[0]
        # Coupling variable
        ac.vertical_tail.area = x_in[1]

        airframe.eval_wing_design(ac)
        airframe.eval_vtp_design(ac)
        airframe.eval_htp_design(ac)
        airframe.eval_vtp_statistical_sizing(ac)
        airframe.eval_htp_statistical_sizing(ac)

        y_out = numpy.array([x_in[0] - ac.horizontal_tail.area,
                             x_in[1] - ac.vertical_tail.area])

        return y_out
    #-------------------------------------------------------------------------

    htp_area_i = init.htp_area(aircraft.wing.area)
    vtp_area_i = init.vtp_area(aircraft.wing.area)

    x_ini = numpy.array([htp_area_i, vtp_area_i])

    fct_arg = (aircraft,)

    result, _, _ = newton_solve(fct_aircraft_pre_design,
                                x_ini,  # dres_dy=jac,
                                args=fct_arg)

    # Coupling variable
    aircraft.horizontal_tail.area = result[0]
    # Coupling variable
    aircraft.vertical_tail.area = result[1]

    airframe.eval_wing_design(aircraft)
    airframe.eval_vtp_design(aircraft)
    airframe.eval_htp_design(aircraft)

    propulsion.eval_propulsion_design(aircraft)

    airplane.eval_aerodynamics_design(aircraft)

    return


#=========================================================================
def eval_aircraft_pre_design(aircraft):
    """
    Solves the coupling carried by nacelle geometry only
    Do not perform tail area sizing
    """

    airframe.eval_cabin_design(aircraft)
    airframe.eval_fuselage_design(aircraft)

    #=========================================================================
    def fct_aircraft_pre_design(x_in, aircraft):

        ac = aircraft

        # Coupling variable
        ac.turbofan_nacelle.width = x_in[0]
        # Coupling variable
        ac.turbofan_nacelle.y_ext = x_in[1]

        airframe.eval_wing_design(ac)
        airframe.eval_vtp_design(ac)
        airframe.eval_htp_design(ac)

        propulsion.eval_propulsion_design(ac)

        y_out = numpy.array([x_in[0] - ac.turbofan_nacelle.width,
                             x_in[1] - ac.turbofan_nacelle.y_ext])

        return y_out
    #-------------------------------------------------------------------------

    x_ini = numpy.array(
        [aircraft.turbofan_nacelle.width, aircraft.turbofan_nacelle.y_ext])

    fct_arg = (aircraft,)

    result, _, _ = newton_solve(fct_aircraft_pre_design,
                                x_ini,  # dres_dy=jac,
                                args=fct_arg)

    # Coupling variable
    aircraft.turbofan_nacelle.width = result[0]
    # Coupling variable
    aircraft.turbofan_nacelle.y_ext = result[1]

    airframe.eval_wing_design(aircraft)
    airframe.eval_vtp_design(aircraft)
    airframe.eval_htp_design(aircraft)

    propulsion.eval_propulsion_design(aircraft)

    airplane.eval_aerodynamics_design(aircraft)

    return


#=========================================================================
def eval_aircraft_statistical_pre_design(aircraft):
    """
    Perform tail area sizing according to statistical rules on top of the coupling carried by nacelle geometry
    """

    airframe.eval_cabin_design(aircraft)
    airframe.eval_fuselage_design(aircraft)

    #=========================================================================
    def fct_aircraft_pre_design(x_in, aircraft):

        ac = aircraft

        # Coupling variable
        ac.horizontal_tail.area = x_in[0]
        # Coupling variable
        ac.vertical_tail.area = x_in[1]

        eval_aircraft_pre_design(ac)

        airframe.eval_vtp_statistical_sizing(ac)
        airframe.eval_htp_statistical_sizing(ac)

        y_out = numpy.array([x_in[0] - ac.horizontal_tail.area,
                             x_in[1] - ac.vertical_tail.area])

        return y_out
    #-------------------------------------------------------------------------

    htp_area_i = init.htp_area(aircraft.wing.area)
    vtp_area_i = init.vtp_area(aircraft.wing.area)

    x_ini = numpy.array([htp_area_i, vtp_area_i])

    fct_arg = (aircraft,)
    result, _, _ = newton_solve(fct_aircraft_pre_design,
                                x_ini,  # dres_dy=jac,
                                args=fct_arg)

    # Coupling variable
    aircraft.horizontal_tail.area = result[0]
    # Coupling variable
    aircraft.vertical_tail.area = result[1]

    airframe.eval_wing_design(aircraft)
    airframe.eval_vtp_design(aircraft)
    airframe.eval_htp_design(aircraft)

    propulsion.eval_propulsion_design(aircraft)

    airplane.eval_aerodynamics_design(aircraft)

    return


#=========================================================================
def eval_aircraft_statistical_pre_design_2(aircraft):
    """
    Solves the coupling carried by nacelle geometry on top of the statistical sizing of tail areas
    """

    airframe.eval_cabin_design(aircraft)
    airframe.eval_fuselage_design(aircraft)

    #=========================================================================
    def fct_aircraft_statistical_pre_design(x_in, aircraft):

        ac = aircraft

        # Coupling variable
        ac.turbofan_nacelle.width = x_in[0]
        # Coupling variable
        ac.turbofan_nacelle.y_ext = x_in[1]

        eval_tail_statistical_sizing(ac)            # Nested solving

        propulsion.eval_propulsion_design(ac)

        y_out = numpy.array([x_in[0] - ac.turbofan_nacelle.width,
                             x_in[1] - ac.turbofan_nacelle.y_ext])

        return y_out
    #-------------------------------------------------------------------------

    x_ini = numpy.array(
        [aircraft.turbofan_nacelle.width, aircraft.turbofan_nacelle.y_ext])

    fct_arg = (aircraft,)

    result, _, _ = newton_solve(fct_aircraft_statistical_pre_design,
                                x_ini,  # dres_dy=jac,
                                args=fct_arg)

    # Coupling variable
    aircraft.turbofan_nacelle.width = result[0]
    # Coupling variable
    aircraft.turbofan_nacelle.y_ext = result[1]

    airframe.eval_wing_design(aircraft)
    airframe.eval_vtp_design(aircraft)
    airframe.eval_htp_design(aircraft)

    propulsion.eval_propulsion_design(aircraft)

    airplane.eval_aerodynamics_design(aircraft)

    return


#=========================================================================
def eval_mass_breakdown(aircraft):
    """
    Estimate mass and CGs of the airplane
    Takes MTOW,k MZFW & MLW as input
    """

    airframe.eval_cabin_mass(aircraft)
    airframe.eval_fuselage_mass(aircraft)
    airframe.eval_vtp_mass(aircraft)
    airframe.eval_wing_mass(aircraft)
    airframe.eval_htp_mass(aircraft)
    airframe.eval_landing_gear_mass(aircraft)

    propulsion.eval_propulsion_mass(aircraft)
    propulsion.eval_battery_mass(aircraft)
    propulsion.eval_tank_data(aircraft)

    airplane.eval_system_mass(aircraft)
    airplane.eval_payload_mass(aircraft)
    airplane.eval_aircraft_weights(aircraft)

    airplane.eval_aircraft_cg(aircraft)

    return


#=========================================================================
def eval_mass_estimation(aircraft):
    """
    Estimate mass and CGs of the airplane
    Takes MTOW as input but solves the coupling carried by MZFW and MLW
    """

    #=========================================================================
    def fct_mass(x_in, aircraft):

        aircraft.weights.mlw = x_in[0]      # Coupling variable
        aircraft.weights.mzfw = x_in[1]     # Coupling variable

        eval_mass_breakdown(aircraft)

        y_out = numpy.array([aircraft.weights.mass_constraint_1,
                             aircraft.weights.mass_constraint_2])

        return y_out
    #-------------------------------------------------------------------------

    x_ini = numpy.array([aircraft.weights.mlw,
                         aircraft.weights.mzfw])

    fct_arg = (aircraft,)

    result, _, _ = newton_solve(fct_mass,
                                x_ini,  # dres_dy=jac,
                                args=fct_arg)

    # Coupling variable
    aircraft.weights.mlw = result[0]
    # Coupling variable
    aircraft.weights.mzfw = result[1]

    # Update mass
    #-------------------------------------------------------------------------
    eval_mass_breakdown(aircraft)

    return


#=========================================================================
def eval_mass_mission_adaptation(aircraft):
    """
    Perform mass - mission adaptation and update mass and CGs
    """

    #=========================================================================
    def fct_mass_mission(x_in, aircraft):

        aircraft.weights.mtow = x_in[0]             # Coupling variable
        aircraft.weights.mlw = x_in[1]              # Coupling variable
        aircraft.weights.mzfw = x_in[2]             # Coupling variable

        # Mass
        #----------------------------------------------------------------------
        eval_mass_breakdown(aircraft)

        # Mission
        #----------------------------------------------------------------------
        sub_proc.eval_nominal_mission(aircraft)

        y_out = numpy.array([aircraft.weights.mass_constraint_1,
                             aircraft.weights.mass_constraint_2,
                             aircraft.weights.mass_constraint_3])

        return y_out
    #-------------------------------------------------------------------------

    x_ini = numpy.array([aircraft.weights.mtow,
                         aircraft.weights.mlw,
                         aircraft.weights.mzfw])

    fct_arg = (aircraft,)

    result, _, _ = newton_solve(fct_mass_mission,
                                x_ini,  # dres_dy=jac,
                                args=fct_arg)

    # Coupling variable
    aircraft.weights.mtow = result[0]
    # Coupling variable
    aircraft.weights.mlw = result[1]
    # Coupling variable
    aircraft.weights.mzfw = result[2]

    # Update mass data
    #-------------------------------------------------------------------------
    eval_mass_breakdown(aircraft)

    # Update mission data
    #-------------------------------------------------------------------------
    sub_proc.eval_nominal_mission(aircraft)

    return


#=========================================================================
def eval_payload_range_analysis(aircraft):
    """
    Compute Payload - Range diagram corner points
    """
    disa = 0.
    altp = aircraft.design_driver.ref_cruise_altp
    mach = aircraft.design_driver.cruise_mach

    # Max payload mission
    #-------------------------------------------------------------------------
    tow = aircraft.weights.mtow
    payload = aircraft.payload.maximum

    aircraft.max_payload_mission.tow = tow
    aircraft.max_payload_mission.payload = payload

    range, block_fuel, block_time, total_fuel = sub_proc.mission_range(aircraft, tow, payload,
                                                                       altp, mach, disa)

    aircraft.max_payload_mission.range = range
    aircraft.max_payload_mission.block_fuel = block_fuel
    aircraft.max_payload_mission.block_time = block_time
    aircraft.max_payload_mission.total_fuel = total_fuel

    # Max fuel mission
    #-------------------------------------------------------------------------
    tow = aircraft.weights.mtow
    total_fuel = aircraft.weights.mfw

    aircraft.max_fuel_mission.tow = tow
    aircraft.max_fuel_mission.total_fuel = total_fuel

    range, payload, block_fuel, block_time = sub_proc.mission_fuel_limited(
        aircraft, tow, total_fuel, altp, mach, disa)

    aircraft.max_fuel_mission.payload = payload
    aircraft.max_fuel_mission.range = range
    aircraft.max_fuel_mission.block_fuel = block_fuel
    aircraft.max_fuel_mission.block_time = block_time

    # zero fuel mission
    #-------------------------------------------------------------------------
    total_fuel = aircraft.weights.mfw
    tow = aircraft.weights.owe + total_fuel

    aircraft.zero_payload_mission.tow = tow
    aircraft.zero_payload_mission.total_fuel = total_fuel

    range, payload, block_fuel, block_time = sub_proc.mission_fuel_limited(
        aircraft, tow, total_fuel, altp, mach, disa)

    aircraft.zero_payload_mission.range = range
    aircraft.zero_payload_mission.block_fuel = block_fuel
    aircraft.zero_payload_mission.block_time = block_time

    return


#=========================================================================
def eval_climb_performances(aircraft):
    """
    Compute climb performances
    """
    # Ceilings
    #-------------------------------------------------------------------------
    toc = aircraft.design_driver.top_of_climb_altp
    oei_ceil_req = aircraft.low_speed.req_oei_altp

    vz_clb, vz_crz, oei_path, oei_mach = perfo.ceilings(
        aircraft, toc, oei_ceil_req)

    aircraft.low_speed.eff_oei_path = oei_path
    aircraft.high_speed.eff_vz_climb = vz_clb
    aircraft.high_speed.eff_vz_cruise = vz_crz

    aircraft.low_speed.perfo_constraint_3 = (
        oei_path - aircraft.low_speed.req_oei_path) / aircraft.low_speed.req_oei_path

    aircraft.high_speed.perfo_constraint_1 = vz_clb - \
        aircraft.high_speed.req_vz_climb
    aircraft.high_speed.perfo_constraint_2 = vz_crz - \
        aircraft.high_speed.req_vz_cruise

    # Time to climb to requested altitude
    #-------------------------------------------------------------------------
    toc = aircraft.high_speed.req_toc_altp
    disa = 0.
    mass = aircraft.weights.mtow
    vcas1 = aircraft.high_speed.cas1_ttc
    vcas2 = aircraft.high_speed.cas2_ttc
    mach = aircraft.design_driver.cruise_mach

    ttc = perfo.time_to_climb(aircraft, toc, disa, mass, vcas1, vcas2, mach)

    aircraft.high_speed.eff_ttc = ttc

    aircraft.high_speed.perfo_constraint_3 = (
        aircraft.high_speed.req_ttc - ttc) / aircraft.high_speed.req_ttc

    return


#=========================================================================
def eval_performance_analysis(aircraft):
    """
    Compute operational performances
    """
    # Nominal mission
    #-------------------------------------------------------------------------
    sub_proc.eval_nominal_mission(aircraft)

    # Take off field length
    #-------------------------------------------------------------------------
    sub_proc.eval_take_off_performances(aircraft)

    # Approach speed
    #-------------------------------------------------------------------------
    sub_proc.eval_landing_performances(aircraft)

    # Climb performances
    #-------------------------------------------------------------------------
    eval_climb_performances(aircraft)

    # Environment
    #-------------------------------------------------------------------------
    sub_proc.eval_co2_metric(aircraft)

    # Cost mission
    #-------------------------------------------------------------------------
    sub_proc.eval_cost_mission(aircraft)

    # Economics
    #-------------------------------------------------------------------------
    sub_proc.eval_economics(aircraft)

    return


#=========================================================================
def eval_handling_quality_analysis(aircraft):
    """
    Compute CG limits from handling qualities point of view
    """

    c_g = aircraft.center_of_gravity

    # Forward limit : trim landing
    #-------------------------------------------------------------------------
    altp = unit.m_ft(0.)
    disa = 0.
    nei = 0
    speed_mode = 1
    hld_conf = aircraft.aerodynamics.hld_conf_ld
    mass = c_g.max_fwd_mass

    cg_max_fwd_stall, speed, fn, aoa, ih, c_z, cx_trimmed = h_q.forward_cg_stall(aircraft, altp, disa,
                                                                                 nei, hld_conf, speed_mode, mass)

    c_g.max_fwd_trim_cg = cg_max_fwd_stall         # Forward cg limit

    c_g.cg_constraint_1 = c_g.max_fwd_trim_cg - c_g.max_fwd_req_cg

    # Backward limit : static stability
    #-------------------------------------------------------------------------
    stability_margin = regul.static_stability_margin()

    cg_max_bwd_stab = h_q.backward_cg_stab(aircraft, stability_margin)

    c_g.max_bwd_stab_cg = cg_max_bwd_stab          # Backward cg limit

    c_g.cg_constraint_2 = c_g.max_bwd_req_cg - c_g.max_bwd_stab_cg

    # Vertical tail sizing
    #-------------------------------------------------------------------------

    h_q.vertical_tail_sizing(aircraft)

    c_g.cg_constraint_3 = c_g.max_bwd_oei_req_cg - c_g.max_bwd_oei_cg

    return


#=========================================================================
def eval_cg_coupling(aircraft):
    """
    This function provides coupling relations to perlorm empennage sizing as a constraint satisfaction problem
    """

    c_g = aircraft.center_of_gravity

    c_g.max_fwd_req_cg = c_g.max_fwd_trim_cg
    c_g.max_bwd_stab_cg = c_g.max_bwd_req_cg
    c_g.max_bwd_oei_cg = c_g.max_bwd_oei_req_cg

    return


#=========================================================================
def eval_hq0(aircraft):
    """
    Perform hq based empennage sizing without updating characteristic masses MTOW, MLW & MZFW
    """

    # Start HQ optimization mode
    aircraft.center_of_gravity.cg_range_optimization = 1

    #=========================================================================
    def fct_hq_optim(x_in, aircraft):

        c_g = aircraft.center_of_gravity

        aircraft.wing.x_root = x_in[0]
        aircraft.horizontal_tail.area = x_in[1]
        aircraft.vertical_tail.area = x_in[2]

        # Solves geometrical coupling without tails areas
        eval_aircraft_pre_design(aircraft)
        # Just mass analysis without any solving
        eval_mass_breakdown(aircraft)
        eval_performance_analysis(aircraft)
        eval_handling_quality_analysis(aircraft)

        y_out = numpy.array([c_g.cg_constraint_1,
                             c_g.cg_constraint_2,
                             c_g.cg_constraint_3])
        return y_out
    #-------------------------------------------------------------------------

    x_ini = numpy.array([aircraft.wing.x_root,
                         aircraft.horizontal_tail.area,
                         aircraft.vertical_tail.area])

    fct_arg = (aircraft,)

    result, _, _ = newton_solve(fct_hq_optim,
                                x_ini,  # dres_dy=jac,
                                args=fct_arg)

    if (result[0] is None):
        raise Exception("Convergence problem in HQ optimization")

    aircraft.wing.x_root = result[0]
    aircraft.horizontal_tail.area = result[1]
    aircraft.vertical_tail.area = result[2]

    eval_mda0(aircraft)

    eval_handling_quality_analysis(aircraft)

    return


#=========================================================================
def eval_mda0(aircraft):
    """
    Run the design sequence with statistical empennage sizing but without satisfying mass constraints
    """

    # Solves geometrical coupling including
    eval_aircraft_statistical_pre_design(aircraft)

    # Just mass analysis without any solving
    eval_mass_breakdown(aircraft)

    eval_performance_analysis(aircraft)

    return


#=========================================================================
def eval_mda1(aircraft):
    """
    Run the design sequence with statistical empennage sizing and local mass constraints satisfaction (mass_constraint_1&2)
    but without mass-mission adaptation
    """

    # Solves geometrical coupling including tail areas statistical assessment
    eval_aircraft_statistical_pre_design(aircraft)

    # Solves internal mass coupling on MZFW and MLW
    eval_mass_estimation(aircraft)

    eval_performance_analysis(aircraft)

    return


#=========================================================================
def eval_mda2(aircraft):
    """
    Run the design sequence with mass-mission adaptation (mass_constraint_1,2&3) and statistical empennage sizing
    """

    # Solves geometrical coupling including tail areas statistical assessment
    eval_aircraft_statistical_pre_design(aircraft)

    # Solves mass coupling on MZFW, MLW and MTOW
    eval_mass_mission_adaptation(aircraft)

    eval_performance_analysis(aircraft)

    return


#=========================================================================
def eval_mda3(aircraft):
    """
    Run full MDA design with mass-mission adaptation (mass_constraint_1,2&3) and hq based empennage sizing
    """

    # Start HQ optimization mode
    aircraft.center_of_gravity.cg_range_optimization = 1

    #=========================================================================
    def fct_hq_optim(x_in, aircraft):

        c_g = aircraft.center_of_gravity

        aircraft.wing.x_root = x_in[0]
        aircraft.horizontal_tail.area = x_in[1]
        aircraft.vertical_tail.area = x_in[2]

        # Solves geometrical coupling without tail area sizing (nested solving)
        eval_aircraft_pre_design(aircraft)
        # Solves mass coupling on MZFW, MLW and MTOW      (nested solving)
        eval_mass_mission_adaptation(aircraft)
        eval_performance_analysis(aircraft)
        eval_handling_quality_analysis(aircraft)

        y_out = numpy.array([c_g.cg_constraint_1,
                             c_g.cg_constraint_2,
                             c_g.cg_constraint_3])
        return y_out
    #-------------------------------------------------------------------------

    x_ini = numpy.array([aircraft.wing.x_root,
                         aircraft.horizontal_tail.area,
                         aircraft.vertical_tail.area])

    fct_arg = (aircraft,)

    result, _, _ = newton_solve(fct_hq_optim,
                                x_ini,  # dres_dy=jac,
                                args=fct_arg)

    aircraft.wing.x_root = result[0]
    aircraft.horizontal_tail.area = result[1]
    aircraft.vertical_tail.area = result[2]

    eval_mda1(aircraft)
    eval_handling_quality_analysis(aircraft)

    return


#=========================================================================
def eval_optim_data(x_in, ac, crit_index, crit_ref, mda_type):
    """
    Compute criterion and constraints
    """

    if (ac.propulsion.architecture == "TF"):

        ac.turbofan_engine.reference_thrust = x_in[0]

    elif (ac.propulsion.architecture == "PTE1"):

        ac.turbofan_engine.reference_thrust = x_in[0]

    elif (ac.propulsion.architecture == "TP"):

        ac.turboprop_engine.reference_thrust = x_in[0]

    else:
        raise Exception("propulsion.architecture index is out of range")

    ac.wing.area = x_in[1]

    # Run MDA
    #-------------------------------------------------------------------------

    if (mda_type == "MDA2"):
        eval_mda2(ac)
    elif (mda_type == "MDA3"):
        eval_mda3(ac)
    else:
        raise Exception("Type of MDA not allowed")

    # Constraints are violated if negative
    #-------------------------------------------------------------------------
    cst = numpy.zeros(6)

    cst[0] = ac.high_speed.perfo_constraint_1
    cst[1] = ac.high_speed.perfo_constraint_2
    cst[2] = ac.low_speed.perfo_constraint_3

    cst[3] = ac.high_speed.perfo_constraint_3
    cst[4] = ac.low_speed.perfo_constraint_1
    cst[5] = ac.low_speed.perfo_constraint_2

#    omag = [ 1. , 1., 1.e-5,  1.e2, 1.e2, 1.e-2 ]

#    cst /= omag

    # All criteria have to be minimized
    #-------------------------------------------------------------------------
    crt = numpy.zeros(5)

    crt[0] = ac.weights.mtow
    crt[1] = ac.cost_mission.block_fuel
    crt[2] = ac.environmental_impact.CO2_metric
    crt[3] = ac.economics.cash_operating_cost
    crt[4] = ac.economics.direct_operating_cost

    crit = crt[crit_index] * (20. / crit_ref)

    return crit, cst


#=========================================================================
def eval_optim_cst(x_in, aircraft, crit_index, crit_ref, mda_type):
    """
    Retrieve constraints
    """

    crit, cst = eval_optim_data(x_in, aircraft, crit_index, crit_ref, mda_type)

    print("cst :", cst)

    return cst


#=========================================================================
def eval_optim_crt(x_in, aircraft, crit_index, crit_ref, mda_type):
    """
    Retreve criteria
    """

    crit, cst = eval_optim_data(x_in, aircraft, crit_index, crit_ref, mda_type)

    print("Design :", x_in)
    print("Crit :", crit)

    return crit


#=========================================================================
def mdf_process(aircraft, search_domain, criterion, mda_type):
    """
    Compute criterion and constraints
    """
    from scipy.optimize import SR1, NonlinearConstraint, BFGS

    if (aircraft.propulsion.architecture == "TF"):
        start_value = (
            aircraft.turbofan_engine.reference_thrust,
            aircraft.wing.area)
    elif (aircraft.propulsion.architecture == "PTE1"):
        start_value = (
            aircraft.turbofan_engine.reference_thrust,
            aircraft.wing.area)
    elif (aircraft.propulsion.architecture == "TP"):
        start_value = (
            aircraft.turboprop_engine.reference_thrust,
            aircraft.wing.area)
    else:
        raise Exception("propulsion.architecture index is out of range")

    if (criterion == "MTOW"):
        crit_index = 0
    elif (criterion == "block_fuel"):
        crit_index = 1
    elif (criterion == "CO2_metric"):
        crit_index = 2
    elif (criterion == "COC"):
        crit_index = 3
    elif (criterion == "DOC"):
        crit_index = 4
    else:
        raise Exception("Criterion name is unknown")

    crit_ref, cst_ref = eval_optim_data(
        start_value, aircraft, crit_index, 1., mda_type)

    res = minimize(eval_optim_crt, start_value, args=(aircraft, crit_index, crit_ref, mda_type,), method="trust-constr",
                   jac="3-point", hess=SR1(), hessp=None, bounds=search_domain, tol=1e-5,
                   constraints=NonlinearConstraint(fun=lambda x: eval_optim_cst(x, aircraft, crit_index, crit_ref, mda_type),
                                                   lb=0., ub=numpy.inf, jac='3-point'),
                   options={'maxiter': 500, 'gtol': 1e-13})
    #              tol=None, callback=None,
    #              options={'grad': None, 'xtol': 1e-08, 'gtol': 1e-08, 'barrier_tol': 1e-08,
    #                       'sparse_jacobian': None, 'maxiter': 1000, 'verbose': 0,
    #                       'finite_diff_rel_step': None, 'initial_constr_penalty': 1.0,
    #                       'initial_tr_radius': 1.0, 'initial_barrier_parameter': 0.1,
    #                       'initial_barrier_tolerance': 0.1, 'factorization_method': None, 'disp': False})

    # res = minimize(eval_optim_crt, start_value, args=(aircraft,crit_index,crit_ref,mda_type,), method="SLSQP", bounds=search_domain,
    #                constraints={"type":"ineq","fun":eval_optim_cst,"args":(aircraft,crit_index,crit_ref,mda_type,)},
    #                jac="2-point",options={"maxiter":30,"ftol":0.0001,"eps":0.01},tol=1e-14)

    # res = minimize(eval_optim_crt, x_in, args=(aircraft,crit_index,crit_ref,mda_type,), method="COBYLA", bounds=((110000,140000),(120,160)),
    #               constraints={"type":"ineq","fun":eval_optim_cst,"args":(aircraft,crit_index,crit_ref,mda_type,)},
    # options={"maxiter":100,"tol":0.1,"catol":0.0002,'rhobeg': 1.0})
    print(res)

    return res


#=========================================================================
def plot_mdf_process(aircraft, search_domain, criterion):
    """
    Compute criterion and constraints
    """

    if (aircraft.propulsion.architecture == "TF"):
        start_value = (
            aircraft.turbofan_engine.reference_thrust,
            aircraft.wing.area)
    elif (aircraft.propulsion.architecture == "PTE1"):
        start_value = (
            aircraft.turbofan_engine.reference_thrust,
            aircraft.wing.area)
    elif (aircraft.propulsion.architecture == "TP"):
        start_value = (
            aircraft.turboprop_engine.reference_thrust,
            aircraft.wing.area)
    else:
        raise Exception("propulsion.architecture index is out of range")

    if (criterion == "MTOW"):
        crit_index = 0
    elif (criterion == "block_fuel"):
        crit_index = 1
    elif (criterion == "CO2_metric"):
        crit_index = 2
    elif (criterion == "COC"):
        crit_index = 3
    elif (criterion == "DOC"):
        crit_index = 4
    else:
        raise Exception("Criterion name is unknown")

    # res = minimize(eval_optim_crt, start_value, args=(aircraft,crit_index,), method="SLSQP", bounds=search_domain,
    #                constraints={"type":"ineq","fun":eval_optim_cst,"args":(aircraft,crit_index,)},
    #                jac="2-point",options={"maxiter":30,"ftol":0.0001,"eps":0.01})

    def obj_catch(xval):
        try:
            out = eval_optim_cst(xval, aircraft, crit_index)
        except:
            out = [0, 0, 0, 0, 0, 0]
        return out

    pmot = numpy.linspace(search_domain[0][0], search_domain[0][1], 21)
    sref = numpy.linspace(search_domain[1][0], search_domain[1][1], 21)

    func_vals1 = [obj_catch([pmoti, sref[10]]) for pmoti in pmot]
    func_vals2 = [obj_catch([pmot[10], srefi]) for srefi in sref]

    from matplotlib import pylab
    pylab.plt.plot(pmot, func_vals1, label="pmot obj")
    pylab.figure()
    pylab.plt.plot(sref, func_vals2, label="sref obj")
    pylab.legend()
    pylab.show()
