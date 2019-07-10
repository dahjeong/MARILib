#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:22:21 2019

@author: DRUOT Thierry : original Scilab implementation
         ROCHES Pascal : portage to Python
"""

import numpy

from marilib.tools.math import lin_interp_1d

from marilib.earth import environment as earth

from marilib.airplane.propulsion.turbofan.turbofan_models \
    import turbofan_sfc, turbofan_thrust, turbofan_nacelle_drag, \
        turbofan_oei_drag

from marilib.airplane.propulsion.hybrid_pte1.hybrid_pte1_models \
    import hybrid_sfc, hybrid_thrust, electric_nacelle_drag


#===========================================================================================================
def sfc(aircraft,pamb,tamb,mach,rating,nei):
    """
    Bucket SFC for a turbofan
    """

    propulsion = aircraft.propulsion

    if (propulsion.architecture=="TF"):

        sfc = turbofan_sfc(aircraft,pamb,tamb,mach,rating,nei)

    elif (propulsion.architecture=="PTE1"):

        sfc = hybrid_sfc(aircraft,pamb,tamb,mach,rating,nei)

    else:
        raise Exception("propulsion.architecture index is out of range")

    return sfc


#===========================================================================================================
def thrust(aircraft,Pamb,Tamb,Mach,rating,nei):
    """
    Calculation of thrust for pure turbofan airplane
    Warning : ALL engine thrust returned
    """

    propulsion = aircraft.propulsion

    if (propulsion.architecture=="TF"):

        fn,data = turbofan_thrust(aircraft,Pamb,Tamb,Mach,rating,nei)

    elif (propulsion.architecture=="PTE1"):

        fn,sec,data = hybrid_thrust(aircraft,Pamb,Tamb,Mach,rating,nei)

    else:
        raise Exception("propulsion.architecture index is out of range")

    return fn,data


#===========================================================================================================
def nacelle_drag(aircraft,Re,Mach):
    """
    All nacelle drag coefficient
    """

    propulsion = aircraft.propulsion

    if (propulsion.architecture=="TF"):

        nacelle = aircraft.turbofan_nacelle
        nacelle_cxf,nacelle_nwa = turbofan_nacelle_drag(aircraft,nacelle,Re,Mach)

    elif (propulsion.architecture=="PTE1"):

        nacelle = aircraft.turbofan_nacelle
        t_nacelle_cxf,t_nacelle_nwa = turbofan_nacelle_drag(aircraft,nacelle,Re,Mach)

        nacelle = aircraft.electric_nacelle
        e_nacelle_cxf,e_nacelle_nwa = electric_nacelle_drag(aircraft,nacelle,Re,Mach)

        nacelle_cxf = t_nacelle_cxf + e_nacelle_cxf
        nacelle_nwa = t_nacelle_nwa + e_nacelle_nwa

    else:
        raise Exception("propulsion.architecture index is out of range")

    return nacelle_cxf,nacelle_nwa


#===========================================================================================================
def oei_drag(aircraft,pamb,tamb):
    """
    Inoperative engine drag coefficient
    """

    propulsion = aircraft.propulsion

    if (propulsion.architecture=="TF"):

        nacelle = aircraft.turbofan_nacelle
        dcx = turbofan_oei_drag(aircraft,nacelle,pamb,tamb)

    elif (propulsion.architecture=="PTE1"):

        nacelle = aircraft.turbofan_nacelle
        dcx = turbofan_oei_drag(aircraft,nacelle,pamb,tamb)

    else:
        raise Exception("propulsion.architecture index is out of range")

    return dcx


#===========================================================================================================
def thrust_pitch_moment(aircraft,fn,pamb,mach,dcx_oei):

    propulsion = aircraft.propulsion
    wing = aircraft.wing

    gam = earth.heat_ratio()

    if (propulsion.architecture=="TF"):

        nacelle = aircraft.turbofan_nacelle

    elif (propulsion.architecture=="PTE1"):

        nacelle = aircraft.turbofan_nacelle

    else:
        raise Exception("propulsion.architecture index is out of range")

    cm_prop = nacelle.z_ext*(dcx_oei - fn/(0.5*gam*pamb*mach**2*wing.area))

    return cm_prop


#===========================================================================================================
def thrust_yaw_moment(aircraft,fn,pamb,mach,dcx_oei):
    """
    Assumed right engine inoperative
    """

    propulsion = aircraft.propulsion
    wing = aircraft.wing

    gam = earth.heat_ratio()

    if (propulsion.architecture=="TF"):

        nacelle = aircraft.turbofan_nacelle

    elif (propulsion.architecture=="PTE1"):

        nacelle = aircraft.turbofan_nacelle

    else:
        raise Exception("propulsion.architecture index is out of range")

    cn_prop = (nacelle.y_ext/wing.mac)*(fn/(0.5*gam*pamb*mach**2*wing.area) + dcx_oei)

    return cn_prop


#===========================================================================================================
def tail_cone_drag_effect(aircraft):
    """
    Effect of rear fan installation on fuselage tail cone drag
    """

    propulsion = aircraft.propulsion

    if (propulsion.architecture=="TF"):

        fac = 1.

    elif (propulsion.architecture=="PTE1"):

        dfus = aircraft.fuselage.width
        dhub = aircraft.electric_nacelle.hub_width

        fac = (1.-(dhub/dfus)**2)

    else:
        raise Exception("propulsion.architecture index is out of range")

    return fac

