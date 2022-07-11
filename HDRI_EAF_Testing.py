import os
import sys
sys.path.append("C:/Users/CKIEFER/GreenSteel/HDRI-EAF-Technoeconomic-model")
sys.path.append("C:/Users/CKIEFER/NREL/HOPP/examples")
import Enthalpy_functions
from NREL.HOPP.examples.H2_Analysis import h2_main
import pandas as pd 
import matplotlib.pyplot as plt
#plt.rcParams['savefig.dpi'] = 1200
#plt.rcParams["figure.figsize"] = [10,10]
import numpy as np
import numpy_financial as npf #for calculating the NPV and IRR 

#import SALib 

def establish_save_output_dict():
    """
    Establishes and returns a 'save_outputs_dict' dict
    for saving the relevant analysis variables for each site.
    """

    save_outputs_dict = dict()
    save_outputs_dict['Net Present Value'] = list()
    save_outputs_dict['Internal Rate of Return'] = list()
    save_outputs_dict['H2 Production per year'] = list()
    save_outputs_dict['Electrolyzer Capacity'] = list()
    save_outputs_dict['Total Electricity'] = list()
    save_outputs_dict['Total Electricity Price'] = list()
    save_outputs_dict['Plant Life'] = list()
    save_outputs_dict['Lang Factor'] = list()
    save_outputs_dict['Emissions'] = list()
    save_outputs_dict['Yearly Steel Production'] = list()
    save_outputs_dict['Total Revenue'] = list()
    save_outputs_dict['Total Costs'] = list()
    save_outputs_dict['Total Incomce'] = list()

    return save_outputs_dict

##Code is for Hydrogen Driven Plant##
def HDRI_EAF_Model(eta_el,h2_prod_yr,plant_life,tax_rate,interest_rate,electricity_cost,iron_ore_cost,emission_cost,carbon_steel_price,O2_price,el_spec,h2_investment_2020,emission_factor,elec_limit):
    #molecular weights of the materials fed into DRI from national institute of standards and technology
    #Iron and impurities
    mol_weight_fe=      55.845  #grams/mol
    mol_weight_fe203=   159.69  #grams/mol
    mol_weight_sio2=    60.084  #grams/mol
    mol_weight_al2o3=   101.961 #grams/mol
    mol_weight_feo=     71.84   #grams/mol
    mol_weight_H2=      2.01588 #grams/mol
    mol_weight_H2O=     18.0153 #grams/mol
    mol_weight_cao=     56.08   #grams/mol
    mol_weight_mgo=     40.30   #grams/mol
    mol_weight_C=       12.011  #grams/mol




    # Enthalpy calculations of the different streams
    ##Equations and values based of NIST equation --->https://webbook.nist.gov/cgi/cbook.cgi?Source=1998CHA1-1951&Mask=2

    def H2_enthalpy_1(T):   #H2 Temp of 0 C / 298 K. Returns KJ/g
        t=T/1000
        A=33.066178
        B=-11.363417
        C=11.432816
        D=-2.772874
        E=-0.158558
        F=-9.980797
        G=172.707974
        H=0
        H_t=(A*t)+(B*(t**2)/2)+(C*(t**3)/3)+(D*(t**4)/4)-(E/t)+(F-H)/mol_weight_H2
        return H_t
    def H2_enthalpy_2(T):   #H2 Temp 900 C / 1173 K. Returns KJ/g
        t=T/1000
        A=18.563083
        B=12.257357
        C=-2.859786
        D=0.0268238
        E=1.977990
        F=-1.147438
        G=156.288133
        H=0
        H_t=((A*t)+(B*(t**2)/2)+(C*(t**3)/3)+(D*(t**4)/4)-(E/t)+(F-H))/mol_weight_H2
        return H_t
    def H2O_enthalpy(T):    #h2o gas phase 500 K to 1700 K. Returns KJ/g
        t=T/1000
        A=30.09200
        B=6.832514
        C=6.793435
        D=-2.534480
        E=0.082139
        F=-250.8810
        G=223.3967
        H=-241.8264
        H_t=((A*t)+(B*(t**2)/2)+(C*(t**3)/3)+(D*(t**4)/4)-(E/t)+(F-H))/mol_weight_H2O
        return H_t
    def fe_enthalpy_1(T):   #solid fe at any temp from 298 K to 1809 K, melting temp. Returns KJ/g
        t=T/1000
        A=23.97449
        B=8.367750
        C=0.000277
        D=-0.000086
        E=-0.000005
        F=0.268027
        G=62.06336
        H=7.788015
        H_t=((A*t)+(B*(t**2)/2)+(C*(t**3)/3)+(D*(t**4)/4)-(E/t)+(F-H))/mol_weight_fe
        return H_t 
    def fe_enthalpy_2(T):   #Liquid Fe enthalpy any temp above 1809 K. Returns KJ/g
        t=T/1000
        A=46.02400
        B=-1.884667*10**(-8)
        C=6.094750*10**(-9)
        D=-6.640301*10**(-10)
        E=-8.246121*10**(-9)
        F=-10.80543
        G=72.54094
        H=12.39502
        H_t=((A*t)+(B*(t**2)/2)+(C*(t**3)/3)+(D*(t**4)/4)-(E/t)+(F-H))/mol_weight_fe
        return H_t
    def feo_enthalpy(T):    #temp form 298 to 1650 K. Returns KJ/g
        t=T/1000
        A=45.75120
        B=18.78553
        C=-5.952201
        D=0.852779
        E=-0.081265
        F=-286.7429
        G=110.3120
        H=-272.0441
        H_t=((A*t)+(B*(t**2)/2)+(C*(t**3)/3)+(D*(t**4)/4)-(E/t)+(F-H))/mol_weight_feo
        return H_t
    def al2o3_enthalpy(T):  #diffenent values presented. Returns KJ/g
        t=T/1000
        A=106.0880
        B=36.33740
        C=-13.86730
        D=2.141221
        E=-3.133231
        F=-1705.970
        G=153.9350
        H=-1662.300
        H_t=((A*t)+(B*(t**2)/2)+(C*(t**3)/3)+(D*(t**4)/4)-(E/t)+(F-H))/mol_weight_al2o3
        return H_t
    def sio2_enthalpy(T):   #sio2 Temp above 847 k. Returns KJ/g
        t=T/1000
        A=58.75340
        B=10.27925
        C=-0.131384
        D=0.025210
        E=0.025601
        F=-929.3292
        G=105.8092
        H=-910.8568
        H_t=((A*t)+(B*(t**2)/2)+(C*(t**3)/3)+(D*(t**4)/4)-(E/t)+(F-H))/mol_weight_sio2
        return H_t
    def mgo_enthalpy(T):    #temp from 298 K to 3105 K. Returns KJ/g
        t=T/1000
        A=47.25995
        B=5.681621
        C=-0.872665
        D=0.104300
        E=-1.053955
        F=-619.1316
        G=76.46176
        H=-601.2408
        H_t=((A*t)+(B*(t**2)/2)+(C*(t**3)/3)+(D*(t**4)/4)-(E/t)+(F-H))/mol_weight_mgo
        return H_t
    def cao_enthalpy(T):    #temp from 298 K - 3200 K. Returns KJ/g
        t=T/1000
        A=49.95403
        B=4.887916
        C=-0.352056
        D=0.046187
        E=-0.825097
        F=-652.9718
        G=92.56096
        H=-635.0894
        H_t=((A*t)+(B*(t**2)/2)+(C*(t**3)/3)+(D*(t**4)/4)-(E/t)+(F-H))/mol_weight_cao
        return H_t




    #Temperatures in System all in Kelvin
    T1= 298     #Pellets entering DRI
    T2= 973     #95% reduced Iron exiting DRI/Entering EAF !!Assuming 0 heat Losses!!
    T3= 1923    #Liquid Steel exiting EAF
    T4= 1173    #100% H2 exiting heater/entering DRI/ Saft Temperature
    T5= 573     #H2/H20 stream exiting DRI/entering recuperator
    T6= 298     #Carbon Fines entering EAF
    T7= 298     #Lime/Slag Formers entering EAF
    T8= 1923    #slag exiting EAF
    T9= 1773    #exhaust has exiting EAF
    T10=343     #H2O exiting Electrolyzer/entering recuperator
    T11=443     #H2 Exiting recuperator/entering heater
    T12=393     #H2O exiting recuperator/entering condenser
    T13=343     #H2O exiting condenser/entering electrolyzer
    T14=298     #O2 exiting electrolyzer/entering EAF




    ##Masses, Metallization rate
    alpha=0.94          #Metallization rate of DRI
    Fe2O3_pure=0.95     #ammount of Fe2O3 in raw material. accounts for 5% impurities in raw material

    Fe_O_ratio=((2*mol_weight_fe)/mol_weight_fe203)     #ratio of fe weight to fe2o3 weight

    m3=1000             #kg per metric tonne steel

    m1=m3/(Fe2O3_pure*Fe_O_ratio*alpha)     #amount of raw material needed for tonne steel, 
    m2_feo=(m1*Fe2O3_pure*Fe_O_ratio*(1-alpha))         #amount of IronOxide exiting DRI

    sio2_percent=.03    #percent sio2 in raw materials
    al2o3_percent=.02   #percent al2o3 in raw materials

    m1_sio2=(sio2_percent*m1)       #mass of sio2 in raw materials kg per tonne steel
    m1_al2o3=(al2o3_percent*m1)     #mass of al2o3 in raw materials kg per tonne steel
    m2_fe=(m1-(m1_sio2+m1_al2o3+m2_feo))*Fe_O_ratio     #mass metallic iron out of DRI per tonne steel


    #print("Raw Iron Ore input in (Kg/tls): ",np.around(m1,2))
    #print("Metallic Stream at SF Outlet in (kg/tls): ",np.around(m2_fe,2))
    #print("Molten Metal at EAF Outlet with no reduction in EAF (kg/tls): ",np.around(m3,2))



    ##Stoichiometric Calculations of DRI

    H2_per_mol=3/2         #3 mol of H2 per 2 mol Fe
    H2_weight_per_mol=(H2_per_mol*mol_weight_H2)*1000   #Kg h2 per 1 mol fe
    mol_per_ton_fe=(1000)/mol_weight_fe                 #mols of iron in a 1000 kg of fe
    m4_stoich=(H2_weight_per_mol*mol_per_ton_fe)/1000   #stoich minimum mass of H2 needed per tonne steel

    lambda_h2=1.2           #Extra H2 is needed. Lambda is ratio of actual over stoichiometric requirement

    m4=m4_stoich*lambda_h2  #mass of hydrogen inputted into DRI in Kg per tonne steel

    


    #Hydrogen Plant data
    #h2_prod_yr=200000000 #in kg per year
    """"""
    #####input h2_main results for h2 production a year here####
    
    #h2_prod_yr=h2_main 

    """"""
    operating_hours=365*24*.95 #operating 95% of the time

    h2_prod_hr=h2_prod_yr/operating_hours  #h2 produced per hour


    ##Calculations of amount of produced h2 as excess h2 from DRI will be recycled into DRI
    #steel produced per hour

    h2_per_ton_actual=m4_stoich*lambda_h2       #mass of hydrogen inputted in Kg per tonne steel
    m4=h2_per_ton_actual                        #mass of hydrogen inputted in Kg !!unsure why original author put this in twice, returns same value
    steel_per_hr=h2_prod_hr/h2_per_ton_actual   #Steel prduced per hour based off h2 restriction

    #print("Hydrogen at SF inlet :",m4)

    #steel plant Data
    plant_life=20 #years
    lang_factor=3   #estimated ratio of the total cost of creating a process within a plant, to the cost of all major technical components

    steel_prod_yr=steel_per_hr*operating_hours  #total steel produced in a year in tonnes


    O2_sold=.6 #assumed 60% oxygen sold to market


    ##Enthalpy of hydrogen entering DRI Shaft
    #shaft temperature=T4

    if T4<1000:
        h4=(H2_enthalpy_1(T4)*m4*1000)  #in kJ/tls
        h4_kwh=h4/3600      #3600 kJ in 1 kWh
    elif T4>1000:
        h4=(H2_enthalpy_2(T4)*m4*1000)  #in kJ/tls
        h4_kwh=h4/3600      #3600 kJ in 1 kWh


    ##Mass flow rate of waste gas from DRI Shaft
    #Mass flwo rate of water in the waste stream per tonne of steel

    water_tls=((3*mol_weight_H2O)/(2*mol_weight_fe))*1000  #water produced per tonne liquid steel

    ##Hydrogen in waste stream
    m5_h2=(m4_stoich*(lambda_h2-1))  #mass of hyrdogen remaing in exhaust after DRI in kg
    m5_h20=water_tls        #Mass water produced in DRI in kg
    m5=(m5_h2+m5_h20)       #Total Mass of exhaust in kg


    #print("SF Exhaust Stream :",m5)


    ##Exhaust Entholpy
    #Temp of exhaust = T5

    h5_h20=(m5_h20*H2O_enthalpy(T5)*1000)   #kJ/tls Enthalpyof h20 in exhaust
    h5_h2=(m5_h2*H2_enthalpy_1(T5)*1000)    #kJ/tls Enthalpy of H2 in exhaust
    h5=(h5_h20+h5_h2)                       #kJ/tls Enthalpy of exhaust
    h5_kwh=h5/3600                          #Conversion Kj to Kwh


    #Reaction enthalpy Calculation

    h_activation=35     #Kj/mol activation energy
    h_endothermic=99.5  #kJ/mol reaction energy absorbed

    h_reaction=h_activation+h_endothermic #energy needed per 1 mol
    h_reaction_total=(h_reaction*m1*1000*Fe2O3_pure)/mol_weight_fe203  #total energy per tonne steel
    h_reaction_total_kwh=h_reaction_total/3600  #kJ to kwh conversion

    #print("Reaction Enthalpy of liquid steel :",h_reaction_total_kwh)


    ##Energy in solid stream exiting DRI
    #Exit temp is T2

    h2=1000*((fe_enthalpy_1(T2)*m2_fe)+(feo_enthalpy(T2)*m2_feo)
        +(sio2_enthalpy(T2)*m1_sio2)+(al2o3_enthalpy(T2)*m1_al2o3))  #Enthalpy of metallic stream exiting DRI/exiting EAF in kj/kg

    h2_kwh=h2/3600      #conversion kj to kwh

    #print("Metallic stream at SF outlet :",h2_kwh)

    #Energy balance of the shaft furnace
    Q_dri=((h4)-(h5+h2+h_reaction_total))/3600  #Enthalpy of Hydrogen minus enthalpy of exhaust, metal stream, reaction enthalpy
    ##Gives out Negative value so heat is leaving the system, does not need electrical heat at input H2 Temp
    if Q_dri > 0:
        eta_el_dri=.6
        el_dri=Q_dri/eta_el_dri
    

    ##Heat Recuperator
    ##Exhaust Gas is at T5
    ##T12 is H2O from recuperator high enough that there is no condensation
    ##T13 is Condensed h20
    ##T10 is H2 into recuperator

    #T5=573 #Exhaust from DRI shouldbe T5
    #T12=393  #recuperator to condenser T12

    #T10=343  #hydrogen from electrolyzer T10
    ratio_burner=0.80  #not used in authors code
    m10=m4          #hydrogen from electrolyzer = hydrogen into DRI

    m12_h20=m5_h20  #Mass h2o of exhaust stream = mass h2o to condensor 
    m12_h2=m5_h2    #mass h2 of exhaust stream = mass h2
    m13=m12_h20     #mass h2o from condenser to electrolyzer

    h12_h2o=(m12_h20*H2O_enthalpy(T12)*1000)   #exit h20 in recuperator to condenser stream
    h12_h2=(m12_h2*H2_enthalpy_1(T12)*1000)    #exit h2 in recuperator to condenser stream
    h12=(h12_h2+h12_h2o)                       #total enthalpy in exit of recuperator to condensor
    h12_kwh=h12/3600                           #conversion kj to kwh
    h10=(H2_enthalpy_1(T10)*m10*1000)          #electrolyzer to recuperator
    h10_kwh=(h10/3600)                         #conversion kj to kwh

    #print('Mass H2O leaving DRI in exhaust in (Kg/tls): ',np.around(m12_h20,3))

    h11=((h5-h12)+h10)                          #remaining h2 enthalpy into heater


    #T9=413  !!!!!Authours Old Code, Cleaned up to match models
    #m9=m10
    #h9=(H2_enthalpy_1(T9)*m9*1000)
    #h9_kwh=(h9/3600)
    #eta_rec=((h9-h10)/(h5-h12))
    #Q_heater=(h4-h9)/3600


    ###Heater inlet
    T11_heat_in=413      #Assumes 30 degree heat loss from recuperator to heater

    m11_heat_in=m10     #mass of hydrogen into heater = mass hydrogen into recuperator
    h11_heat_in=(H2_enthalpy_1(T11_heat_in)*m11_heat_in*1000)  #enthalpy of stream into heater
    h11_heat_in_kwh=(h11_heat_in/3600)  #kj to kwh conversion
    eta_rec=((h11_heat_in-h10)/(h5-h12))  #recuperator efficiency

    Q_heater=(h4-h11_heat_in)/3600  #energy needed to be provided to heater


    eta_el_heater=.6        #Efficiency of heater

    el_heater=(Q_heater/eta_el_heater)  #electricity need at heater

    #print("Electricity of Heater needed in kwh :", el_heater)


    #####Not necessary for code

    ##EAF Carbon input
    m6=10 #kg
    m7=50 #kg
    m7_cao=.75*m7
    m7_mgo=.25*m7

    ##Solution enthalpy for carbon = 0.62 kwh/kgc
    H_carbon=.62*m6
    reduction_enthalpy=3.59 #kwh/kg Carbon
    m2_feo_reduced=(m2_feo*.7)  #assumed 70% feo in EAF gets reduced
    c_required=(mol_weight_C/mol_weight_feo)*m2_feo_reduced
    H_feo_red=c_required*reduction_enthalpy
    c_remaining=m10-c_required
    # Oxidation of carbon to carbon monoxide consider that remaining carbon is converted to CO
    # reaction C+0.5 O2---- CO + 9.10 kWh/kg of carbon
    H_co=c_remaining*9.10 

    m3_actual=m2_feo_reduced+m3
    #print('Extra Molten Metal through reduction in EAF (Kg/tls): ',np.around(m2_feo_reduced,2))
    #print('Actual tonnes liquid steel with reduction in EAF: ',np.around(m3_actual/1000,2))

    def electrolyzer_npv(eta_el,tax_rate,interest_rate,electricity_cost,iron_ore_cost,emission_cost,carbon_steel_price,O2_price,el_spec,h2_investment_2020,emission_factor):
        save_outputs_dict = establish_save_output_dict()
        ## Electrolyzer : Mass and Energy flow
        #elec_spec=50 #kwh/kgh2  #specification of electrolyzer
        water_spec=11   #11 kg of water is required for 1 kg h2 #1kg h2o = 1 liter h2o

        el_elec=(m4*el_spec)  #Electrolyzer Electricity
        water_total=(m4*water_spec) ##total water in system needed for 1 tls in kg h20/tls
        extra_h2o=(water_total-m13) #total water needed into electrolyzer if h2o is recycled

        #print('Mass H2O needed to be inputted into electrolyzer (kg/tls): ',np.around(extra_h2o,2))
        #print('Total Mass H2O in system (kg/tls): ',np.around(water_total,3))

        #### electric arc furnace mass and energy balance

        ##energy to melt Fe in the Eaf
        Hfe_melting=247 #kJ/kg
        Hfe_T2=fe_enthalpy_1(T2)   #Enthalpy of DRI entering Eaf
        Hfe_T3=fe_enthalpy_2(T3)    #Enthalpy steel exiting EAF
        h3=((Hfe_T3-Hfe_T2)*m2_fe*1000)+(m2_fe*Hfe_melting)  #Total Enthalpy at output

        h3_kwh=h3/3600


        #eta_el=0.6  #Efficiency of the transformer, arc,
        #heat transfer, cooling losses, waste gas stream taken into consideration
        # The efficincy is considered lower to account for the loss of energy from the
        #scrap stream, the use of slag formers etc

        el_eaf=h3_kwh/eta_el
        #print ("Electrical energy input in kWh/ton of liquid steel",el_eaf)

        #Specific energy consumption of HDRI-EAF system

        EL_total=(el_eaf+el_elec+el_heater) #kwh/tls
        El_total_yr=EL_total*steel_prod_yr #kwh/yr
        EL_total_MWh=EL_total/1000  #Mwh/tls
        EL_total_MWh_yr=EL_total_MWh*steel_prod_yr
        #print('Steel Production per Year (kg): ',steel_prod_yr)                                                                                                                                                                                                                                                                                                                                           
        #print('Electrical Demands per tls: ', EL_total)
        #print('Yearly Electrical demands (kw): ',np.around(El_total_yr))
        #print('Steel Production ins tonnes: ',steel_prod_yr/1000)
        #print('Electrical demands yearly (MW): ',EL_total_MWh_yr)
        # Specific CO2 emissions from HDRI-EAF System

        #emission_factor=0.413 # corresponds to emissions from the grid
        EAF_co2=0.050 #ton/tls
        cao_emission=0.056 #tco2/tls
        co2_eaf_electrode=0.0070 #tco2/tls
        pellet_production=0.12 #tco2/tls 
        direct_emissions=EAF_co2+cao_emission+co2_eaf_electrode+pellet_production
        indirect_emissions_eaf=(el_eaf*emission_factor)/1000 #tonnes
        indirect_emissions_el=(el_elec*emission_factor)/1000 #tonnes
        total_emission=direct_emissions+indirect_emissions_eaf+indirect_emissions_el
        
        #print('Total Emissions (CO2/tls): ',total_emission)
        # input parameters for NPV calculation

        # Plant is operating 95% of the year
        operating_hours=365*24*.95
        steel_per_hr=steel_prod_yr/operating_hours
        steel_prod_yr_Mil=steel_prod_yr/10**6
        h2_per_hour_kg=steel_per_hr*m4
        h2_per_second_kg=(h2_per_hour_kg)/3600
        lhv_h2=120.1 #Mj/kg low heating value
        h2_capacity_MW=h2_per_second_kg*lhv_h2  
        electrolyzer_efficiency=0.67
        el_capacity_mwel=h2_capacity_MW/electrolyzer_efficiency
        h2_per_year=h2_per_hour_kg*operating_hours
        #print('Hydrogen Production per year: ',h2_per_year) #check
        
        #if el_capacity_mwel>=1000:
            #el_capacity_gwel=el_capacity_mwel/10**3
            #print('Electrolyzer Capacity (GW): ',np.around(el_capacity_gwel,2))
        #elif el_capacity_mwel<1000:
            #print('Electrolyzer Capacity (MW): ',np.around(el_capacity_mwel,2))

        #print('Total Steel Produced per year in tonnes: ',np.around(steel_prod_yr_Mil,2))
        #print("Hydrogen Capacity in MW: ",h2_capacity_MW)
        #print("Electrolyzer El in MW: ",el_capacity_mwel)


        stack_lifetime=90000 # hours
    
        stack_replacement_year=stack_lifetime/operating_hours
        stack_replacement_number=1
        Euro_dollar_conversion=1.18
        ## 600 USD /KW

        h2_investment_MW_h2=(h2_investment_2020*Euro_dollar_conversion)

        electrolyer_cost=h2_investment_MW_h2*h2_capacity_MW
        # only the stack is replaced, which is 60% of the total el capital cost 
        # it is assumed that by the time stacks are replaced the cost of electrolyzers would 
        #have fallen to 0.45 Million Euro/MW of H2
        # replacement is considered only once for the plant as its assumed that the lifetime of 
        #next generation stacks would be more than 15 years 

        h2_investment_2030=.45

        #0.6 represents 60% of the stack needing to be replaced after lifetime
        percent_stack_replaced=.6
        replacement_cost=electrolyer_cost*percent_stack_replaced*h2_investment_2030*np.round(stack_replacement_number)
        total_electrolyzer_cost=(electrolyer_cost+replacement_cost)


        # The value includes installation costs 
        #https://iea-etsap.org/E-TechDS/PDF/I02-Iron&Steel-GS-AD-gct.pdf
        eaf_cost_per_ton_yr=140  #USD/ton/per year
        eaf_total_cost=(eaf_cost_per_ton_yr*steel_prod_yr)/10**6 #million USD

        #print('Total EAF cost in Million USD over plant life: ',np.around(eaf_total_cost,0))
        
        # Cost of new DRI plant with 2 MT/annum =5.5 million euros or 6 million USD
        # A reduction in cost is considered as larger plants have smaller costs.
        # The plant costs include all the costs including ISBL, OSBL, area development etc 
        # no Lang factor multiplication is required for such a plant
        #https://www.voestalpine.com/group/en/media/press-releases/2013-07-04-
        #voestalpine-entrusts-construction-of-the-direct-reduction-plant-in-texas-to-siemens-and-midrex/

        dri_cost_per_ton_yr=240  #USD/ton/per year
        dri_total_cost=(dri_cost_per_ton_yr*steel_prod_yr)/10**6 #million USD


        ##pressure swing adsorber
        #accounts fot plant equipment like condenser, heat exhanger, heater
        #million USD
        total_capital_cost=((electrolyer_cost)+eaf_total_cost+dri_total_cost)*lang_factor #million USD
        
        ##Doesn't include costs of indirect emissons
        total_emission_cost=((EAF_co2+cao_emission+co2_eaf_electrode+pellet_production)*emission_cost*steel_prod_yr)/10**6 #million USD
        #print(total_emission_cost)
        #operational costs
        eaf_op_cost_tls=32 #t/yr of dri
        dri_op_cost_tls=13 #t/yr of dri
        #ng_cost=((ng_price/21)*natural_gas_kg)# converting price/MMBTU to price/kg
        iron_ore_tls=m1/1000    # Converting iron ore required from kg to tonnes
        iron_ore_cost_tls=iron_ore_tls*iron_ore_cost
        electricity_cost_tls=EL_total_MWh*electricity_cost
        operational_cost_annual=((iron_ore_cost_tls+electricity_cost_tls+eaf_op_cost_tls+dri_op_cost_tls)*steel_prod_yr)/10**6 #million USD


        el_eaf_yr=el_eaf*steel_prod_yr
        el_elec_yr=el_elec*steel_prod_yr
        el_heater_yr=el_heater*steel_prod_yr
        #print('Yearly EAF electricity required (Kw): ',el_eaf_yr)
        #print('Yearly Electrolyzer electricity required (kw): ',el_elec_yr)
        #print('Yearly Heater electricity required (kw): ',el_heater_yr)

        #print('Electricity required in the Electric Arc Furnace in (kwh/tls): ',np.around(el_eaf,2))
        #print('Electricity required in the Electrolyzer in (kwh/tls): ',np.around(el_elec,3))
        #print('Electricity required in the Heater in (kwh/tls): ',np.around(el_heater,2))
        #print('Total Electricity in (MWh): ',np.around(EL_total_MWh,2))
        electricity_cost_total=electricity_cost_tls*steel_prod_yr/10**6
        #print('Electricity Cost in Millions USD: ',np.around(electricity_cost_total))
        ##maintenance cost
        maintenance_cost_percent=0.05   #5% of capitol cost
        labour_cost_tls=40 #USD/tls

        maintenance_cost_yr=(total_capital_cost*maintenance_cost_percent)
        labour_cost_yr=(labour_cost_tls*steel_prod_yr)/10**6 #Million USD

        depreciation_yr=total_capital_cost/plant_life

        #Revenue  USD/ton
        O2_produced_tls=m4*8 ##8 because of 2:1 mol ratio and 2:16 molecular weight ratio  kg
        total_O2_produced=(O2_produced_tls*steel_prod_yr)/1000 #tonnes
        #print(O2_produced_tls)
        O2_revenue=(total_O2_produced*O2_price*O2_sold) #USD
        total_revenue=((carbon_steel_price*steel_prod_yr)+O2_revenue)/10**6 #Million USD

        cost_tls=[iron_ore_cost_tls+electricity_cost_tls+eaf_op_cost_tls+dri_op_cost_tls+labour_cost_tls+eaf_op_cost_tls+dri_op_cost_tls
                +(direct_emissions*emission_cost)]###Double check these
        
        ## NPV and IRR calculations
        years=np.arange(0,plant_life,1)
        Years=np.round(years,0)
        production_steel=np.repeat(steel_prod_yr/10**6,plant_life).tolist()
        production_hydrogen=np.repeat(h2_prod_yr/10**6,plant_life).tolist()

        capitol_cost_yr=np.repeat(0,plant_life).tolist()
        capitol_cost_yr[0]=((total_capital_cost-replacement_cost)/3)
        capitol_cost_yr[1]=((total_capital_cost-replacement_cost)*(2/3))
        capitol_cost_yr[11]=replacement_cost
        operational_cost=np.repeat(operational_cost_annual,plant_life).tolist()
        maintenance_cost=np.repeat(maintenance_cost_yr,plant_life).tolist()
        labour_cost=np.repeat(labour_cost_yr,plant_life).tolist()
        emission_cost=np.repeat(total_emission_cost,plant_life).tolist()
        depreciation_cost=np.repeat(depreciation_yr,plant_life).tolist()
        total_revenue=np.repeat(total_revenue,plant_life).tolist()
        
        
        
        #print('Total Income of plant in Million USD: ',np.around(income,2))
        
        for i in range(2):
            production_steel[i]=0
            operational_cost[i]=0
            labour_cost[i]=0
            maintenance_cost[i]=0
            emission_cost[i]=0
            depreciation_cost[i]=0
            total_revenue[i]=0

            #tax calculation

            tax_cost=[(total_revenue[i]-(capitol_cost_yr[i]+operational_cost[i]+labour_cost[i]
                        +maintenance_cost[i]+emission_cost[i]+depreciation_cost[i]))*tax_rate  for i in range(len(operational_cost))]

            tax_cost[0]=0
            tax_cost[1]=0

            cash_flow=[total_revenue[i]-(capitol_cost_yr[i]+operational_cost[i]+labour_cost[i]
                        +maintenance_cost[i]+emission_cost[i]+tax_cost[i]) for i in range(len(total_revenue))]

        costs_cash_flow=[(capitol_cost_yr[i]+operational_cost[i]+labour_cost[i]
                        +maintenance_cost[i]+emission_cost[i]+tax_cost[i]) for i in range(len(total_revenue))]
        #print(costs_cash_flow)
        npv_hdri=npf.npv(interest_rate,cash_flow)
        irr=npf.irr(cash_flow)
       
        annual_cost_capital = npf.pmt(interest_rate, plant_life, -total_capital_cost)
        annual_cost_yr=operational_cost_annual+labour_cost_yr+total_emission_cost+maintenance_cost_yr+annual_cost_capital
        lcos_annual= (annual_cost_yr*10**6)/(steel_prod_yr) #$/tonne Levelized Cost of Steel using annual 
        #print(annual_cost_yr*10**3)
        #print(steel_prod_yr/10**3)
        #print('Levelized Cost of Steel annual: ($/tls)',lcos_annual)
        
        #total_costs=np.sum(capitol_cost_yr)+np.sum(operational_cost)+np.sum(maintenance_cost)+np.sum(labour_cost)+np.sum(emission_cost)
        #total_steel_prod=steel_prod_yr*plant_life
        #lcos_total=(total_costs)/total_steel_prod
        #print(total_costs)
        #print(total_steel_prod)
        #print('Levelized Cost per Steel total: ',lcos_total)
        

        #npv_total_costs=npf.npv(interest_rate,costs_cash_flow)
        #lcos=(npv_total_costs*10**6)/(steel_prod_yr)
        #print('Levelized Cost of Steel: ($/tls)',lcos)
       
        annual_cost_capital_elec=npf.pmt(interest_rate,plant_life,-(electrolyer_cost*lang_factor))
        annual_cost_yr_elec=annual_cost_capital_elec+(maintenance_cost_percent*annual_cost_capital_elec)+(el_elec*steel_prod_yr*(electricity_cost/1000))
        lcoh=(annual_cost_yr_elec)/h2_per_year #$/kg Levelized Cost of Hydrogen
        #print(el_elec)
        #print(electricity_cost_tls)
        #print(electricity_cost)
        
        #print(maintenance_cost_percent)
        #print(annual_cost_capital_elec)
        #print(annual_cost_yr_elec)
        #print('H2 per year',h2_per_year)
        #print('Levelized Cost of Hydrogen (USD/kg): ',lcoh)
        #print(cash_flow)
        #print('Net Present Value: (testing) ',npv_hdri)
        #print(total_emission_cost)

        #print('Steel Produced per year: ',steel_prod_yr)
        return [npv_hdri,irr,cash_flow,lcos_annual,El_total_yr,lcoh,steel_prod_yr]

    
    
    #######Sensitivity analysis#######
    #from SALib.sample import saltelli
    #from SALib.analyze import sobol

    #problem = {
    #    'num_vars': 9,
    #    'names': ['tax rate', 'interest_rate', 'eletrcity_cost', 'iron_ore_cost', 
    #    'emission_cost', 'carbon_steel_price', 'O2_price', 'elec_spec','h2_investment_2020'],
    #  'bounds': [
    #  [0.25,0.35],      
    #  [0.06,0.12],
    #  [20,60],
    #  [75,95],
    #  [40,100],
    #  [600,700],
    #  [20,40],
    #  [40,55],
    #  [0.4,0.7]]}

    # Generate samples
    #param_values = saltelli.sample(problem,1000)


    #Y=np.array([electrolyzer_npv(*param_values[i][:])[0] for i in range(len(param_values))])

    #Si =sobol.analyze(problem, Y)

    #Si.keys()
    #save_outputs_dict_df = pd.DataFrame(save_outputs_dict)

    #plant_life=20 #years
    #tax_rate=0.25 #percent
    #interest_rate=0.10 #percent

    #electricity_cost=56.12 #USD/mwh
    #iron_ore_cost=90 #USD/ton
    #emission_cost=30 #USD per tonne
    #carbon_steel_price=700 #USD/ton

    #emission_factor=.413 ##Can be modeled as 0 if assume plant is powered by renewables
    #carbon_price=200
    #O2_price=40 #USD/tonne
    #el_spec=50 #KWh/kgh2
    #h2_investment_2020=.6 #Million euros/MWh2
    #Assume that 60% of the carbon produced could be sold

    baseline=[eta_el,tax_rate,interest_rate, electricity_cost,iron_ore_cost,
                emission_cost,carbon_steel_price,O2_price,el_spec,
                            h2_investment_2020,emission_factor]
    #baseline_npv=electrolyzer_npv(*baseline)[0]
    Outputs=electrolyzer_npv(*baseline)
    #print(Outputs)
     
    el_demand=Outputs[4]
    

    if elec_limit<el_demand:
        print('Electricity Demand Exceeds Production')

        Electric_driven=True

        if Electric_driven:
            print('Plant Confined by Electricity Production')
            scale_factor=elec_limit/el_demand
        
            steel_prod_yr=scale_factor*steel_prod_yr
            h2_prod_yr=scale_factor*h2_prod_yr
            Outputs=electrolyzer_npv(eta_el,tax_rate,interest_rate,electricity_cost,iron_ore_cost,emission_cost,carbon_steel_price,O2_price,el_spec,h2_investment_2020,emission_factor)
            #print('Scale Factor: ',scale_factor)
            #print('Steel Produced per year Scaled: ',steel_prod_yr)
            
    
    #print(Outputs)
    npv_hdri_eaf=Outputs[0]

    lcos=Outputs[3]
    lcoh=Outputs[5]

    el_demand=Outputs[4]
    steel_prod_yr=Outputs[6]

    #save_output_dict=establish_save_output_dict()
    #print(Output)
    
    #print(x[3])
    
    #baseline_irr=electrolyzer_npv(*baseline)[1]
    #baseline_npv

    #print(electricity_cost)



    #variation=np.arange(-20,21,5)
    #p=[1+(i/100) for i in variation]
    #change= [x*100 for x in p]

    #tax_rate_n=[tax_rate*x for x in p] # 35% tax rate has been assumed
    #interest_rate_n=[interest_rate*x for x in p] # Interest rate of 10%
    #iron_ore_cost_n=[iron_ore_cost*x for x in p] #USD/ton
    #steel_prod_yr_n=[steel_prod_yr*p) # plant capacity of 2.5 Mt/year
    #electricity_cost_n=[electricity_cost*x for x in p]#usd/MWh
    #emission_factor_n=[emission_factor*p)
    #emission_cost_n=[emission_cost*x for x in p] #usd/ton
    #carbon_steel_price_n=[carbon_steel_price*x for x in p] #usd/ton

    #O2_price_n=[O2_price*x for x in p]
    #el_spec_n=[el_spec*x for x in p]
    #h_invest_n=[h2_investment_2020*x for x in p]

    #Tax_rate is varied 
    #tax_rate_s=[electrolyzer_npv(tax_rate_n[i],interest_rate,electricity_cost,
    #            iron_ore_cost,emission_cost,carbon_steel_price,O2_price,el_spec,h2_investment_2020) 
    #          for i in range(len(tax_rate_n))][0]
    #Interest rate is varied
    #interest_rate_s=[(electrolyzer_npv(tax_rate,interest_rate_n[i],electricity_cost,
    #            iron_ore_cost,emission_cost,carbon_steel_price,O2_price,el_spec,h2_investment_2020))
    #               for i in range(len(tax_rate_n))][0]

    #iron_ore_cost_s=[(electrolyzer_npv(tax_rate,interest_rate,electricity_cost_n[i],
    #            iron_ore_cost,emission_cost,carbon_steel_price,O2_price,el_spec,h2_investment_2020))
    #               for i in range(len(tax_rate_n))][0]


    #electricity_cost_s=[(electrolyzer_npv(tax_rate,interest_rate,electricity_cost,
    #            iron_ore_cost_n[i],emission_cost,carbon_steel_price,O2_price,el_spec,h2_investment_2020))
    #                  for i in range(len(tax_rate_n))][0]              
                    
    # Emission prices are varied              
    #emission_cost_s=[electrolyzer_npv(tax_rate,interest_rate,electricity_cost,
    #            iron_ore_cost,emission_cost_n[i],carbon_steel_price,O2_price,el_spec,h2_investment_2020)
    #               for i in range(len(tax_rate_n))][0] #usd/ton

    #carbon_steel_price_s=[(electrolyzer_npv(tax_rate,interest_rate,electricity_cost,
    #            iron_ore_cost,emission_cost,carbon_steel_price_n[i],O2_price,el_spec,h2_investment_2020))
    #                    for i in range(len(tax_rate_n))][0] #usd/t


    #O2_price_s=[(electrolyzer_npv(tax_rate,interest_rate,electricity_cost,
    #            iron_ore_cost,emission_cost,carbon_steel_price,O2_price_n[i],el_spec,h2_investment_2020))
    #              for i in range(len(tax_rate_n))][0]#u



    #el_spec_s=[(electrolyzer_npv(tax_rate,interest_rate,electricity_cost,
    #            iron_ore_cost,emission_cost,carbon_steel_price,O2_price_n,el_spec_n[i],h2_investment_2020))
    #          for i in range(len(tax_rate_n))][0] #u

    #h_invest_s=[electrolyzer_npv(electrolyzer_npv(tax_rate,interest_rate,electricity_cost,
    #            iron_ore_cost,emission_cost,carbon_steel_price,O2_price_n,el_spec,h2_investment_2020_n[i]))
 
    #          for i in range(len(tax_rate_n))][0]
    return(h12_h2o,lcoh,lcos,npv_hdri_eaf,el_demand,steel_prod_yr)



#HDRI_EAF_Model(.6,200000000,20,.25,.10,56.12,90,30,700,40,50,.6,.413,1955356891)

#plant_life=20 #years
tax_rate=0.25 #percent
interest_rate=0.10 #percent
#electricity_cost=56.12 #USD/mwh
iron_ore_cost=90 #USD/ton
emission_cost=30 #USD per tonne
carbon_steel_price=700 #USD/ton
emission_factor=.413 ##Can be modeled as 0 if assume plant is powered by renewables
carbon_price=200
O2_price=40 #USD/tonne
el_spec=50 #KWh/kgh2
h2_investment_2020=.6 #Million euros/MWh2

