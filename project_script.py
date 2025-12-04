import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import dm4bem
plt.style.use('seaborn-v0_8-poster')

# -----------------------------------------------------------------------------------------------
def plot_all_surfaces(north_rad, south_rad, east_rad, west_rad, roof_rad, pnls_rad):
    # Create 2x3 subplots with shared axes
    fig, axes = plt.subplots(3, 2, figsize=(15, 15), sharex=True, sharey=True)

    # Flatten axes array for easier iteration
    axes_flat = axes.ravel()

    data_nodes = [
        (north_rad,  "Northern wall"),
        (south_rad,  "Southern wall"),
        (east_rad,   "Eastern wall"),
        (west_rad,   "Western wall"),
        (roof_rad,   "Roof"),
        (pnls_rad,   "Solar panels"),
    ]

    for ax, (node, subtitle) in zip(axes_flat, data_nodes):
        # Plot all components
        node.plot(ax=ax, legend=False)

        # Plot total
        total = node.sum(axis=1)
        total.plot(ax=ax, label="total", legend=False)

        # Subplot title (subtitle)
        ax.set_title(subtitle, fontsize=24)

        # Y axis: up to 4 ticks
        ax.yaxis.set_major_locator(MaxNLocator(4))

    # Global labels
    fig.supxlabel("Time", fontsize=20)
    fig.supylabel("Solar irradiance,  Φ / (W·m⁻²)", fontsize=20)

    # Main title
    fig.suptitle("Radiation on surfaces", fontsize=32)

    # Global legend at the bottom
    fig.legend(["direct", "diffuse", "reflected", "total"], loc="center right", bbox_to_anchor=(1, 0.5))

    # Adjust layout so titles and legend fit nicely
    fig.subplots_adjust(right=0.88)
    plt.show()
    return

# -------------------------   SYSTEM OF EQUATIONS   ---------------------------
def thermal_circuit(g10,g11,g12,g13,g14,g15,g20,g21,g22,g23,g24,g25,
                    g30,g31,g32,g33,g34,g35,g40,g41,g42,g43,g44,g45,
                    g50,g60,g61,g62,g63,g64,g65,g70,Kp,
                    c12,c14,c22,c24,c32,c34,c42,c44,c62,c64,c00):

    # Temperature nodes
    nodes = ['11','12','13','14','15', '21','22','23','24','25',
             '31','32','33','34','35', '41','42','43','44','45',
             '61','62','63','64','65', '00']
    
    # Branches for the heat flow
    flow_branches = ['q10','q11','q12','q13','q14','q15', 'q20','q21','q22','q23','q24','q25',
                     'q30','q31','q32','q33','q34','q35', 'q40','q41','q42','q43','q44','q45', 'q50',
                     'q60','q61','q62','q63','q64','q65', 'q70', 'q80']
    
    A = np.zeros([len(flow_branches),len(nodes)])
    A[0, 0] = 1                     #q10
    A[1, 0], A[1, 1] = -1, 1        #q11
    A[2, 1], A[2, 2] = -1, 1        #q12
    A[3, 2], A[3, 3] = -1, 1        #q13
    A[4, 3], A[4, 4] = -1, 1        #q14
    A[5, 4], A[5, 25] = -1, 1       #q15
    A[6, 5] = 1                     #q20
    A[7, 5], A[7, 6] = -1, 1        #q21
    A[8, 6], A[8, 7] = -1, 1        #q22
    A[9, 7], A[9, 8] = -1, 1        #q23
    A[10, 8], A[10, 9] = -1, 1      #q24
    A[11, 9], A[11, 25] = -1, 1     #q25
    A[12, 10] = 1                   #q30
    A[13, 10], A[13, 11] = -1, 1    #q31
    A[14, 11], A[14, 12] = -1, 1    #q32
    A[15, 12], A[15, 13] = -1, 1    #q33
    A[16, 13], A[16, 14] = -1, 1    #q34
    A[17, 14], A[17, 25] = -1, 1    #q35
    A[18, 15] = 1                   #q40
    A[19, 15], A[19, 16] = -1, 1    #q41
    A[20, 16], A[20, 17] = -1, 1    #q42
    A[21, 17], A[21, 18] = -1, 1    #q43
    A[22, 18], A[22, 19] = -1, 1    #q44
    A[23, 19], A[23, 25] = -1, 1    #q45
    A[24, 25] = 1                   #q50
    A[25, 20] = 1                   #q60
    A[26, 20], A[26, 21] = -1, 1    #q61
    A[27, 21], A[27, 22] = -1, 1    #q62
    A[28, 22], A[28, 23] = -1, 1    #q63
    A[29, 23], A[29, 24] = -1, 1    #q64
    A[30, 24], A[30, 25] = -1, 1    #q65
    A[31, 25] = 1                   #q70
    A[32, 25] = 1                   #q80
    A_dataframe = pd.DataFrame(A, index=flow_branches, columns=nodes)
    
    G = np.array(np.hstack(
        [g10,g11,g12,g13,g14,g15, g20,g21,g22,g23,g24,g25, g30,g31,g32,g33,g34,g35,
         g40,g41,g42,g43,g44,g45, g50, g60,g61,g62,g63,g64,g65, g70, Kp]))
    G_dataframe = pd.Series(G, index=flow_branches)
    
    C = np.array([0,c12,0,c14,0,0,c22,0,c24,0,0,c32,0,c34,0,0,c42,0,c44,0,0,c62,0,c64,0,c00])
    C_dataframe = pd.Series(C, index=nodes)
    
    b = pd.Series(['To',0,0,0,0,0,'To',0,0,0,0,0,'To',0,0,0,0,0,'To',0,0,0,0,0,'To',
                   'To',0,0,0,0,0,'To','Ti_sp'], index=flow_branches)
    
    flow_sources = pd.Series(['r11', 0, 0, 0, 'r15','r21', 0, 0, 0, 'r25',
                              'r31', 0, 0, 0, 'r35','r41', 0, 0, 0, 'r45',
                              'r61', 0, 0, 0, 'r65','Qa'], index=nodes)
    
    y = np.zeros(len(nodes))
    y[[25]]=1
    y_dataframe = pd.Series(y,index=nodes)
    
    TC = {"A":A_dataframe,"G":G_dataframe,"C":C_dataframe,"b":b,"f":flow_sources,"y":y_dataframe}
    return TC

# --------------------   STEADY-STATE AND STEP RESPONSE   ---------------------
def steady_state(T0, Ti_sp, TC):
    [As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)
    bt_len = len(TC["b"][TC["b"]!=0]) # the non-zero elements of vector b (BC)
    bt = T0 * np.ones(bt_len) # using T0 from boundary conditions, not from weather data.
    bt[-1:] = Ti_sp # also from boundary conditions.
    fq = np.zeros(len(TC["f"][TC["f"]!=0])) # these are the flow sources, at the steady state they are all zero
    uss = np.hstack([bt,fq]) # input vector for state space
    
    inv_As = pd.DataFrame(np.linalg.inv(As), columns=As.index, index=As.index)
    yss = (-Cs @ inv_As @ Bs + Ds) @ uss
    yss = float(yss.values[0])
    
    # Eigenvalues analysis
    eigenvalues = np.linalg.eig(As)[0]  
    delta_tmax = 2 * min(-1. / eigenvalues) # max time step for stability of Euler explicit
    
    "Time step"
    dt = dm4bem.round_time(delta_tmax)/2
    
    "Settle time"
    t_settle = 4 * max(-1. / eigenvalues)
    t_settle = np.ceil(t_settle / 3600) * 3600
    
    dm4bem.print_rounded_time('Δtmax', delta_tmax) # max time step for stability of Euler explicit
    dm4bem.print_rounded_time('dt', dt) # assumed time step
    dm4bem.print_rounded_time('duration', t_settle) # settling time
    return delta_tmax, dt, t_settle
    
# ---------------------   WEATHER DATA FOR STEADY STATE   ---------------------
def weather_of_dubendorf():
    weather_file = 'weather_data/CHE_ZH_Dubendorf.AP.066099_TMYx.2004-2018.epw'
    [weather_data, meta] = dm4bem.read_epw(weather_file, coerce_year=None)
    
    month_year = weather_data.index.strftime('%m-%Y') # Extract the month and year from the DataFrame index with the format 'MM-YYYY'
    unique_month_years = sorted(set(month_year))      # Create a set of unique month-year combinations
    unique_month_years = pd.DataFrame(unique_month_years,
                         columns=['Month-Year'])      # Create a DataFrame from the unique month-year combinations
    # print(unique_months_years)
    weather_data = weather_data[["temp_air", "dir_n_rad", "dif_h_rad"]]
    weather_data.index = weather_data.index.map(lambda t: t.replace(year=2025))
    print("Example weather record:")
    print(weather_data.loc['2025-06-29 12:00'])
    
    "Outdoor air temperature "
    weather_data['temp_air'].plot()
    plt.xlabel("Time")
    plt.ylabel("Dry-bulb outdoor air temperature, θ / °C")
    plt.legend([])
    plt.show()

    "Solar radiation: normal direct and horizontal diffuse"
    weather_data[['dir_n_rad', 'dif_h_rad']].plot()
    plt.xlabel("Time")
    plt.ylabel("Solar radiation: direct and diffuse, Φ / (W·m⁻²)")
    plt.legend(['$Φ_{direct}$', '$Φ_{diffuse}$'])
    plt.show()
    return weather_data


def radiation_on_the_walls(weather_data, surfaces, plot_radiations,
                           albedo_ground=0.25, albedo_roof=0, α_wSW=0.25, α_rSP=0.92, τ_gSW=0.3):
    radiation_dict = {}
    "Northern Wall"
    north_orientation = {'slope':90,'azimuth':180,'latitude':47.4}
    north_rad = dm4bem.sol_rad_tilt_surf(weather_data, north_orientation, albedo_ground)
    Etot_north = north_rad.sum(axis=1)  # convert Wh→W after resampling
    radiation_dict["r11"] = Etot_north * α_wSW  * surfaces["north"] 
    
    "Southern Wall"
    south_orientation = {'slope':90,'azimuth':0,'latitude':47.4}
    south_rad = dm4bem.sol_rad_tilt_surf(weather_data, south_orientation, albedo_ground)
    Etot_south = south_rad.sum(axis=1)
    radiation_dict["r21"] = Etot_south * α_wSW  * surfaces["south"] 
    
    "Eastern Wall"
    east_orientation = {'slope':90,'azimuth':270,'latitude':47.4}
    east_rad = dm4bem.sol_rad_tilt_surf(weather_data, east_orientation, albedo_ground)
    Etot_east = east_rad.sum(axis=1)
    radiation_dict["r31"] = Etot_east * α_wSW  * surfaces["east"]
    
    "Western Wall"
    west_orientation = {'slope':90,'azimuth':90,'latitude':47.4}
    west_rad = dm4bem.sol_rad_tilt_surf(weather_data, west_orientation, albedo_ground)
    Etot_west = west_rad.sum(axis=1)
    radiation_dict["r41"] = Etot_west * α_wSW  * surfaces["west"]
    
    "Roof & Solar panels"
    roof_orientation = {'slope':0,'azimuth':90,'latitude':47.4}
    pnls_orientation = {'slope':30,'azimuth':90,'latitude':47.4}
    roof_is_at_the_shadow = np.zeros([len(weather_data),len(weather_data.columns)])
    roof_is_at_the_shadow = pd.DataFrame(roof_is_at_the_shadow, index = weather_data.index, columns = weather_data.columns)
    roof_rad = dm4bem.sol_rad_tilt_surf(roof_is_at_the_shadow, roof_orientation, albedo_roof)
    pnls_rad = dm4bem.sol_rad_tilt_surf(weather_data, pnls_orientation, albedo_roof)
    
    Etot_roof = roof_rad.sum(axis=1) # [W/m2]
    radiation_dict["r61"] = Etot_roof * α_wSW * surfaces["roof"] # [W]
    
    radiation_dict["Etot_panels"] = pnls_rad.sum(axis=1) # [W/m2]
    radiation_dict["q61_panels"] = radiation_dict["Etot_panels"] * α_rSP # relative heat flow through panels [W/m2]
    radiation_dict["r61_panels"] = radiation_dict["q61_panels"] * surfaces["roof"] # absolute heat flow [W]
    
    "In the interior of the substation"
    _north = north_rad.sum(axis=1) * surfaces["windows"] * 6 # [W]
    _south = south_rad.sum(axis=1) * surfaces["windows"] * 6 # [W]
    _east  = east_rad.sum(axis=1)  * surfaces["windows"] * 4 # [W]
    _west  = west_rad.sum(axis=1)  * surfaces["windows"] * 4 # [W]
    radiation_dict["r15"] = τ_gSW * α_wSW * (_south + _east + _west) # interior of northern wall
    radiation_dict["r25"] = τ_gSW * α_wSW * (_north + _east + _west) # interior of southern wall
    radiation_dict["r35"] = τ_gSW * α_wSW * (_north + _south + _west) # interior of eastern wall
    radiation_dict["r45"] = τ_gSW * α_wSW * (_north + _south + _east) # interior of western wall
    radiation_dict["r65"] = τ_gSW * α_wSW * (_north + _south + _east + _west) # interior of roof
    if plot_radiations:
        plot_all_surfaces(north_rad, south_rad, east_rad, west_rad, roof_rad, pnls_rad)
    return radiation_dict


# ---------------------   SIMULATION OF THE STEADY STATE   --------------------
def simulation_steadystate(dt, t_settle, delta_tmax, T0, Ti_sp, f_90, TC, summer=True, end_date=None):
    """
    In this section, I am simulating the transient behaviour of my building
    under fixed boundary conditions:
        - constant outdoor temperature T0,
        - constant indoor setpoint Ti_sp = 0, no controller switch,
        - No solar radiation, r = 0,
        - No internal loads, Qa = 0.
    
    This first block answers to the question:
        How does the system evolve from an arbitrary initial state under fixed
        boundary conditions and no internal gains?
    """
    
    # time vector
    n = int(np.floor(t_settle / dt))    # number of time steps
    start_date = '2025-09-01 00:00:00' if summer else '2025-01-01 00:00:00'
    if end_date==None:
        end_date = '2025-09-02 00:00:00' if summer else '2025-01-02 00:00:00'
    time = pd.date_range(start= start_date, periods=n, freq=f"{int(dt)}s")
    
    To = T0 * np.ones(n)                     # outdoor temperature
    Ti_sp = Ti_sp * np.ones(n)               # indoor temperature set point
    r65 = 0 * np.ones(n)                     # solar radiation in the interior of the roof
    r11=r15=r21=r25=r31=r35=r41=r45=r61=r65  # all the solar radiation inside and outside
    Qa = f_90 * np.ones(n)                   # flow sources inside the room
    
    input_data_set = pd.DataFrame({
        'To':  To, 'Ti_sp': Ti_sp,
        'r11': r11, 'r21': r21, 'r31': r31, 'r41': r41, 'r61': r61,
        'r15': r15, 'r25': r25, 'r35': r35, 'r45': r45, 'r65': r65,
        'Qa': Qa}, index=time)
    
    [As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)
    u = dm4bem.inputs_in_time(us, input_data_set) # inputs in time from input_data_set
    
    "Simulation, time integration"
    # Initial conditions
    θ_exp = pd.DataFrame(index=u.index)     # empty df with index for explicit Euler
    θ_imp = pd.DataFrame(index=u.index)     # empty df with index for implicit Euler
    
    θ0 = 0.0                    # initial temperatures
    θ_exp[As.columns] = θ0      # fill θ for Euler explicit with initial values θ0
    θ_imp[As.columns] = θ0      # fill θ for Euler implicit with initial values θ0
    
    I = np.eye(As.shape[0])     # identity matrix
    for k in range(u.shape[0] - 1):
        θ_exp.iloc[k + 1] = (I + dt * As)\
            @ θ_exp.iloc[k] + dt * Bs @ u.iloc[k]
        θ_imp.iloc[k + 1] = np.linalg.inv(I - dt * As)\
            @ (θ_imp.iloc[k] + dt * Bs @ u.iloc[k])
            
    # outputs
    y_exp = (Cs @ θ_exp.T + Ds @  u.T).T
    y_imp = (Cs @ θ_imp.T + Ds @  u.T).T
    
    "Plot results"
    y = pd.concat([y_exp, y_imp], axis=1, keys=['Explicit', 'Implicit'])
    y.columns = y.columns.get_level_values(0) # Flatten the two-level column labels into a single level
    ax = y.plot()
    ax.hlines(T0, start_date, end_date, color="black", linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature, $\\theta_i$ / °C')
    ax.set_title(f'Time step: $dt$ = {dt:.0f} s; $dt_{{max}}$ = {delta_tmax:.0f} s')
    ax.set_xlim(start_date, end_date)
    ax.set_ylim(0,15)
    plt.legend(['$Indoor_{explicit}$', '$Indoor_{implicit}$', '$Outdoor_{BC}$'])
    plt.show()
    return
        
# ------------   SIMULATION WITH WEATHER DATA AND SOLAR RADIATION   -----------
def simulation_weather(weather_data, T0, Ti_sp, f_90, Kp, delta_tmax, t_settle, dt, TC, surfaces, summer=True, mode='both',
                       albedo_ground=0.25, albedo_roof=0, α_wSW=0.25, α_rSP=0.92, τ_gSW=0.3,
                       plot_radiations = False):
    """
    In this section, I am simulating the transient behaviour of my building
    under dynamic conditions, considering that the outdoor temperature is variable
    in my system, and comes from the weather data I imported. Now, I am adding the
    solar radiation, that until this moment has been 0.
        - variable outdoor temperature T0(t),
        - constant indoor setpoint with controller switch,
        - solar radiation variable r = r(t),
        - No internal loads, Qa = 0.
    
    """
    if mode not in ['both', 'temperature', 'radiation']:
        raise ValueError("mode can be one of the followings: 'temperature', 'radiation', 'both'.")
    
    # Define start and end dates, time is 00:00 if not indicated
    if summer:
        start_date = '2025-09-01 00:00:00'
    else:
        start_date = '2025-01-01 00:00:00'
    
    # time vector
    n = int(np.floor(t_settle / dt))    # number of time steps
    time = pd.date_range(start=start_date, periods=n, freq=f"{int(dt)}s", tz=weather_data.index.tz)
    start_date = pd.Timestamp(start_date, tz=weather_data.index.tz)
    end_date = str(time[-1])
    weather_data = weather_data.loc[start_date:end_date] # Filter the data based on the start and end dates
    weather_data = weather_data.reindex(time, method='ffill')
    
    if mode=='temperature':
        radiation_dict = {'r11':0, 'r21':0, 'r31':0, 'r41':0, 'r61':0,
                          'r15':0, 'r25':0, 'r35':0, 'r45':0, 'r65':0,
                          'Etot_panels':0, 'q61_panels':0, 'r61_panels':0}
        
    else:
        radiation_dict = radiation_on_the_walls(weather_data, surfaces, plot_radiations, albedo_ground, albedo_roof, α_wSW, α_rSP, τ_gSW)

    if mode=='radiation':
        temperature_outdoor = T0
    else:
        temperature_outdoor = weather_data['temp_air'].values
    To = temperature_outdoor         # outdoor temperature
    Ti_sp = Ti_sp * np.ones(n)       # indoor temperature set point
    Qa = f_90 * np.ones(n)           # flow sources inside the room, in W
    
    input_data_set = pd.DataFrame({
        'To':  To,  'Ti_sp': Ti_sp,
        'r11': radiation_dict['r11'], 'r21': radiation_dict['r21'], 'r31': radiation_dict['r31'],
        'r41': radiation_dict['r41'], 'r61': radiation_dict['r61'], 'r15': radiation_dict['r15'],
        'r25': radiation_dict['r25'], 'r35': radiation_dict['r35'], 'r45': radiation_dict['r45'],
        'r65': radiation_dict['r65'], 'Qa': Qa, 'Etot_panels': radiation_dict['Etot_panels'],
        'q61_panels': radiation_dict['q61_panels'], 'r61_panels': radiation_dict['r61_panels']},
        index=time)
    
    [As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)
    u = dm4bem.inputs_in_time(us, input_data_set) # inputs in time from input_data_set
    
    "Simulation, time integration"
    # Initial conditions
    θ0 = T0 # initial temperatures
    θ_exp = pd.DataFrame(index=u.index)
    θ_exp[As.columns] = θ0 # Fill θ with initial valeus θ0
    
    # time integration
    I = np.eye(As.shape[0]) # identity matrix
    for k in range(u.shape[0] - 1):
        θ_exp.iloc[k + 1] = (I + dt * As)\
            @ θ_exp.iloc[k] + dt * Bs @ u.iloc[k]
            
    # outputs
    y = (Cs @ θ_exp.T + Ds @  u.T).T
    q_HVAC = Kp * (u['q80'] - y['00']) / surfaces["roof"]  # W/m²
    data = pd.DataFrame({'To': input_data_set['To'], 'θi': y['00'], 'q_HVAC': q_HVAC,
                          'Etot_panels': input_data_set['Etot_panels'],
                          'r61_panels': input_data_set['r61_panels'],
                          'q61_panels': input_data_set['q61_panels']})
    
    "Plot results"
    if mode == 'temperature':
        fig, axs = plt.subplots(1, 1, figsize=(15, 5))
        data[['To', 'θi']].plot(ax=axs, xticks=[], ylabel='Temperature, $θ$ / °C')
        axs.legend(['$θ_{outdoor}$', '$θ_{indoor}$'], loc='upper right')
        axs.set_title('Indoor temperature vs. Outdoor temperature', fontsize=24)
        plt.tight_layout()
        plt.show()
    
    else:
        fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        # 1) Indoor vs Outdoor temperature
        data[['To', 'θi']].plot(
            ax=axs[0],
            xticks=[],
            ylabel='Temperature, $θ$ / °C')
        axs[0].legend(['$θ_{outdoor}$', '$θ_{indoor}$'], loc='upper right')
        axs[0].set_ylabel('Temperature, $θ$ / °C', fontsize=20)
        axs[0].set_title('Indoor temperature vs. Outdoor temperature', fontsize=24)
        
        # 2) Solar panels heat flow vs Heating system consumption
        data[['q61_panels', 'q_HVAC']].plot(
            ax=axs[1],
            xticks=[],
            ylabel='Heat rate, $q$ / (W·m⁻²)')
        axs[1].legend(['$q_{panels}$', '$q_{HVAC}$'], loc='upper right')
        axs[1].set_ylabel('Heat rate, $q$ / (W·m⁻²)', fontsize=20)
        axs[1].set_title('Solar panels heat flow vs. Heating system consumption', fontsize=24)
        
        plt.tight_layout()
        plt.show()
    
    max_load = data['q_HVAC'].max()
    max_load_index = data['q_HVAC'].idxmax()
    energy = (data['q_HVAC'].clip(lower=0) * dt).sum() * surfaces["roof"] /3.6e6
    print(f'Min. indoor temperature: {data["θi"].min():.4f} °C')
    print(f'Max. indoor temperature: {data["θi"].max():.4f} °C')
    print(f"Max. load: {max_load:.1f} W/m² at {max_load_index}")
    print(f"Energy consumption: {energy:.1f} kWh")
    
    if mode!='temperature':
        max_panels = data['q61_panels'].max() /1e3
        max_panels_index = data['q61_panels'].idxmax()
        energy_panels = abs(data['q61_panels'] * dt).sum() /3.6e6
        print(f"Max. load panels: {max_panels:.1f} kW at {max_panels_index}")
        print(f"Energy production of the solar panels: {energy_panels:.1f} kWh")
        if Kp!=0:
            coverage = energy_panels / energy *100
            print(f"Energy covered by the solar panels: {coverage:.2f} %")
    return