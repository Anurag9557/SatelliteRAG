"""
Labelled evaluation questions.

100 questions with accept conditions, plus 2 negative cases that
have no correct answer and are kept out of the Hit@k statistics.

A retrieved chunk counts as relevant if it satisfies ANY condition in
`accept`. Within one condition every key must match. Keys:

    text_contains       substring, case-insensitive
    text_contains_any   any of the substrings
    anything else       exact match on that metadata field

Procedure-number cases carry a text fallback as well, so they still hold
if extraction changes and the numbering shifts.

Two things to know when reading a score. One question, odhs_diag, cannot
be hit on the current build: no caption in the store names the ODHS, so
the ceiling is 99 and not 100. It stays in so the figures are not
flattered. And eleven questions are satisfied by a large share of the
store; they are listed in BROAD below and evaluate.py reports MRR with and
without them.

The questions are written against Aditya-L1. For another satellite,
either replace this file or point the evaluator at a satellite-specific
copy in satellites/<slug>/testset.py.
"""

TEST_SET = [
    # Procedures whose full body is in the file
    {"id": "obc_on", "q": "What is the procedure for switching on OBC?",
     "accept": [{"fcp_number": "5201"}], "note": "FCP-5201 body present"},
    {"id": "obc_reset", "q": "How do I perform an OBC system reset?",
     "accept": [{"fcp_number": "5203"}], "note": "FCP-5203 body present"},
    {"id": "obc_cpu", "q": "OBC CPU reset procedure",
     "accept": [{"fcp_number": "5204"}], "note": "FCP-5204 body present"},
    {"id": "tc_link", "q": "telecommand link test procedure",
     "accept": [{"fcp_number": "5215"}], "note": "FCP-5215 body present"},

    # "Which procedure for X" answered by the parsed index
    {"id": "hga_motors", "q": "Which FCP is used to switch on HGA motors?",
     "accept": [{"fcp_number": "5957"}], "note": "index record"},
    {"id": "tone_rng", "q": "tone ranging of transponder-1 procedure",
     "accept": [{"fcp_number": "6009"}], "note": "index record"},
    {"id": "star_ss1", "q": "powering on star sensor-1",
     "accept": [{"fcp_number": "5412"}], "note": "index record"},
    {"id": "gyro_on", "q": "gyro power on, off and selection",
     "accept": [{"fcp_number": "5601"}], "note": "index record"},
    {"id": "latch_valve", "q": "opening and closing of latch valves",
     "accept": [{"fcp_number": "5652"}], "note": "index record"},
    {"id": "suit_on", "q": "SUIT payload on procedure",
     "accept": [{"fcp_number": "6219"}], "note": "index record"},

    # Concepts spanning MOR / guidelines / manual
    #
    # These accept on a distinctive phrase rather than a bare topic word.
    # "eeprom" alone matched FCP-5203, whose notes merely mention an EEPROM
    # setting, and scored a hit for a question about FCP-5247. A condition
    # that any chunk mentioning the subject satisfies measures nothing.
    {"id": "safety", "q": "What are the safety logics?",
     "accept": [{"fcp_number": "5245"},
                {"text_contains_any": ["power safety logic", "safety logics",
                                       "enabling power safety"]}],
     "note": "FCP-5245 power safety logics"},
    {"id": "payload_gl", "q": "What are the guidelines for Payload operations?",
     "accept": [{"text_contains_any": ["payload sequencer", "master mep",
                                       "payload operation guideline"]}],
     "note": "manual / guidelines payload section"},
    {"id": "orbit", "q": "What are the Aditya-L1 orbit constraints?",
     "accept": [{"doc_type": "MOR",
                 "text_contains_any": ["halo orbit", "sun earth line",
                                       "orbit constraint"]}],
     "note": "halo-orbit design and selection"},

    # Figure retrieval
    #
    # "any MOR caption" would be satisfied by 200 chunks, so the figure
    # cases require the caption to actually name the thing asked for.
    {"id": "mrs_diagram", "q": "Provide me the MRS recovery flow diagram",
     "accept": [{"text_contains_any": ["master recovery sequencer",
                                       "safe mode recovery", "mrs"]}],
     "note": "MRS block diagram"},

    # Command log
    {"id": "cmds_date", "q": "What commands were uplinked on 2026-06-09?",
     "accept": [{"date": "2026-06-09"}], "note": "date filter to commands store"},

    # Procedures, index-backed
    {"id": "obc_scrub", "q": "OBC memory scrub operation", "accept": [{"fcp_number": "5225"}, {"text_contains_any": ["5225", "memory scrub"]}], "note": "FCP-5225"},
    {"id": "obc_wdt", "q": "OBC watchdog timer limit and reset procedure", "accept": [{"fcp_number": "5257"}, {"text_contains_any": ["5257", "watchdog timer"]}], "note": "FCP-5257"},
    {"id": "tm_prog", "q": "programmable telemetry configuration procedure", "accept": [{"fcp_number": "5231"}, {"text_contains_any": ["5231", "programmable telemetry"]}], "note": "FCP-5231"},
    {"id": "eeprom_mgmt", "q": "EEPROM management procedure", "accept": [{"fcp_number": "5247"}, {"text_contains_any": ["5247", "eeperom management", "eeprom management"]}], "note": "FCP-5247"},
    {"id": "sun_acq", "q": "sun acquisition using the auto sequencer", "accept": [{"fcp_number": "5001"}, {"text_contains_any": ["5001", "sun acquisition"]}], "note": "FCP-5001"},
    {"id": "mag_on", "q": "which FCP powers on the magnetometer", "accept": [{"fcp_number": "6191"}, {"text_contains_any": ["6191", "magnetometer on"]}], "note": "FCP-6191"},
    {"id": "wheel_on", "q": "powering on the reaction wheels", "accept": [{"fcp_number": "5551"}, {"text_contains_any": ["5551", "operations of wheels"]}], "note": "FCP-5551"},
    {"id": "wheel_unload", "q": "reaction wheel momentum dumping", "accept": [{"fcp_number": "5070"}, {"text_contains_any": ["momentum dumping", "momentum dump"]}], "note": "FCP-5070"},
    {"id": "star_ss2", "q": "powering on star sensor-2", "accept": [{"fcp_number": "5470"}, {"text_contains_any": ["5470", "star sensor -2"]}], "note": "FCP-5470"},
    {"id": "xband_t1", "q": "switch on X-band transmitter-1 to LGA or MGA", "accept": [{"fcp_number": "6006"}, {"text_contains_any": ["6006", "x-band transmitter"]}], "note": "FCP-6006"},
    {"id": "solexs_on", "q": "SoLEXS payload ON procedure", "accept": [{"fcp_number": "6193"}, {"text_contains_any": ["6193", "solex on"]}], "note": "FCP-6193"},
    {"id": "velc_on", "q": "VELC payload switch on procedure", "accept": [{"fcp_number": "6196"}, {"text_contains_any": ["6196", "velc switch on"]}], "note": "FCP-6196"},
    {"id": "fdir", "q": "How does the FDIR system work?", "accept": [{"text_contains_any": ["fdir", "fault detection and isolation", "fault isolation"]}], "note": "FDIR logics"},
    {"id": "safe_mode", "q": "What happens when the spacecraft enters safe mode?", "accept": [{"text_contains_any": ["safe mode detection", "safe mode recovery", "enters safe mode", "safe mode configuration"]}], "note": "safe mode detection and recovery"},
    {"id": "halo", "q": "Explain the halo orbit design and selection", "accept": [{"text_contains_any": ["halo orbit design", "halo orbit selection", "class-1 halo", "class 1 halo"]}], "note": "halo orbit design"},
    {"id": "crtbp", "q": "circular restricted three-body problem dynamics", "accept": [{"text_contains_any": ["crtbp", "restricted three-body", "restricted three body"]}], "note": "CRTBP dynamics"},
    {"id": "station_keep", "q": "station keeping strategy for the halo orbit", "accept": [{"text_contains_any": ["station keeping", "station-keeping"]}], "note": "station keeping"},
    {"id": "thermal", "q": "thermal requirements and special heater needs", "accept": [{"text_contains_any": ["thermal requirement", "special heater", "heater requirement"]}], "note": "thermal"},
    {"id": "power_budget", "q": "spacecraft power budget", "accept": [{"text_contains_any": ["power budget"]}], "note": "power budget"},
    {"id": "energy_bal", "q": "energy balance analysis", "accept": [{"text_contains_any": ["energy balance"]}], "note": "energy balance"},
    {"id": "odhs_diag", "q": "ODHS block diagram", "accept": [{"content_type": "image_caption", "text_contains_any": ["odhs", "data handling"]}], "note": "ODHS block diagram, caption; no caption names it, a known miss"},
    {"id": "prop_diag", "q": "propulsion system schematic diagram", "accept": [{"content_type": "image_caption", "text_contains_any": ["propulsion"]}], "note": "propulsion schematic, caption"},
    {"id": "suit_diag", "q": "functional diagram of the SUIT payload", "accept": [{"content_type": "image_caption", "text_contains_any": ["suit"]}], "note": "SUIT functional diagram, caption"},
    {"id": "cmds_obc", "q": "OBC commands uplinked on 2026-06-09", "accept": [{"content_type": "command_log", "subsystem": "OBC"}], "note": "commands store"},
    {"id": "cmds_dtg", "q": "DTG gyro commands uplinked on 2026-06-09", "accept": [{"content_type": "command_log", "subsystem": "DTG"}], "note": "commands store"},
    {"id": "obc_thermistor", "q": "How is the OBC supply routed for thermistor and thermocouple selection?", "accept": [{"text_contains_any": ["thermo-couple"]}], "note": "FCP-5206"},
    {"id": "obc_pwr_interrupt", "q": "How do you interrupt one OBC's power using the other OBC?", "accept": [{"text_contains_any": ["power interruption using other obc", "obc power interruption"]}], "note": "FCP-5208"},
    {"id": "diff_time_tag", "q": "Procedure for differential time-tagged commanding", "accept": [{"text_contains_any": ["differential time tagged"]}], "note": "FCP-5216"},
    {"id": "event_uplink_exec", "q": "How is an event uplinked and then executed?", "accept": [{"text_contains_any": ["event uplink procedure"]}], "note": "FCP-5217"},
    {"id": "obt_commanding", "q": "On-board-time based commanding operations", "accept": [{"text_contains_any": ["obt based commanding"]}], "note": "FCP-5218"},
    {"id": "bus1553_sel", "q": "Selecting the active 1553 bus", "accept": [{"text_contains_any": ["1553 bus selection"]}], "note": "FCP-5220"},
    {"id": "msg1553", "q": "Transferring messages over the 1553 bus", "accept": [{"text_contains_any": ["1553 message transfer"]}], "note": "FCP-5221"},
    {"id": "block_checksum", "q": "Block checksum operation on the OBC", "accept": [{"text_contains_any": ["block checksum"]}], "note": "FCP-5227"},
    {"id": "pcs", "q": "What is the programmable command sequencer?", "accept": [{"text_contains_any": ["programmable command sequencer"]}], "note": "FCP-5228"},
    {"id": "ccb", "q": "Configurable command block operation", "accept": [{"text_contains_any": ["configurable command block"]}], "note": "FCP-5232"},
    {"id": "ebc", "q": "Event based commanding on the OBC", "accept": [{"text_contains_any": ["event based commanding"]}], "note": "FCP-5233"},
    {"id": "obc_fdi", "q": "Enabling OBC fault detection and isolation", "accept": [{"text_contains_any": ["fault detection and isolation"]}], "note": "FCP-5248"},
    {"id": "rate_damp", "q": "Rate damp mode with the prime controller", "accept": [{"text_contains_any": ["rate damp mode"]}], "note": "FCP-5003"},
    {"id": "iac_mode", "q": "What is IAC mode?", "accept": [{"text_contains_any": ["iac mode"]}], "note": "FCP-5004"},
    {"id": "normal_modes_nohapss", "q": "Changing over to normal modes without HAPSS", "accept": [{"text_contains_any": ["without hapss"]}], "note": "FCP-5006"},
    {"id": "auto_firing", "q": "Enabling or disabling auto thruster firing", "accept": [{"text_contains_any": ["auto firing enable"]}], "note": "FCP-5008"},
    {"id": "manual_thruster", "q": "Manually firing a thruster", "accept": [{"text_contains_any": ["manual thruster firing"]}], "note": "FCP-5009"},
    {"id": "rw_logics", "q": "Reaction-wheel related control logics", "accept": [{"text_contains_any": ["reaction wheel related logics"]}], "note": "FCP-5011"},
    {"id": "auto_momentum", "q": "Auto momentum dumping using thrusters", "accept": [{"text_contains_any": ["auto momentum dumping"]}], "note": "FCP-5014"},
    {"id": "orbit_models", "q": "Orbit models and their selection for control", "accept": [{"text_contains_any": ["orbit models"]}], "note": "FCP-5017"},
    {"id": "gyro_cal", "q": "Gyro calibration operation", "accept": [{"text_contains_any": ["gyro cal operation"]}], "note": "FCP-5044"},
    {"id": "mrs_enable", "q": "Enabling the master recovery sequencer", "accept": [{"text_contains_any": ["master recovery sequencer"]}], "note": "FCP-5057"},
    {"id": "gyro_bias", "q": "Setting the gyro bias", "accept": [{"text_contains_any": ["gyro bias setting"]}], "note": "FCP-5602"},
    {"id": "csa_main", "q": "CSA main on/off procedure", "accept": [{"text_contains_any": ["csa main"]}], "note": "FCP-5608"},
    {"id": "fpga_power", "q": "FPGA main or redundant on/off operations", "accept": [{"text_contains_any": ["fpga main or redundant"]}], "note": "FCP-5702"},
    {"id": "papa_on", "q": "Nominal switch-on of the PAPA-1 payload", "accept": [{"text_contains_any": ["nominal switch on of papa1", "switch on of papa"]}], "note": "FCP-6151"},
    {"id": "aspex_swis", "q": "ASPEX SWIS power-on and data-acquisition start", "accept": [{"text_contains_any": ["swis power on and data acquisition"]}], "note": "FCP-6173"},
    {"id": "helios_on", "q": "Switching on the HELIOS payload", "accept": [{"text_contains_any": ["switch on helios"]}], "note": "FCP-6210"},
    {"id": "odhs_on", "q": "ODHS switch-on procedure", "accept": [{"text_contains_any": ["odhs on"]}], "note": "FCP-6222"},
    {"id": "odhs_record", "q": "Configuring all three ODHS chains in record mode", "accept": [{"text_contains_any": ["chains in record mode"]}], "note": "FCP-6223"},
    {"id": "obs_id", "q": "Setting the observation ID of the payloads", "accept": [{"text_contains_any": ["observation id of payloads"]}], "note": "FCP-6230"},
    {"id": "camera_config", "q": "Configuration settings of camera-1 and camera-2", "accept": [{"text_contains_any": ["camera-1 and camera-2"]}], "note": "FCP-6229"},
    {"id": "mission_overview", "q": "Give a high-level overview of the Aditya-L1 mission", "accept": [{"text_contains_any": ["mission overview"]}], "note": "MOR 1"},
    {"id": "mission_phases", "q": "What are the distinct phases of the mission?", "accept": [{"text_contains_any": ["mission phases", "earth bound phase"]}], "note": "MOR 4"},
    {"id": "l1_insertion", "q": "What happens during the L1 insertion phase?", "accept": [{"text_contains_any": ["l1 insertion"]}], "note": "MOR 4.3.1"},
    {"id": "transfer_traj", "q": "How is the transfer trajectory designed?", "accept": [{"text_contains_any": ["transfer trajectory"]}], "note": "MOR"},
    {"id": "orbit_det", "q": "Orbit dynamics and orbit determination", "accept": [{"text_contains_any": ["orbit determination"]}], "note": "MOR 3"},
    {"id": "prop_budget", "q": "What is the propellant budget?", "accept": [{"text_contains_any": ["propellant budget"]}], "note": "MOR"},
    {"id": "hapss", "q": "What is HAPSS, the high-accuracy active pixel sun sensor?", "accept": [{"text_contains_any": ["high accuracy active pixel", "hapss"]}], "note": "MOR 5.4.5"},
    {"id": "pointing_budget", "q": "What is the attitude pointing budget?", "accept": [{"text_contains_any": ["pointing budget"]}], "note": "MOR 5.3.1"},
    {"id": "bus_ctrl_fdir", "q": "How does the bus-controller FDIR work?", "accept": [{"text_contains_any": ["bus controller fdir"]}], "note": "MOR 7.9.1"},
    {"id": "power_safe_mode", "q": "Explain the power-based safe mode", "accept": [{"text_contains_any": ["power based safe mode"]}], "note": "MOR 7.10.1"},
    {"id": "crab_pointing", "q": "Calibration pointing at the Crab source", "accept": [{"text_contains_any": ["crab source"]}], "note": "MOR 6.3.7.1"},
    {"id": "suit_desc", "q": "What is the Solar Ultraviolet Imaging Telescope (SUIT)?", "accept": [{"text_contains_any": ["solar ultraviolet imaging telescope"]}], "note": "MOR 5.11.2"},
    {"id": "config_before_safe", "q": "Spacecraft configuration before safe mode", "accept": [{"text_contains_any": ["configuration before safe mode"]}], "note": "manual"},
    {"id": "stellar_cal", "q": "Stellar calibration guidelines", "accept": [{"text_contains_any": ["stellar calibration"]}], "note": "manual"},
    {"id": "velc_occulter", "q": "Aligning the VELC occulter center to the sun", "accept": [{"text_contains_any": ["occulter center to sun", "occultor center to sun"]}], "note": "manual"},
    {"id": "moi", "q": "Spacecraft moment-of-inertia parameters", "accept": [{"text_contains_any": ["moment of inertia"]}], "note": "manual"},
    {"id": "hard_emergency", "q": "What is on the hard emergency list?", "accept": [{"text_contains_any": ["hard emergency"]}], "note": "guidelines"},
    {"id": "thermal_limits", "q": "Thermal limits for the spacecraft", "accept": [{"text_contains_any": ["thermal limits"]}], "note": "guidelines"},
    {"id": "apss_on", "q": "Switching on the APSS sun sensor", "accept": [{"text_contains_any": ["switch on apss"]}], "note": "guidelines"},
    {"id": "xband_system", "q": "The X-band system", "accept": [{"text_contains_any": ["x-band system"]}], "note": "guidelines"},
    {"id": "momentum_bias", "q": "3-reaction-wheel configuration and momentum bias", "accept": [{"text_contains_any": ["momentum bias"]}], "note": "guidelines"},
    {"id": "heater_chart", "q": "Diagram of on-board heater power-status telemetry", "accept": [{"content_type": "image_caption", "text_contains_any": ["heater"]}], "note": "caption"},
    {"id": "thruster_cosines", "q": "Diagram of thruster direction cosines", "accept": [{"content_type": "image_caption", "text_contains_any": ["direction cosines"]}], "note": "caption"},
    {"id": "pressure_valves", "q": "Diagram of pressure transducers, latches and valves", "accept": [{"content_type": "image_caption", "text_contains_any": ["pressure transducer", "latches"]}], "note": "caption"},
    {"id": "od_residue", "q": "Plot of orbit-determination residues against LISA Pathfinder", "accept": [{"content_type": "image_caption", "text_contains_any": ["residue"]}], "note": "caption"},
    {"id": "cmd_cid", "q": "Which command had CID OBCS2370?", "accept": [{"text_contains_any": ["obcs2370"]}], "note": "commands"},
    {"id": "cmd_mep", "q": "Was the MEP enable command uplinked?", "accept": [{"text_contains_any": ["mep_ena", "mep enable"]}], "note": "commands"},
    {"id": "cmd_station", "q": "Commands uplinked from SCC to station N18", "accept": [{"text_contains_any": ["scc to n18"]}],
     "note": "commands; unmatchable until the log's station columns were read in the right order"},
]

# Questions that a large share of the store satisfies, counted on the
# current build. They are fair questions, but a hit on them says little:
# the command log covers one day from one station, so every record answers
# the three log questions, and the concept phrases recur through a whole
# chapter. The count after each id is how many chunks or records match.
BROAD = {
    "cmds_date",       # 203 of 203 records
    "cmd_station",     # 203 of 203
    "cmds_obc",        # 201 of 203
    "orbit",           # 62 chunks
    "fdir",            # 56
    "mrs_diagram",     # 51
    "od_residue",      # 40
    "mission_phases",  # 32
    "wheel_unload",    # 31
    "orbit_det",       # 26
    "station_keep",    # 23
}

# Out of scope. The system should say it cannot answer these, so they are
# reported separately and kept out of the Hit@k figures.
NEGATIVE_CASES = [
    {"id": "neg_cost", "q": "What is the total budget cost of the Aditya-L1 mission?",
     "accept": [], "note": "expect not-found"},
    {"id": "neg_chair", "q": "Who is the current chairman of ISRO?",
     "accept": [], "note": "expect not-found"},
]


# Routing labels
#
# What each question asks for, judged from its wording rather than from
# the document that happens to hold the answer: "What is IAC mode?" is a
# concept question even though a procedure answers it. A few are
# judgement calls (rate_damp, camera_config, stellar_cal, rw_logics,
# orbit_models). evaluate_router.py scores the routers against these.
INTENT_LABELS = {
    "procedure": """
        obc_on obc_reset obc_cpu tc_link tone_rng star_ss1 gyro_on
        latch_valve suit_on obc_scrub obc_wdt tm_prog eeprom_mgmt sun_acq
        wheel_on wheel_unload star_ss2 xband_t1 solexs_on velc_on
        obc_thermistor obc_pwr_interrupt diff_time_tag event_uplink_exec
        obt_commanding bus1553_sel msg1553 block_checksum ccb ebc obc_fdi
        rate_damp normal_modes_nohapss auto_firing manual_thruster
        auto_momentum gyro_cal mrs_enable gyro_bias csa_main fpga_power
        papa_on aspex_swis helios_on odhs_on odhs_record obs_id
        camera_config apss_on velc_occulter
    """,
    "which_procedure": "hga_motors mag_on",
    "figure": """
        mrs_diagram odhs_diag prop_diag suit_diag heater_chart
        thruster_cosines pressure_valves od_residue
    """,
    "command_history": """
        cmds_date cmds_obc cmds_dtg cmd_cid cmd_mep cmd_station
    """,
    "concept": """
        safety payload_gl orbit fdir safe_mode halo crtbp station_keep
        thermal power_budget energy_bal pcs iac_mode rw_logics orbit_models
        mission_overview mission_phases l1_insertion transfer_traj
        orbit_det prop_budget hapss pointing_budget bus_ctrl_fdir
        power_safe_mode crab_pointing suit_desc config_before_safe
        stellar_cal moi hard_emergency thermal_limits xband_system
        momentum_bias
    """,
}

_listed = [qid for ids in INTENT_LABELS.values() for qid in ids.split()]
if sorted(_listed) != sorted(case["id"] for case in TEST_SET):
    raise ValueError("INTENT_LABELS must name every question exactly once")

_intent_of = {qid: intent for intent, ids in INTENT_LABELS.items()
              for qid in ids.split()}
for _case in TEST_SET:
    _case["intent"] = _intent_of[_case["id"]]
