"""
Routing-only questions.

Each has an intent label and nothing else: no accept conditions, because
nothing is retrieved. evaluate_router.py scores the routers on these next
to the 100 benchmark questions.

Two kinds are mixed:

    paraphrases    the same intents in wording the benchmark does not use
    failure cases  wording that trips the rules: "show me the steps",
                   "power budget", "how is", a date inside a question
                   about the documents

They were written after the rules and with their weak spots in mind, so
they are a regression check, not an unbiased held-out estimate. For that,
someone who has not read the rules should write a fresh set.

Subjects and identifiers are generic on purpose. Nothing here comes from
the mission documents; the CID and the dates are made up.
"""

ROUTER_CASES = [
    # procedure
    {"id": "r_proc_showme", "intent": "procedure",
     "q": "Show me the steps to switch on the onboard computer"},
    {"id": "r_proc_online", "intent": "procedure",
     "q": "How can I bring the star tracker back online?"},
    {"id": "r_proc_turn_off", "intent": "procedure",
     "q": "Turn off the second transponder"},
    {"id": "r_proc_heater", "intent": "procedure",
     "q": "What steps do I follow to enable the battery heater?"},
    {"id": "r_proc_isolate", "intent": "procedure",
     "q": "Isolate a failed reaction wheel"},
    {"id": "r_proc_first_on", "intent": "procedure",
     "q": "First switch-on of the coronagraph payload"},
    {"id": "r_proc_counter", "intent": "procedure",
     "q": "Reset the watchdog counter on the redundant computer"},
    {"id": "r_proc_valves", "intent": "procedure",
     "q": "Close the propulsion latch valves before a manoeuvre"},
    {"id": "r_proc_bus", "intent": "procedure",
     "q": "Change the active data bus to the redundant side"},
    {"id": "r_proc_table", "intent": "procedure",
     "q": "Uploading a new telemetry table to the spacecraft"},

    # which_procedure
    {"id": "r_which_heater", "intent": "which_procedure",
     "q": "Which FCP covers the heater controller switch-on?"},
    {"id": "r_which_number", "intent": "which_procedure",
     "q": "What is the procedure number for enabling the safe-mode logic?"},
    {"id": "r_which_list", "intent": "which_procedure",
     "q": "List all procedures for the X-band subsystem"},
    {"id": "r_which_recover", "intent": "which_procedure",
     "q": "Which procedure do I use to recover from a bus under-voltage?"},
    {"id": "r_which_boom", "intent": "which_procedure",
     "q": "FCP number for magnetometer boom deployment"},
    {"id": "r_which_flight", "intent": "which_procedure",
     "q": "Which flight control procedure handles wheel speed limits?"},

    # figure
    {"id": "r_fig_power", "intent": "figure",
     "q": "Block diagram of the power distribution system"},
    {"id": "r_fig_thermal", "intent": "figure",
     "q": "Can I see the figure of the thermal control layout?"},
    {"id": "r_fig_orbit", "intent": "figure",
     "q": "Plot of the halo orbit around L1"},
    {"id": "r_fig_rf", "intent": "figure",
     "q": "Schematic of the X-band RF chain"},
    {"id": "r_fig_mounting", "intent": "figure",
     "q": "Illustration of the reaction wheel mounting geometry"},
    {"id": "r_fig_flow", "intent": "figure",
     "q": "Flow chart of the safe-mode entry sequence"},

    # command_history
    {"id": "r_cmd_date", "intent": "command_history",
     "q": "Which telecommands went up on 2025-11-03?"},
    {"id": "r_cmd_error", "intent": "command_history",
     "q": "Did any telecommand come back with an error status?"},
    {"id": "r_cmd_cid", "intent": "command_history",
     "q": "Show the log entry for CID ABCD1234"},
    {"id": "r_cmd_history", "intent": "command_history",
     "q": "Command history for the star sensor"},
    {"id": "r_cmd_sent", "intent": "command_history",
     "q": "Commands sent to the payload during the last pass"},
    {"id": "r_cmd_executed", "intent": "command_history",
     "q": "Was the heater enable command executed?"},
    {"id": "r_cmd_uplink_verb", "intent": "command_history",
     "q": "Did we uplink the wheel speed update yesterday?"},
    {"id": "r_cmd_log", "intent": "command_history",
     "q": "What does the telecommand log show for the magnetometer?"},

    # concept
    {"id": "r_con_power", "intent": "concept",
     "q": "What is the spacecraft power budget at end of life?"},
    {"id": "r_con_selected", "intent": "concept",
     "q": "How was the launch window selected?"},
    {"id": "r_con_date", "intent": "concept",
     "q": "What does the mission report say about the eclipse season "
          "starting 2026-02-10?"},
    {"id": "r_con_operations", "intent": "concept",
     "q": "Give an overview of ground station operations"},
    {"id": "r_con_timetag", "intent": "concept",
     "q": "What is a time-tagged command?"},
    {"id": "r_con_how_is", "intent": "concept",
     "q": "How is the attitude determined on board?"},
    {"id": "r_con_image", "intent": "concept",
     "q": "What image resolution does the ultraviolet telescope achieve?"},
    {"id": "r_con_why", "intent": "concept",
     "q": "Why was a halo orbit chosen instead of a Lissajous orbit?"},
    {"id": "r_con_redundancy", "intent": "concept",
     "q": "Describe the redundancy scheme of the data handling system"},
    {"id": "r_con_switching", "intent": "concept",
     "q": "Explain the switching logic between the two onboard computers"},
]
