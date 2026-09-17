"""
Labelled evaluation questions for the public ISRO demo dataset in
satellites/aditya-l1/ (the four public ISRO/ISSDC booklets: mission
booklet, spacecraft/orbit booklet, Announcement of Opportunity, and
mission brochure).

This is a separate file from testset.py on purpose: testset.py is written
against the confidential real Aditya-L1 corpus (FCP procedures, telecommand
log) which has FCP/TCH document types this public dataset does not have.
Running testset.py's questions against this dataset would just report
misses across the board, so this file has its own questions grounded in
what is actually in satellites/aditya-l1/extracted/chunks.json and
image_captions.json.

Same format and matching rules as testset.py: a hit counts as relevant if
it satisfies ANY condition in `accept`; within one condition every key
must match. Keys:

    text_contains       substring, case-insensitive
    text_contains_any   any of the substrings
    anything else       exact match on that metadata field (e.g.
                         source_doc, content_type)

Run with: python evaluate_public_demo.py
"""

TEST_SET = [
    # Science objectives / mission overview (MOR, OPS_MANUAL)
    {"id": "science_objectives", "q": "What are the major science objectives of the Aditya-L1 mission?",
     "accept": [{"text_contains_any": ["coronal heating", "solar wind acceleration"]}],
     "note": "MOR/OPS_MANUAL science objectives"},
    {"id": "halo_orbit_advantage", "q": "What is the advantage of placing Aditya-L1 in a halo orbit around L1?",
     "accept": [{"text_contains_any": ["continuously viewing", "without any occultation", "without occultation"]}],
     "note": "MOR/OPS_MANUAL halo orbit rationale"},
    {"id": "payload_split", "q": "How many of Aditya-L1's payloads directly view the Sun versus do in-situ measurements?",
     "accept": [{"text_contains_any": ["four payloads", "remote sensing and three"]}],
     "note": "MOR p5 / GUIDELINES p2"},
    {"id": "uniqueness", "q": "What is unique about the Aditya-L1 mission compared to other solar missions?",
     "accept": [{"text_contains_any": ["first time spatially resolved solar disk", "uniqueness"]}],
     "note": "MOR/OPS_MANUAL uniqueness section"},
    {"id": "l5_vantage", "q": "Why is the L5 Lagrange point mentioned as relevant to studying the Sun?",
     "accept": [{"text_contains_any": ["l5", "earth directed cme"]}],
     "note": "MOR p11, limitations discussion"},

    # Payloads (MOR, OPS_MANUAL, HANDBOOK)
    {"id": "velc_desc", "q": "What does the VELC payload study?",
     "accept": [{"text_contains_any": ["visible emission line coronagraph", "reflective coronagraph"]}],
     "note": "VELC description"},
    {"id": "suit_desc", "q": "What does the SUIT payload image?",
     "accept": [{"text_contains_any": ["solar ultra-violet imaging telescope", "photosphere and chromosphere"]}],
     "note": "SUIT description"},
    {"id": "aspex_subsystems", "q": "What subsystems does the ASPEX payload comprise?",
     "accept": [{"text_contains_any": ["swis and steps"]}],
     "note": "ASPEX / SWIS / STEPS"},
    {"id": "magnetometer_boom", "q": "Where are the Magnetometer sensors located on the Aditya-L1 spacecraft?",
     "accept": [{"text_contains_any": ["deployable boom"]}],
     "note": "MAG payload placement"},
    {"id": "velc_developer", "q": "Which institute developed the VELC instrument?",
     "accept": [{"text_contains_any": ["indian institute of astrophysics"]}],
     "note": "VELC developer, IIA Bengaluru"},
    {"id": "suit_developer", "q": "Which institute developed the SUIT instrument?",
     "accept": [{"text_contains_any": ["inter university centre for astronomy", "inter-university centre for astronomy"]}],
     "note": "SUIT developer, IUCAA Pune"},

    # Orbit / trajectory (MOR, OPS_MANUAL)
    {"id": "lagrange_count", "q": "How many Lagrange points exist in a two-body gravitational system?",
     "accept": [{"text_contains_any": ["five lagrange points", "total five"]}],
     "note": "Lagrange point explanation"},
    {"id": "l1_distance", "q": "How far is the L1 Lagrange point from Earth?",
     "accept": [{"text_contains_any": ["1.5 million km", "1.5 million kilometres", "1.5 million kilometers"]}],
     "note": "L1 distance"},
    {"id": "travel_time_l1", "q": "How long did it take Aditya-L1 to travel from launch to the L1 point?",
     "accept": [{"text_contains_any": ["four months"]}],
     "note": "cruise phase duration"},

    # Launch (MOR, OPS_MANUAL, HANDBOOK, GUIDELINES)
    {"id": "launch_vehicle", "q": "What launch vehicle placed Aditya-L1 into orbit?",
     "accept": [{"text_contains_any": ["pslv"]}],
     "note": "PSLV / PSLV XL / PSLV-C57"},
    {"id": "launch_site", "q": "From where was Aditya-L1 launched?",
     "accept": [{"text_contains_any": ["satish dhawan space centre", "sathish dhawan space centre", "sriharikota"]}],
     "note": "SDSC-SHAR, Sriharikota"},
    {"id": "spacecraft_mass", "q": "What is the mass of the Aditya-L1 spacecraft at launch?",
     "accept": [{"text_contains_any": ["1480.7"]}],
     "note": "HANDBOOK PSLV-C57 mission specs table"},
    {"id": "launch_date", "q": "When was Aditya-L1 launched?",
     "accept": [{"text_contains_any": ["september 2, 2023"]}],
     "note": "GUIDELINES p2 launch date"},

    # Announcement of Opportunity / operations (GUIDELINES)
    {"id": "ao_deadline", "q": "What is the deadline for submitting proposals in the first Aditya-L1 AO cycle?",
     "accept": [{"text_contains_any": ["6 feb, 2026", "february 6, 2026"]}],
     "note": "AO proposal deadline"},
    {"id": "ao_instruments", "q": "Which instruments can be proposed for during the first Aditya-L1 Announcement of Opportunity cycle?",
     "accept": [{"text_contains_any": ["velc and suit"]}],
     "note": "AO cycle scope"},
    {"id": "alpps", "q": "What system is used to submit Aditya-L1 observation proposals?",
     "accept": [{"text_contains_any": ["proposal processing system", "alpps"]}],
     "note": "ALPPS"},
    {"id": "proprietary_period", "q": "Is there a proprietary period for Aditya-L1 data from the first AO cycle?",
     "accept": [{"text_contains_any": ["no proprietary period", "not any proprietary period"]}],
     "note": "data rights"},
    {"id": "poc_institutions", "q": "Which institutions host the Payload Operations Centre for VELC and SUIT?",
     "accept": [{"text_contains_any": ["indian institute of astrophysics", "inter-university centre for astronomy",
                                       "inter university centre for astronomy"]}],
     "note": "POC at IIA and IUCAA"},
    {"id": "data_format", "q": "In what data format is Aditya-L1 payload data released?",
     "accept": [{"text_contains_any": ["cdf and fits"]}],
     "note": "CDF / FITS format"},

    # Figures (image_caption chunks)
    {"id": "velc_photo", "q": "Show me a photograph of the VELC payload",
     "accept": [{"content_type": "image_caption", "text_contains_any": ["velc"]}],
     "note": "VELC payload photo caption"},
    {"id": "trajectory_diagram", "q": "Diagram of Aditya-L1's trajectory to the L1 point",
     "accept": [{"content_type": "image_caption", "text_contains_any": ["trajectory", "halo orbit"]}],
     "note": "trajectory/orbit diagram caption"},
    {"id": "launch_photo", "q": "Photograph of the rocket launch for Aditya-L1",
     "accept": [{"content_type": "image_caption", "text_contains_any": ["pslv", "launch"]}],
     "note": "launch photo caption"},

    # Synthesis / comparison questions (exercise combining two grounded
    # facts, not just single-fact lookup)
    {"id": "velc_suit_diff", "q": "What is the difference between the VELC coronagraph and the SUIT imager?",
     "accept": [{"text_contains_any": ["reflective coronagraph"]},
                {"text_contains_any": ["uv telescope", "ultra-violet imaging telescope"]}],
     "note": "requires combining two separately-stated facts"},
    {"id": "aspex_papa_diff", "q": "What is the difference between the ASPEX and PAPA payloads?",
     "accept": [{"text_contains_any": ["solar wind particle experiment"]},
                {"text_contains_any": ["plasma analyser package"]}],
     "note": "requires combining two separately-stated facts"},
]

# Out of scope for this dataset. The system should say it cannot answer
# these, so they are reported separately and kept out of the Hit@k figures.
NEGATIVE_CASES = [
    {"id": "neg_budget", "q": "What is the total budget cost of the Aditya-L1 mission?",
     "accept": [], "note": "expect not-found"},
    {"id": "neg_chairman", "q": "Who is the current chairman of ISRO?",
     "accept": [], "note": "expect not-found"},
    {"id": "neg_lifetime", "q": "What is the expected operational lifetime of the Aditya-L1 spacecraft in years?",
     "accept": [], "note": "expect not-found, not stated in these documents"},
]
