"""
Checks on the parts of answering that are decided in code.

    python tests/test_answering.py

No model, no vector store and no document is needed: every identifier,
passage and command record below is made up. That is the point of having
them. On real answers the grounding guard has not removed anything since
the current prompt came in, so the only way to know it still works is to
hand it an answer that needs it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import generate, grounding, llm, retrieve          # noqa: E402
from core.retrieve import Hit                                 # noqa: E402
from core.router import QueryPlan                             # noqa: E402

CONTEXT = """[Source 1 | FCP | FCP-7001 | p.9 | "XYZ- 7001: HEATER ON"]
XYZ- 7001: HEATER ON
PROCEDURE:
Send HTR_MAIN_ON with CID HTRS1234 and check TMH0456.
END OF PROCEDURE

[Source 2 | FCP | FCP-7002 | p.1 | "FCP Index"]
FCP 7002 - XYZ - HEATER OFF"""


# The guard

def test_guard_leaves_a_grounded_answer_alone():
    answer = "Send HTR_MAIN_ON (CID HTRS1234), then check TMH0456."
    out, removed = grounding.guard(answer, CONTEXT)
    assert out == answer and removed == []


def test_guard_removes_an_identifier_that_is_not_in_the_passages():
    answer = "Send HTR_MAIN_ON (CID HTRS1234), then confirm with TMH0999."
    out, removed = grounding.guard(answer, CONTEXT)
    assert removed == ["TMH0999"]
    assert "TMH0999" not in out and grounding.PLACEHOLDER in out
    assert "HTRS1234" in out                      # the real one stays
    assert "1 command or telemetry ID(s)" in out  # and the reader is told


def test_guard_checks_hex_values_and_values_after_cid():
    answer = "Set the register to 0x3F and use CID 4471."
    out, removed = grounding.guard(answer, CONTEXT)
    assert sorted(removed) == ["0x3F", "4471"]
    assert "0x3F" not in out and "4471" not in out


def test_guard_is_not_fooled_by_a_longer_identifier():
    # HTRS12345 is not HTRS1234 with something after it.
    out, removed = grounding.guard("Use HTRS12345.", CONTEXT)
    assert removed == ["HTRS12345"]


def test_guard_does_not_catch_what_it_was_never_meant_to():
    # A made-up name with no digits passes. The module says so; this pins
    # it, so nobody reads the guard as more than it is.
    out, removed = grounding.guard("Send HEATER_BOOST_NOW.", CONTEXT)
    assert removed == []


# The context note

def test_note_names_the_procedure_whose_steps_are_present():
    note = grounding.context_note(CONTEXT)
    assert "procedure table (numbered steps) for FCP 7001" in note
    assert "index lines only" in note             # 7002 is a row, no steps


def test_note_for_index_rows_only():
    note = grounding.context_note("FCP 7002 - XYZ - HEATER OFF")
    assert note.startswith("Note: The context has no procedure tables.")
    assert grounding.NO_STEPS_LINE in note


def test_no_note_when_there_is_no_procedure():
    assert grounding.context_note("The orbit is a halo orbit.") == ""


# The no-steps sentence under a procedure that has steps

def test_no_steps_under_a_procedure_with_its_table_is_removed():
    answer = (f"FCP 7001 - XYZ - HEATER ON\nSend HTR_MAIN_ON.\n"
              f"{grounding.NO_STEPS_LINE}\n")
    out, n = grounding.drop_false_no_steps(answer, CONTEXT)
    assert n == 1 and grounding.NO_STEPS_LINE not in out
    assert "Send HTR_MAIN_ON." in out


def test_no_steps_under_an_index_only_procedure_stays():
    answer = (f"FCP 7001: send HTR_MAIN_ON.\n"
              f"FCP 7002 - XYZ - HEATER OFF\n{grounding.NO_STEPS_LINE}")
    out, n = grounding.drop_false_no_steps(answer, CONTEXT)
    assert n == 0 and out == answer


def test_no_steps_with_no_procedure_named_before_it_stays():
    answer = f"{grounding.NO_STEPS_LINE}"
    out, n = grounding.drop_false_no_steps(answer, CONTEXT)
    assert n == 0 and out == answer


# Citations

def test_a_cited_answer_is_left_alone():
    text, cited = generate.check_citations("It is a halo orbit [Source 2].", 5)
    assert cited == [2] and generate.UNCITED_NOTE not in text


def test_a_citation_to_a_source_that_was_not_given_is_removed():
    text, cited = generate.check_citations(
        "It is a halo orbit [Source 9]. It is large [Source 1].", 5)
    assert cited == [1] and "Source 9" not in text


def test_an_uncited_answer_is_flagged():
    text, cited = generate.check_citations("It is a halo orbit.", 5)
    assert cited == [] and text.endswith(generate.UNCITED_NOTE)


def test_a_refusal_is_not_flagged():
    text, cited = generate.check_citations(generate.NOT_FOUND_MESSAGE, 5)
    assert text == generate.NOT_FOUND_MESSAGE


def test_a_refusal_with_citations_is_still_a_refusal():
    text = f"{generate.NOT_FOUND_MESSAGE} [Source 1] [Source 2]"
    assert generate.is_refusal(text)
    out, cited = generate.check_citations(text, 5)
    assert out == generate.NOT_FOUND_MESSAGE and cited == []


def test_the_note_asks_for_related_material_only_when_told():
    assert "other passages describe" in grounding.context_note(CONTEXT, related="any")
    assert "names that FCP" not in grounding.context_note(CONTEXT)


def test_the_v9_note_forbids_the_other_procedures_steps_by_number():
    note = grounding.context_note(CONTEXT, related="named")
    assert "Never give the steps of a different FCP (FCP 7001)" in note


def test_the_fixed_reply_with_more_after_it_must_cite():
    text = (f"FCP 7002 - XYZ - HEATER OFF\n{grounding.NO_STEPS_LINE}\n\n"
            "The heater is switched off when the panel is above its limit.")
    assert not grounding.only_no_steps(text)
    out, _ = generate.check_citations(text, 5)
    assert generate.UNCITED_NOTE in out


def test_the_fixed_reply_alone_is_only_no_steps():
    assert grounding.only_no_steps(f"FCP 7002 - XYZ - HEATER OFF\n{grounding.NO_STEPS_LINE}")
    assert grounding.only_no_steps(f"FCP 7002 - XYZ - HEATER OFF. {grounding.NO_STEPS_LINE}")


def test_a_lone_not_found_after_an_answer_is_taken_off():
    text = f"The heater is on below 5 C. [Source 1]\n\n{generate.NOT_FOUND_MESSAGE}"
    out, removed = generate.drop_trailing_not_found(text)
    assert removed and out == "The heater is on below 5 C. [Source 1]"


def test_a_bare_refusal_or_a_named_gap_is_kept():
    bare = generate.NOT_FOUND_MESSAGE
    assert generate.drop_trailing_not_found(bare) == (bare, False)
    named = f"The heater is on below 5 C. [Source 1]\n\nThe upper limit: {bare}"
    assert generate.drop_trailing_not_found(named) == (named, False)


def test_the_fixed_no_steps_reply_is_not_flagged():
    reply = f"FCP 7002 - XYZ - HEATER OFF\n{grounding.NO_STEPS_LINE}"
    text, _ = generate.check_citations(reply, 5)
    assert generate.UNCITED_NOTE not in text


# The command log

def record(n, mnemonic, cid, subsystem, time):
    return Hit(f"TCH_{n}", "", 1.0, "demo", {
        "content_type": "command_log", "date": "2031-01-02", "time": time,
        "mnemonic": mnemonic, "cid": cid, "subsystem": subsystem,
        "status": "DONE", "source_station": "AAA", "dest_station": "BBB",
        "orbit_no": "12"})


RECORDS = [record(1, "HTR_ON", "HTRS0001", "POWER", "01:00:00"),
           record(2, "HTR_ON", "HTRS0001", "POWER", "02:00:00"),
           record(3, "GYR_SEL", "GYRS0002", "DTG", "03:00:00")]


def test_a_date_lookup_is_recognised():
    plan = QueryPlan("command_history", ["commands"], date="2031-01-02")
    assert generate.is_log_lookup(RECORDS, plan)


def test_a_similarity_search_of_the_log_is_not_a_lookup():
    plan = QueryPlan("command_history", ["commands"])
    assert not generate.is_log_lookup(RECORDS, plan)


def test_records_for_another_date_are_not_a_lookup():
    plan = QueryPlan("command_history", ["commands"], date="2031-01-03")
    assert not generate.is_log_lookup(RECORDS, plan)


def test_the_summary_counts_every_record():
    plan = QueryPlan("command_history", ["commands"], date="2031-01-02")
    text = generate.answer_from_commands(RECORDS, plan, "what was sent")
    assert text.startswith("3 command records in the telecommand log "
                           "for date 2031-01-02.")
    assert "First at 2031-01-02 01:00:00, last at 2031-01-02 03:00:00." in text
    assert "HTR_ON (CID HTRS0001): 2 times, 01:00:00 to 02:00:00" in text
    assert "GYR_SEL (CID GYRS0002): 1 time, at 03:00:00" in text
    assert "Uplinked from AAA to BBB (3)." in text


def test_the_summary_narrows_to_a_subsystem_and_says_so():
    plan = QueryPlan("command_history", ["commands"], date="2031-01-02")
    text = generate.answer_from_commands(RECORDS, plan, "gyro commands sent")
    assert "1 of the 3 are DTG commands" in text
    assert "GYR_SEL" in text and "HTR_ON" not in text


def test_a_log_lookup_does_not_call_the_model():
    plan  = QueryPlan("command_history", ["commands"], date="2031-01-02")
    reply = generate.answer("what was sent on 2031-01-02", RECORDS, plan=plan)
    assert reply.path == "commands"


def test_what_is_answered_without_a_model():
    log  = QueryPlan("command_history", ["commands"], date="2031-01-02")
    fig  = QueryPlan("figure", ["captions"])
    plain = QueryPlan("command_history", ["commands"])
    caption = Hit("fig", "A diagram.", 0, "demo",
                  {"content_type": "image_caption", "page": 2})
    assert generate.answer_without_model(RECORDS, log, "q")[1] == "commands"
    assert generate.answer_without_model([caption], fig, "q")[1] == "caption"
    assert generate.answer_without_model([], plain, "q")[1] == "none"
    assert generate.answer_without_model(RECORDS, plain, "q") == ("", "")
    # A figure question whose top hit is not a caption goes to the model.
    assert generate.answer_without_model(RECORDS, fig, "q") == ("", "")


def test_the_demo_does_not_send_a_log_lookup_to_the_models():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))
    import server
    plan = QueryPlan("command_history", ["commands"], date="2031-01-02")
    saved = server.router.plan, server.retrieve.search, server.SAT
    try:
        server.router.plan = lambda question, retrieval: plan
        server.retrieve.search = lambda sats, question, top_k, plan: RECORDS
        server.SAT = type("Sat", (), {"retrieval": None})()
        job = server.prepare("what was sent on 2031-01-02")
    finally:
        server.router.plan, server.retrieve.search, server.SAT = saved
    assert job["fixed"].startswith("3 command records")
    assert "command-log lookup" in job["why"]


# Figures: a caption reply keeps the caption and its image to check against

def test_a_caption_reply_keeps_the_caption_and_its_image():
    figure = Hit("fig", "A block diagram of the heater loop.", 0, "demo",
                 {"doc_type": "MOR", "content_type": "image_caption",
                  "page": 4, "image_file": "images/fig_4_1.png"})
    plan  = QueryPlan("figure", ["captions"])
    reply = generate.answer("show the heater diagram", [figure], plan=plan)
    assert reply.path == "caption"
    assert "A block diagram of the heater loop." in reply.context
    assert "images/fig_4_1.png" in reply.context


# Procedures: one candidate each, steps attached afterwards

ROW  = Hit("row", "FCP 7001 - XYZ - HEATER ON", 0, "demo",
           {"doc_type": "FCP", "content_type": "toc_index", "fcp_number": "7001"})
BODY = Hit("body", "XYZ- 7001: HEATER ON\nPROCEDURE:\n...", 0, "demo",
           {"doc_type": "FCP", "content_type": "procedure", "fcp_number": "7001"})
PART = Hit("body_s1", "more steps", 0, "demo",
           {"doc_type": "FCP", "content_type": "procedure", "fcp_number": "7001"})
PROSE = Hit("prose", "About heaters.", 0, "demo",
            {"doc_type": "MOR", "content_type": "section"})
PROCEDURES = {"rows": {"7001": ROW}, "bodies": {"7001": BODY}}


def test_the_pieces_of_a_procedure_become_one_candidate():
    out = retrieve.rows_for_bodies([PROSE, PART, BODY, ROW], PROCEDURES)
    assert [h.chunk_id for h in out] == ["prose", "row"]


def test_the_row_is_swapped_for_its_steps():
    out = retrieve.steps_for_rows([ROW, PROSE], PROCEDURES)
    assert [h.chunk_id for h in out] == ["body", "prose"]


def test_a_row_with_no_steps_in_the_store_stays_a_row():
    other = Hit("row2", "FCP 7002 - XYZ - HEATER OFF", 0, "demo",
                {"doc_type": "FCP", "content_type": "toc_index",
                 "fcp_number": "7002"})
    out = retrieve.steps_for_rows([other], PROCEDURES)
    assert out[0].chunk_id == "row2"


def test_a_split_body_is_joined_without_its_overlap():
    whole = "".join(f"step {i} of the procedure. " for i in range(40))
    first, second = whole[:500], whole[300:]
    assert retrieve._join_pieces([first, second]) == whole


def test_a_whole_procedure_is_not_cut_at_the_prose_budget():
    long_body = Hit("b", "x" * 5000, 0, "demo", BODY.metadata)
    assert "x" * 5000 in retrieve.build_context([long_body])
    long_prose = Hit("p", "y" * 5000, 0, "demo", PROSE.metadata)
    assert "y" * 2001 not in retrieve.build_context([long_prose])


# Trying a failed request once more

def test_a_request_that_fails_once_is_tried_again():
    calls = []

    def flaky(messages, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            error = llm.LLMError("server busy")
            error.transient = True
            raise error
        return "ok"

    saved = llm._ollama_chat, llm.RETRY_WAIT_S, llm.BACKEND
    llm._ollama_chat, llm.RETRY_WAIT_S, llm.BACKEND = flaky, 0, "ollama"
    try:
        assert llm.chat([], model="m") == "ok" and len(calls) == 2
    finally:
        llm._ollama_chat, llm.RETRY_WAIT_S, llm.BACKEND = saved


def test_a_refused_request_is_not_tried_again():
    calls = []

    def refused(messages, **kwargs):
        calls.append(1)
        raise llm.LLMError("no such model")

    saved = llm._ollama_chat, llm.BACKEND
    llm._ollama_chat, llm.BACKEND = refused, "ollama"
    try:
        try:
            llm.chat([], model="m")
        except llm.LLMError:
            pass
        assert len(calls) == 1
    finally:
        llm._ollama_chat, llm.BACKEND = saved


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}  {exc}")
    print(f"\n  {len(tests) - failed} of {len(tests)} passed")
    sys.exit(1 if failed else 0)
