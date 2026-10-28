"""
Query routing.

Every question gets one plan before anything is searched: what it is
asking for, which store can answer it, and any exact values worth
filtering on. Retrieval and answer generation both read that same plan,
so they can no longer disagree about whether a question wants a figure.

Three routers, chosen per satellite with retrieval.router:

    rules   the regular expressions below
    soft    the rules decide commands and figures, but procedure
            questions are not classified at all: the procedure document
            is fused into every knowledge search instead
    llm     the local text model decides intent and store; the rules
            still pull out dates, CIDs and mnemonics, and take over
            whenever the model is unavailable or answers off-schema

Identifiers never come from the model. A date or CID that is not written
in the question cannot end up in a filter.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

INTENTS = ("procedure", "which_procedure", "figure",
           "command_history", "concept")
ROUTERS = ("rules", "soft", "llm")

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "routing.jsonl"


@dataclass
class QueryPlan:
    intent:   str
    stores:   list = field(default_factory=lambda: ["knowledge"])
    date:     Optional[str] = None
    cid:      Optional[str] = None
    mnemonic: Optional[str] = None     # a candidate; retrieval checks it
    router:   str = "rules"            # the router that was configured
    source:   str = "rules"            # the one that actually decided
    note:     str = ""

    def describe(self) -> str:
        bits = [self.intent, "+".join(self.stores)]
        for key in ("date", "cid", "mnemonic"):
            if getattr(self, key):
                bits.append(f"{key}={getattr(self, key)}")
        bits.append(f"via {self.source}")
        if self.note:
            bits.append(f"({self.note})")
        return "  ".join(bits)


# Rules

DATE_RE = re.compile(
    r'\b(20\d{2})[-/\s](0[1-9]|1[0-2])[-/\s](0[1-9]|[12]\d|3[01])\b')

# Order matters. The first pattern to match wins, so specific component
# names come first and generic words last. "power" appears in almost
# every switch-on question ("gyro power on", "wheel power on", "HGA motor
# power on"), so a POWER entry matching a bare \bpower\b near the top
# swallows every one of them and routes the query to the wrong subsystem.
#
# Only diagnose.py reads this now. Retrieval does not filter on subsystem.
SUBSYSTEM_PATTERNS = [
    ("DTG",        r'\bDTG\b|\bgyro\b|\bCSA\b'),
    ("SENSOR",     r'\bstar\s*sensor\b|\bSS-?[12]\b'),
    ("WHEEL",      r'\bwheel\b|\bWDE\b|\bsuspension\b|\bmomentum\s*dump'),
    ("MECHANISM",  r'\bHGA\b|\bmechanism\b|\bSPDM\b|\bMCE\b|\bMEPU\b|\bdeploy'),
    ("PROPULSION", r'\bthruster\b|\bpropulsion\b|\blatch\s*valve\b'
                   r'|\boxidiser\b|\bpyro\b'),
    ("TTC_XBAND",  r'\bTTC\b|\bX.band\b|\btransponder\b|\bTWTA\b|\bXBS\b'
                   r'|\btone\s*ranging\b'),
    ("ODHS",       r'\bODHS\b|\bSSR\b|\bDFU\b|\bBDH\b|\bdata\s*handling\b'),
    ("PAYLOAD",    r'\bPAPA\b|\bASPEX\b|\bVELC\b|\bHELIOS\b|\bSOLEXS?\b'
                   r'|\bSUIT\b|\bmagnetometer\b|\bpayload\b'),
    ("OBC",        r'\bOBC\b|\bCPU\b|\bEEPROM\b|\bwatchdog\b'
                   r'|\btelecommand\s*link\b'),
    ("AOCS",       r'\bAOCS\b|\bsun\s*acquisition\b|\battitude\b'
                   r'|\brate\s*damp'),
    # Generic last. A bare "power" is too weak a signal to route on.
    ("POWER",      r'\bFPGA\b|\bTCR\b|\bDC.DC\b|\bbattery\b'
                   r'|\bpower\s*(system|bus|supply)\b'),
]

# "show me" used to be in this list. It turned "show me the steps to
# switch on the OBC" into a figure request, and a figure request is
# answered from a caption without the text model seeing the procedure.
DIAGRAM_RE = re.compile(
    r'\b(diagram|figure|chart|plot|image|picture|schematic|flow\s*chart|'
    r'block\s*diagram|visual)\b', re.IGNORECASE)

# A question that asks how to do something wants the procedure itself.
# Without this, a manual section that merely mentions the subsystem
# competes on equal terms with the procedure that answers the question.
#
# Deliberately left as it was. It is broad ("power budget" matches), and
# the llm and soft routers exist to be measured against it rather than
# to hide its behaviour behind a quieter pattern.
HOWTO_RE = re.compile(
    r'\b(fcp|procedure|how\s+to|how\s+do|how\s+is|steps?|switch\w*|'
    r'power\w*|turn\w*|enabl\w*|disabl\w*|reset|select\w*|activat\w*|'
    r'deactivat\w*|operat\w*|execut\w*|configur\w*|uplink\w*|command\w*)\b',
    re.IGNORECASE)

CID_RE = re.compile(r'\bCID\s*[:#]?\s*([A-Z]{2,}\d+)\b', re.IGNORECASE)

# Mnemonics are upper-case words joined by underscores (and sometimes
# hyphens). A candidate becomes a filter only if the command store
# actually holds that mnemonic, so a false match costs nothing.
MNEMONIC_RE = re.compile(r'\b(?=[A-Z0-9-]*_)[A-Z0-9]+(?:[-_][A-Z0-9]+)+\b',
                         re.IGNORECASE)


def wants_commands(query: str) -> bool:
    """
    Route to the telecommand log only on a real command-log signal.

    A bare "command" is far too weak: "programmable command sequencer",
    "configurable command block" and "telecommand link test" are all
    procedure questions, and sending them to the command store loses them
    entirely. The phrasing has to be about commands that were uplinked,
    a CID, or a date.
    """
    if DATE_RE.search(query) or CID_RE.search(query):
        return True
    return bool(re.search(
        r'\bcommands?\s+(were\s+|that\s+were\s+)?(uplink|sent|issued|executed)'
        r'|\buplinked\s+(on|from|to)\b'
        r'|\btelecommand\s+history\b|\bcommand\s+history\b'
        r'|\borbit\s+\d{3,}\b',
        query, re.IGNORECASE))


def find_date(query: str) -> Optional[str]:
    m = DATE_RE.search(query)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def find_cid(query: str) -> Optional[str]:
    m = CID_RE.search(query)
    return m.group(1).upper() if m else None


def find_mnemonic(query: str) -> Optional[str]:
    m = MNEMONIC_RE.search(query)
    return m.group(0).upper() if m else None


def find_subsystem(query: str) -> Optional[str]:
    for sub, pattern in SUBSYSTEM_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return sub
    return None


def wants_diagram(query: str) -> bool:
    return bool(DIAGRAM_RE.search(query))


def wants_procedure(query: str) -> bool:
    return bool(HOWTO_RE.search(query))


def asks_which_procedure(query: str) -> bool:
    """
    'which FCP covers X' wants the index; 'how do I do X' wants the
    procedure. Both search the procedure document today, but the plan
    keeps them apart so routing can be scored on the difference.
    """
    return bool(re.search(
        r'\b(which|what)\s+(fcp|procedure|number)\b|'
        r'\bfcp\s+(number|list|index)\b|\blist\s+(all\s+)?(fcp|procedures)\b',
        query, re.IGNORECASE))


def rule_plan(query: str) -> QueryPlan:
    """The regular-expression router. The first rule that matches wins."""
    ids = {"date": find_date(query), "cid": find_cid(query),
           "mnemonic": find_mnemonic(query)}
    if wants_commands(query):
        return QueryPlan("command_history", ["commands"], **ids)
    if wants_diagram(query):
        return QueryPlan("figure", **ids)
    if asks_which_procedure(query):
        return QueryPlan("which_procedure", **ids)
    if wants_procedure(query):
        return QueryPlan("procedure", **ids)
    return QueryPlan("concept", **ids)


# Model router

LLM_MODEL   = "qwen2.5:7b-instruct"
LLM_TIMEOUT = 60        # seconds; the first call may have to load the model

# Same value as generate.NUM_CTX. Ollama reloads a model whenever its
# options change, so a smaller window here would cost a reload between
# routing a question and answering it.
LLM_NUM_CTX = 8192


class Route(BaseModel):
    intent: Literal["procedure", "which_procedure", "figure",
                    "command_history", "concept"]
    stores: list[Literal["knowledge", "commands"]]


# The examples are invented for this prompt. None of them comes from the
# benchmark or the routing test set, so neither measures a memorised
# answer.
ROUTER_PROMPT = """You route questions for a spacecraft operations assistant.
There are two stores:
  knowledge  the mission documents: flight control procedures (FCPs), the
             mission operations report, the operations manual and the
             operations guidelines, including captions of their figures
  commands   the telecommand log: which commands were actually sent to the
             spacecraft, when, with which CID and with what status

Choose exactly one intent:
  procedure        how to carry out an operation, or its steps.
                   "Steps to arm the pyro circuit"
  which_procedure  which FCP, or which procedure number, covers something.
                   "Which FCP stows the antenna pointing mechanism?"
  figure           a diagram, schematic, plot, chart or picture.
                   "Schematic of the solar array drive electronics"
  command_history  what was commanded: commands sent, their times, CIDs or
                   status as recorded in the log.
                   "Which commands failed during the morning pass?"
  concept          an explanation, description, design, budget, limit or
                   guideline.
                   "What is the thermal design philosophy?"

stores lists where the answer can be found. Use ["commands"] for
command_history and ["knowledge"] for everything else, unless the question
clearly needs both the log and the documents.
Reply with JSON only."""

_client = None


def _ollama():
    global _client
    if _client is None:
        import ollama
        _client = ollama.Client(timeout=LLM_TIMEOUT)
    return _client


def llm_plan(query: str, model: str = LLM_MODEL) -> QueryPlan:
    base = rule_plan(query)
    base.router = "llm"

    # "CID X" can only mean the log. Everything else, dates included, goes
    # to the model: a date alone does not make a question a log lookup.
    if base.cid:
        base.note = "explicit CID"
        return base

    try:
        reply = _ollama().chat(
            model    = model,
            messages = [{"role": "system", "content": ROUTER_PROMPT},
                        {"role": "user",   "content": query}],
            format   = Route.model_json_schema(),
            options  = {"temperature": 0, "num_ctx": LLM_NUM_CTX},
        )
        route = Route.model_validate_json(reply["message"]["content"])
    except Exception as exc:
        # Ollama not running, a timeout, or JSON outside the schema.
        base.note = f"model unavailable: {type(exc).__name__}"
        return base

    stores = list(dict.fromkeys(route.stores)) or ["knowledge"]
    if route.intent == "command_history" and "commands" not in stores:
        stores = ["commands"]
    return QueryPlan(route.intent, stores, date=base.date, cid=base.cid,
                     mnemonic=base.mnemonic, router="llm", source="llm")


def plan(query: str, cfg: dict = None) -> QueryPlan:
    """The plan for one question, from the router the config names."""
    cfg  = cfg or {}
    mode = cfg.get("router") or "rules"
    if mode not in ROUTERS:
        raise ValueError(f"unknown router {mode!r}, "
                         f"expected one of: {', '.join(ROUTERS)}")
    if mode == "llm":
        return llm_plan(query, cfg.get("router_model") or LLM_MODEL)
    result = rule_plan(query)
    result.router = mode
    return result


def log_plan(query: str, plan: QueryPlan, hits: list = None) -> None:
    """
    Append one routing decision to logs/routing.jsonl.

    The log holds the questions people actually asked, so it stays out of
    git. Failing to write it never affects the answer.
    """
    record = {"time": datetime.now().isoformat(timespec="seconds"),
              "query": query, **asdict(plan),
              "sources": [f"{h.satellite}:{h.chunk_id}" for h in hits or []]}
    try:
        LOG_PATH.parent.mkdir(exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass
