#!/usr/bin/env python3
"""
Satellite RAG command line.

Everyday use is one command per satellite:

    python cli.py build aditya-l1        extract, caption, embed
    python cli.py query aditya-l1 "how to switch on OBC"

Setup, once:

    python cli.py prefetch minilm-l6     download the embedding model
    python cli.py models                 what is cached

Per satellite:

    python cli.py new chandrayaan-3      make the folder
    python cli.py scan                   check document detection
    python cli.py list                   state of everything
    python cli.py build all              every satellite

The individual stages, when one needs redoing on its own:

    python cli.py extract aditya-l1 --force
    python cli.py caption aditya-l1
    python cli.py ingest  aditya-l1 --reset

Searching:

    python cli.py query aditya-l1 "MRS diagram" --doc-types MOR GUIDELINES
    python cli.py query all "safe mode recovery"
    python cli.py query aditya-l1 "how to switch on OBC" --router llm
    python cli.py shell aditya-l1
"""

import argparse
import time
from pathlib import Path

from core import (caption, embeddings, extract, generate, ingest, llm,
                  registry, rerank, retrieve, router)


def _table(rows, headers, widths):
    print("\n  " + "".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  " + "-" * sum(widths))
    for row in rows:
        print("  " + "".join(str(c).ljust(w) for c, w in zip(row, widths)))
    print()


def cmd_models(args):
    print(f"\n  models cached in {registry.MODELS_DIR}")
    _table(
        [(n, s.dim, "yes" if embeddings.is_cached(n) else "no", s.note)
         for n, s in embeddings.MODEL_REGISTRY.items()],
        ["name", "dim", "cached", "notes"], [16, 7, 9, 52])
    print(f"  download with: python cli.py prefetch <name>\n")


def cmd_prefetch(args):
    names = args.models or [embeddings.DEFAULT_MODEL]
    print(f"\n  target: {registry.MODELS_DIR}\n")
    for name in names:
        try:
            embeddings.prefetch(name)
        except Exception as exc:
            print(f"  {name:<14} FAILED: {exc}")
    if not args.no_reranker:
        rerank.prefetch()
    print()


def cmd_new(args):
    sat = registry.create_satellite(args.slug, args.name)
    print(f"\n  created {sat.root}")
    print(f"\n  next:")
    print(f"    1. copy the documents into {sat.documents_dir}")
    print(f"    2. python cli.py build {sat.slug}")
    print(f"\n  no filenames to type in: documents are matched by name.\n")


def cmd_list(args):
    sats = registry.list_satellites()
    if not sats:
        print(f"\n  no satellites yet under {registry.SATELLITES_DIR}")
        print(f"\n  add one with:  python cli.py new <slug>")
        print(f"  or make the folder and copy the documents in:")
        print(f"    {registry.SATELLITES_DIR}\\<slug>\\documents\\\n")
        return

    rows = []
    for s in sats:
        rows.append((
            s.slug, s.embedding_model, f"{len(s.documents)}/5",
            "yes" if s.is_extracted() else "no",
            "yes" if s.is_ingested()  else "no",
            s.name))
    _table(rows, ["slug", "model", "docs", "extracted", "built", "name"],
           [20, 15, 7, 11, 7, 24])

    stale = [s for s in sats if s.missing_types() or s.unassigned_files()]
    if stale:
        print(f"  {len(stale)} satellite(s) need attention: "
              f"python cli.py scan\n")


def cmd_scan(args):
    sats = registry.resolve(args.satellite) if args.satellite \
           else registry.list_satellites()
    if not sats:
        print(f"\n  no satellites under {registry.SATELLITES_DIR}\n")
        return

    for s in sats:
        print(f"\n  {s.slug}  ({s.name})")
        print(f"  {s.documents_dir}")
        docs = s.documents
        if not docs:
            print(f"      no documents found, copy them into the folder")
        for doc_type in registry.DOC_TYPES:
            entry = docs.get(doc_type)
            if not entry:
                print(f"      {doc_type:<12} -")
                continue
            missing = "" if entry["path"].exists() else "   FILE NOT FOUND"
            tag     = "" if entry["source"] == "detected" else "   (config)"
            print(f"      {doc_type:<12} {entry['file']}{tag}{missing}")

        unknown = s.unassigned_files()
        if unknown:
            print(f"      not recognised:")
            for p in unknown:
                print(f"          {p.name}")
            print(f"      rename after the type, or map in "
                  f"{registry.CONFIG_NAME}")
    print()


def cmd_extract(args):
    sats = registry.resolve(args.satellite)
    results = extract.extract_many(sats, force=args.force)
    _table([(r["satellite"], r.get("chunks", 0), r.get("commands", 0),
             r.get("images", 0), r["status"]) for r in results],
           ["satellite", "chunks", "commands", "images", "status"],
           [20, 9, 10, 8, 22])


def cmd_caption(args):
    sats = registry.resolve(args.satellite)
    results = caption.caption_many(
        sats, model=args.model, force=args.force,
        min_bytes=args.min_bytes, limit=args.limit)
    _table([(r["satellite"], r.get("captioned", 0), r.get("failed", 0),
             r.get("total", 0), r["status"]) for r in results],
           ["satellite", "new", "failed", "total", "status"],
           [20, 7, 8, 8, 22])


def cmd_ingest(args):
    sats = registry.resolve(args.satellite)
    results = ingest.ingest_many(sats, reset=args.reset, device=args.device)
    _table([(r["satellite"], r.get("model", ""), r.get("knowledge", 0),
             r.get("commands", 0), r["status"]) for r in results],
           ["satellite", "model", "knowledge", "commands", "status"],
           [20, 15, 11, 10, 22])


def cmd_build(args):
    """Extract, caption and embed in one pass."""
    sats = registry.resolve(args.satellite)
    if not sats:
        print("  no satellites matched")
        return

    print(f"\n  building {len(sats)} satellite(s): "
          f"{', '.join(s.slug for s in sats)}")

    summary = []
    for sat in sats:
        row = {"satellite": sat.slug, "chunks": 0, "captions": 0,
               "vectors": 0, "status": "ok"}
        try:
            r = extract.extract(sat, force=args.force)
            row["chunks"] = r.get("chunks", 0)
            if r["status"] not in ("ok", "cached"):
                row["status"] = r["status"]
                summary.append(row)
                continue

            if not args.skip_captions:
                c = caption.caption(sat, model=args.model, force=args.force)
                row["captions"] = c.get("total", 0)

            i = ingest.ingest(sat, reset=args.reset or args.force,
                              device=args.device)
            row["vectors"] = i.get("knowledge", 0)
            row["status"]  = i["status"]
        except Exception as exc:
            row["status"] = f"error: {str(exc)[:40]}"
            print(f"    FAILED: {exc}")
        summary.append(row)

    _table([(r["satellite"], r["chunks"], r["captions"], r["vectors"],
             r["status"]) for r in summary],
           ["satellite", "chunks", "captions", "vectors", "status"],
           [20, 9, 10, 9, 26])

    ready = [r["satellite"] for r in summary if r["status"] == "ok"]
    if ready:
        print(f"  try:  python cli.py query {ready[0]} \"your question\"\n")


def run_query(sats, query, top_k, doc_types, subsystem, show_text,
              no_answer=False, model=generate.DEFAULT_MODEL):
    plan = router.plan(query, sats[0].retrieval)
    print(f"route: {plan.describe()}\n")
    if no_answer:
        hits = retrieve.search(sats, query, top_k=top_k,
                               doc_types=doc_types, subsystem=subsystem,
                               plan=plan)
        router.log_plan(query, plan, hits)
    else:
        # Timed in two parts, so a demo can show where the seconds go: the
        # search runs on this machine, the answer on whichever server
        # SATRAG_LLM points at. The first search in a process also loads
        # the embedding and reranker models, so it is the slowest.
        t0 = time.perf_counter()
        hits = retrieve.search(sats, query, top_k=top_k,
                               doc_types=doc_types, subsystem=subsystem,
                               plan=plan)
        t1 = time.perf_counter()
        answer = generate.generate(query, hits, model=model,
                                   show_satellite=len(sats) > 1, plan=plan)
        t2 = time.perf_counter()
        router.log_plan(query, plan, hits)
        print(f"{answer}\n")
        where = llm.VLLM_URL if llm.BACKEND == "vllm" else llm.OLLAMA_URL
        print(f"[time] search {t1 - t0:.1f} s  |  answer {t2 - t1:.1f} s "
              f"({llm.BACKEND} at {where})  |  total {t2 - t0:.1f} s\n")
        # The model cited none of these when it said nothing matched, so
        # listing them as "Sources" would misrepresent nearest-neighbor
        # noise as relevant citations. Only a reply that is the refusal and
        # nothing else counts: the 7B sometimes ends a real answer with the
        # sentence, and that answer's sources were being hidden.
        if answer.strip().rstrip(".") == generate.NOT_FOUND_MESSAGE.rstrip("."):
            return hits
        print("Sources:")
    print(retrieve.format_hits(hits, show_satellite=len(sats) > 1,
                               show_text=show_text))
    return hits


def use_router(sats, mode):
    """Override the router for this run without touching satellite.yaml."""
    if not mode:
        return
    for s in sats:
        cfg = dict(s.config.get("retrieval") or {})
        cfg["router"] = mode
        s.config["retrieval"] = cfg


def cmd_query(args):
    sats = registry.resolve(args.satellite)
    if not sats:
        print("  no satellites matched")
        return
    use_router(sats, args.router)

    if args.file:
        if args.query:
            raise SystemExit("pass either a query or --file, not both")
        lines = Path(args.file).read_text(encoding="utf-8").splitlines()
        questions = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
        for i, q in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] {q}\n")
            run_query(sats, q, args.k, args.doc_types,
                      args.subsystem, args.show_text,
                      no_answer=args.no_answer, model=args.model)
        return

    if not args.query:
        raise SystemExit("pass a query or --file")

    print()
    run_query(sats, args.query, args.k, args.doc_types,
              args.subsystem, args.show_text,
              no_answer=args.no_answer, model=args.model)
    print()


def cmd_shell(args):
    sats = registry.resolve(args.satellite)
    use_router(sats, args.router)
    print(f"\n  searching: {', '.join(s.slug for s in sats)}")
    print(f"  :sat <slug|all>   switch     :k <n>    result count")
    print(f"  :doc <TYPE ...>   filter     :text     toggle snippets")
    print(f"  :exit             quit\n")

    top_k, doc_types, show_text = args.k, args.doc_types, args.show_text

    while True:
        try:
            line = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == ":exit":
            break
        if line.startswith(":sat "):
            try:
                sats = registry.resolve(line.split(maxsplit=1)[1].split())
                use_router(sats, args.router)
                print(f"  -> {', '.join(s.slug for s in sats)}")
            except Exception as exc:
                print(f"  {exc}")
            continue
        if line.startswith(":k "):
            top_k = int(line.split()[1])
            print(f"  -> top_k {top_k}")
            continue
        if line.startswith(":doc "):
            parts     = line.split()[1:]
            doc_types = None if parts == ["off"] else [p.upper() for p in parts]
            print(f"  -> doc_types {doc_types or 'all'}")
            continue
        if line == ":text":
            show_text = not show_text
            print(f"  -> snippets {'on' if show_text else 'off'}")
            continue

        print()
        try:
            run_query(sats, line, top_k, doc_types, None, show_text)
        except Exception as exc:
            print(f"  error: {exc}")
        print()


def main():
    ap = argparse.ArgumentParser(
        description="Satellite RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = ap.add_subparsers(dest="command")

    sub.add_parser("models", help="show the embedding model registry")

    p = sub.add_parser("prefetch", help="download embedding models")
    p.add_argument("models", nargs="*")
    p.add_argument("--no-reranker", action="store_true",
                   help="skip the cross-encoder reranker download")

    p = sub.add_parser("new", help="scaffold a satellite folder")
    p.add_argument("slug")
    p.add_argument("--name", default=None)

    sub.add_parser("list", help="show satellites and their state")

    p = sub.add_parser("scan", help="check document detection")
    p.add_argument("satellite", nargs="*", default=None)

    p = sub.add_parser("build", help="extract, caption and embed")
    p.add_argument("satellite", nargs="+")
    p.add_argument("--force", action="store_true",
                   help="redo work that is already done")
    p.add_argument("--reset", action="store_true",
                   help="rebuild the vector store from scratch")
    p.add_argument("--skip-captions", action="store_true",
                   help="skip the vision model, text only")
    p.add_argument("--model",  default=caption.DEFAULT_MODEL,
                   help="Ollama vision model for captions")
    p.add_argument("--device", default=None, choices=["cpu", "cuda"])

    p = sub.add_parser("extract", help="read documents into JSON")
    p.add_argument("satellite", nargs="+")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("caption", help="describe figures with a vision model")
    p.add_argument("satellite", nargs="+")
    p.add_argument("--model",     default=caption.DEFAULT_MODEL)
    p.add_argument("--force",     action="store_true")
    p.add_argument("--min-bytes", type=int, default=caption.MIN_BYTES)
    p.add_argument("--limit",     type=int, default=0,
                   help="stop after N images, for a quick trial")

    p = sub.add_parser("ingest", help="embed into the vector store")
    p.add_argument("satellite", nargs="+")
    p.add_argument("--reset",  action="store_true")
    p.add_argument("--device", default=None, choices=["cpu", "cuda"])

    p = sub.add_parser("query", help="one-shot search")
    p.add_argument("satellite", nargs=1)
    p.add_argument("query", nargs="?", default=None,
                   help="the question (omit when using --file)")
    p.add_argument("--file", default=None,
                   help="path to a text file of questions, one per line "
                        "(# lines and blank lines are skipped); runs each "
                        "one in turn instead of a single query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--doc-types", nargs="*", default=None)
    p.add_argument("--subsystem", default=None)
    p.add_argument("--show-text", action="store_true")
    p.add_argument("--no-answer", action="store_true",
                   help="show retrieved sources only, skip the model")
    p.add_argument("--model", default=generate.DEFAULT_MODEL,
                   help="Ollama text model used for the answer")
    p.add_argument("--router", choices=router.ROUTERS, default=None,
                   help="override the satellite's router for this run")

    p = sub.add_parser("shell", help="interactive search")
    p.add_argument("satellite", nargs="+")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--doc-types", nargs="*", default=None)
    p.add_argument("--show-text", action="store_true")
    p.add_argument("--router", choices=router.ROUTERS, default=None,
                   help="override the satellite's router for this run")

    args = ap.parse_args()
    if not args.command:
        ap.print_help()
        return

    {
        "models":   cmd_models,  "prefetch": cmd_prefetch,
        "new":      cmd_new,     "list":     cmd_list,
        "scan":     cmd_scan,    "build":    cmd_build,
        "extract":  cmd_extract, "caption":  cmd_caption,
        "ingest":   cmd_ingest,  "query":    cmd_query,
        "shell":    cmd_shell,
    }[args.command](args)


if __name__ == "__main__":
    main()
