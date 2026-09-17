"""Download public ISRO Aditya-L1 documents into satellites/aditya-l1/documents/.

The real Aditya-L1 mission documents are confidential, so this dataset uses
ISRO/ISSDC's public documents instead. URLs below were verified live (HTTP
200, content-type: application/pdf) before being added here.
"""

import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "satellites" / "aditya-l1" / "documents"

DOCUMENTS = [
    {
        "filename": "adityal1_mission_booklet_mor.pdf",
        "source_url": "https://www.isro.gov.in/media_isro/pdf/Aditya_L1_Booklet.pdf",
        "title": "Aditya-L1 Mission Booklet",
    },
    {
        "filename": "adityal1_spacecraft_orbit_ops_manual.pdf",
        "source_url": "https://www.isro.gov.in/media_isro/pdf/Aditya_L1_second_booklet.pdf",
        "title": "Aditya-L1 Spacecraft and Orbit Booklet",
    },
    {
        "filename": "adityal1_announcement_of_opportunity_guidelines.pdf",
        "source_url": "https://www.isro.gov.in/media_isro/pdf/AdityaL1AO_1cycle050125.pdf",
        "title": "Aditya-L1 Announcement of Opportunity, First Cycle",
    },
    {
        "filename": "adityal1_mission_brochure_handbook.pdf",
        "source_url": "https://www.isro.gov.in/media_isro/pdf/AdityaL1_Mission_Brochure.pdf",
        "title": "Aditya-L1 Mission Brochure",
    },
]

MIN_BYTES = 20_000
UA = "Mozilla/5.0 (SatelliteRAG-fetch-script; research use)"


def fetch(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=60) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} for {url}")
        return resp.read()


def main() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for doc in DOCUMENTS:
        dest = DOCS_DIR / doc["filename"]
        print(f"Fetching {doc['source_url']} -> {dest.name}")
        data = fetch(doc["source_url"])

        if not data.startswith(b"%PDF-"):
            print(f"  SKIP: not a PDF (bad magic bytes) for {doc['filename']}", file=sys.stderr)
            continue
        if len(data) < MIN_BYTES:
            print(f"  SKIP: too small ({len(data)} bytes) for {doc['filename']}", file=sys.stderr)
            continue

        dest.write_bytes(data)
        sha256 = hashlib.sha256(data).hexdigest()
        print(f"  OK: {len(data):,} bytes, sha256={sha256[:12]}...")

        manifest.append(
            {
                "filename": doc["filename"],
                "source_url": doc["source_url"],
                "title": doc["title"],
                "retrieved_date": date.today().isoformat(),
                "bytes": len(data),
                "sha256": sha256,
            }
        )

    manifest_path = DOCS_DIR / "SOURCES.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest: {manifest_path} ({len(manifest)} documents)")
    return 0 if manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())
