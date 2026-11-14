"""
MedSync clinical API - exposes the existing Python checker (interaction_checker.py / patient_screen.py /
drug_identifier.py, all reading Clinical_Safety_Database_MultiSource_v4.xlsx) over HTTP so the Ionic app can use it.

Run (from the repo root):
    server/.venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /api/health   -> {"status": "ok", ...}
    POST /api/check    -> medicine NAMES + patient profile  -> duplicates, interactions, patient alerts
    POST /api/scan     -> one PHOTO (multipart) + patient   -> same, plus the packages found in the photo

Educational prototype - not for real medication decisions.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import sys
import tempfile
import threading
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                     # the checker modules live in the repo root

import cv2                                        # noqa: E402
from dotenv import load_dotenv                    # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile   # noqa: E402
from fastapi.middleware.cors import CORSMiddleware                   # noqa: E402
from pydantic import BaseModel, Field                                # noqa: E402

load_dotenv(ROOT / "server" / ".env")             # VLM credentials (git-ignored); real environment variables win

import drug_identifier as di                                         # noqa: E402
import interaction_checker as ic                                     # noqa: E402
from server import vision                                            # noqa: E402
from patient_screen import Patient, screen_patient                   # noqa: E402

log = logging.getLogger("medsync")
MAX_UPLOAD = 15 * 1024 * 1024
MAX_MEDICINES = 40

@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=kb, daemon=True).start()      # warm the knowledge base without delaying startup
    yield


app = FastAPI(title="MedSync clinical API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.environ.get("MEDSYNC_CORS", "").split(",") if o],
    # dev server, Capacitor webviews (iOS capacitor://localhost, Android http(s)://localhost) and LAN testing
    allow_origin_regex=r"^(https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+)(:\d+)?|capacitor://localhost|ionic://localhost)$",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---- the knowledge base is expensive to build (reads a 350 KB workbook) -> load once, lazily --------------
_kb: Optional[ic.KnowledgeBase] = None
_kb_lock = threading.Lock()
_ocr: Optional[di.OCREngine] = None


def kb() -> ic.KnowledgeBase:
    global _kb
    with _kb_lock:
        if _kb is None:
            _kb = ic.KnowledgeBase(ic.DEFAULT_DB, ic.DEFAULT_ALIASES)
            log.info("knowledge base loaded: %d interaction pairs", len(_kb.pairs))
        return _kb


def ocr_engine() -> di.OCREngine:
    global _ocr
    with _kb_lock:
        if _ocr is None:
            _ocr = di.OCREngine("tesseract")
        return _ocr


# ---- request models ---------------------------------------------------------------------------------------
class PatientIn(BaseModel):
    age: Optional[float] = Field(default=None, ge=0, le=130)
    sex: Optional[str] = None                      # 'F' | 'M' (anything else is ignored)
    pregnant: Optional[bool] = None
    breastfeeding: bool = False
    ancestry: Optional[str] = None
    conditions: list[str] = Field(default_factory=list, max_length=50)
    allergies: list[str] = Field(default_factory=list, max_length=50)

    def to_patient(self) -> Optional[Patient]:
        pt = Patient(age=self.age, sex=self.sex, pregnant=self.pregnant, breastfeeding=self.breastfeeding,
                     ancestry=self.ancestry, conditions=self.conditions, allergies=self.allergies)
        return None if pt.empty else pt


class MedicineIn(BaseModel):
    ref: str = Field(max_length=100)               # the app's own id, echoed back so results can be matched
    name: str = Field(min_length=1, max_length=120)
    strength: Optional[str] = Field(default=None, max_length=40)


class CheckIn(BaseModel):
    medicines: list[MedicineIn] = Field(max_length=MAX_MEDICINES)
    patient: Optional[PatientIn] = None


# ---- shared result builder ---------------------------------------------------------------------------------
def _package_json(p: ic.Package, ref: Optional[str] = None) -> dict:
    d = asdict(p) | {"label": p.label}
    d.pop("box", None)
    d.pop("ocr_text", None)                        # typed names: the "OCR text" is just the name
    if ref is not None:
        d["ref"] = ref
    return d


def _build(k: ic.KnowledgeBase, pkgs: list[ic.Package], patient: Optional[Patient]) -> dict:
    dups = ic.find_duplicates(pkgs)
    found, no_rec, not_ass = ic.check_interactions(k, pkgs)
    pat = screen_patient(k.rules, pkgs, patient) if patient else None
    return {"duplicates": dups, "interactions": found, "no_record": no_rec, "not_assessable": not_ass, "patient": pat}


def _merge(ocr_pkgs: list[ic.Package], vis_pkgs: list[ic.Package], seen: dict[int, dict]):
    """Keep every package OCR identified (verified against the pixels by Tesseract); replace OCR's unidentified
    boxes with what the vision model read, skipping vision items OCR already found. Ids are renumbered 1..n."""
    kept = [p for p in ocr_pkgs if p.status == "identified"]
    covered = Counter(frozenset(p.ingredients) for p in kept)
    merged: list[ic.Package] = list(kept)
    sources: dict[int, str] = {}
    seen_by_id: dict[int, dict] = {}
    from_vision: list[tuple[ic.Package, dict]] = []
    for p in vis_pkgs:
        key = frozenset(p.ingredients)
        if p.status == "identified" and covered[key] > 0:
            covered[key] -= 1                       # same medicine OCR already found
            continue
        from_vision.append((p, seen[p.pkg_id]))
    for i, p in enumerate(merged, start=1):
        p.pkg_id = i
        sources[i] = "ocr"
    for p, info in from_vision:
        p.pkg_id = len(merged) + 1
        merged.append(p)
        sources[p.pkg_id] = "vision"
        seen_by_id[p.pkg_id] = info
    return merged, sources, seen_by_id


# ---- endpoints ---------------------------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    k = kb()
    return {"status": "ok", "interaction_pairs": len(k.pairs), "core_drugs": len(k.core_ingredients),
            "database": ic.DEFAULT_DB.name, "vision_fallback": vision.available()}


@app.post("/api/check")
def check(body: CheckIn) -> dict:
    """Identify each typed medicine name, then check duplicates, interactions and the patient screen."""
    k = kb()
    pkgs: list[ic.Package] = []
    refs: dict[int, str] = {}
    for i, m in enumerate(body.medicines, start=1):
        text = f"{m.name} {m.strength or ''}".strip()
        tokens = [(w, 1.0) for w in text.replace("/", " ").replace(",", " ").split() if w]
        pkg = ic.read_package(k, i, tokens)
        pkg.ocr_text = text
        if pkg.brand is None and pkg.ingredients:   # typed generic name: show it as the user wrote it, not "(brand not read)"
            pkg.brand = m.name
        pkgs.append(pkg)
        refs[i] = m.ref
    res = _build(k, pkgs, body.patient.to_patient() if body.patient else None)
    res["packages"] = [_package_json(p, refs[p.pkg_id]) for p in pkgs]
    return res


@app.post("/api/scan")
def scan(image: UploadFile = File(...), patient: Optional[str] = Form(default=None),
         annotate: bool = Form(default=False)) -> dict:
    """Photo of one or more medicine boxes -> packages found + the same checks as /api/check."""
    data = image.file.read(MAX_UPLOAD + 1)
    if not data:
        raise HTTPException(400, "empty image")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "image too large (max 15 MB)")
    pat: Optional[Patient] = None
    if patient:
        try:
            pat = PatientIn(**json.loads(patient)).to_patient()
        except Exception as e:                      # noqa: BLE001
            raise HTTPException(422, f"invalid patient JSON: {e}") from e

    k = kb()
    # 1) OpenCV segmentation + Tesseract OCR: free and runs on this machine
    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        suffix = ".jpg"
    res: Optional[dict] = None
    ocr_error: Optional[Exception] = None
    with tempfile.TemporaryDirectory() as tmp:     # the checker reads from a path
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        try:
            res = ic.analyse(path, k, engine="tesseract", ocr=ocr_engine(), patient=pat)
        except Exception as e:                      # noqa: BLE001 - unreadable image, OCR missing, ...
            log.exception("scan failed")
            ocr_error = e
    ocr_pkgs: list[ic.Package] = res["packages"] if res else []

    # 2) vision-language model, ONLY when OCR left something unidentified (or found nothing)
    ocr_complete = bool(ocr_pkgs) and all(p.status == "identified" for p in ocr_pkgs)
    if vision.available() and not ocr_complete:
        try:
            vis_pkgs, seen = vision.read_packages(k, data)
        except vision.VisionError as e:
            log.warning("vision fallback unavailable, keeping the OCR result: %s", e)
        else:
            if vis_pkgs:
                merged, sources, seen_by_id = _merge(ocr_pkgs, vis_pkgs, seen)
                out = _build(k, merged, pat)
                out["image"] = image.filename or "upload"
                out["engine"] = "ocr+vision" if any(s == "ocr" for s in sources.values()) else "vision"
                out["packages"] = [_package_json(p) | {"source": sources[p.pkg_id]}
                                   | ({"vision": seen_by_id[p.pkg_id]} if p.pkg_id in seen_by_id else {})
                                   for p in merged]
                return out        # no annotated image: its box numbers would not match the merged list

    if res is None:
        raise HTTPException(422, f"could not analyse the image: {ocr_error}") from ocr_error
    out = ic.to_json(res)
    out["image"] = image.filename or "upload"
    out["engine"] = "ocr"
    out["packages"] = [{k_: v for k_, v in p.items() if k_ != "ocr_text"} | {"source": "ocr"} for p in out["packages"]]
    if annotate:
        ok, buf = cv2.imencode(".png", ic.annotate(res))
        if ok:
            out["annotated_png_base64"] = base64.b64encode(buf.tobytes()).decode()
    return out
