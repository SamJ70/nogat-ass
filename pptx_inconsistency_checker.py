#!/usr/bin/env python3
"""
pptx_inconsistency_checker.py

Single-file CLI tool to analyze a .pptx (and optionally an images folder)
for factual / logical inconsistencies across slides.

Usage:
    python pptx_inconsistency_checker.py /path/to/deck.pptx [--images /path/to/images_dir] [--out report.json]

Requirements:
    pip install python-pptx pillow pytesseract google-genai python-dateutil regex fuzzywuzzy python-Levenshtein
    tesseract-ocr system binary installed
    set GEMINI_API_KEY env var to your Gemini API key
"""

import os
import sys
import json
import tempfile
import re
import argparse
from collections import defaultdict
from datetime import datetime
from dateutil import parser as dateparser

from pptx import Presentation
from PIL import Image
import pytesseract

# Gemini (Google GenAI) client
# docs: https://ai.google.dev/gemini-api/docs/quickstart
import google.generativeai as genai

# fuzzy matching for phrase deduping
from fuzzywuzzy import fuzz

# -------- CONFIG (edit if necessary) ----------
GEMINI_MODEL = "gemini-2.5-flash"
MAX_SLIDES_PER_GEMINI_REQUEST = 10  # chunk slides to keep prompt sizes reasonable
# -----------------------------------------------

# If you're on Windows and not in PATH, set:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------- helpers ----------
NUM_RE = re.compile(r'(?<!\w)(?:\$|€|£)?\s*([+-]?\d{1,3}(?:[,\d]{0,})?(?:\.\d+)?)(\s*%|\s*(?:million|bn|billion|k|M|B))?', re.IGNORECASE)
PERCENT_RE = re.compile(r'([+-]?\d+(?:\.\d+)?)\s*%')
DATE_TOKEN_RE = re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|\d{1,2})[^\n,./-]{0,20}\d{2,4}\b', re.IGNORECASE)

def text_for_shape(shape):
    text = ""
    try:
        if shape.has_text_frame:
            text = "\n".join([p.text.strip() for p in shape.text_frame.paragraphs if p.text and p.text.strip()])
    except Exception:
        text = ""
    return text

def extract_numbers_from_text(text):
    nums = []
    for m in NUM_RE.finditer(text):
        raw = m.group(0).strip()
        val = m.group(1).replace(",", "")
        unit = (m.group(2) or "").strip().lower()
        try:
            num = float(val)
            # apply heuristics for units
            if 'k' in unit and num < 1e6:
                num = num * 1e3
            if 'm' in unit and not '%' in unit:
                num = num * 1e6
            if 'b' in unit:
                num = num * 1e9
        except:
            num = None
        nums.append({"raw": raw, "value": num, "unit": unit})
    return nums

def extract_dates_from_text(text):
    dates = []
    # naive: find date-like tokens, parse them
    for m in DATE_TOKEN_RE.finditer(text):
        token = m.group(0)
        try:
            dt = dateparser.parse(token, default=datetime(2000,1,1))
            dates.append({"raw": token, "parsed": dt.isoformat()})
        except Exception:
            continue
    # also try ISO-like dates
    iso_dates = re.findall(r'\b\d{4}-\d{2}-\d{2}\b', text)
    for d in iso_dates:
        try:
            dt = dateparser.parse(d)
            dates.append({"raw": d, "parsed": dt.isoformat()})
        except:
            pass
    return dates

# ---------- PPTX parsing ----------
def parse_pptx(path):
    prs = Presentation(path)
    slides = []
    tmpdir = tempfile.mkdtemp(prefix="pptx_imgs_")
    for i, slide in enumerate(prs.slides, start=1):
        stext = []
        tables = []
        saved_images = []
        for shape in slide.shapes:
            # text
            t = text_for_shape(shape)
            if t:
                stext.append(t)
            # tables
            try:
                if shape.has_table:
                    rows = []
                    for r in shape.table.rows:
                        rows.append([c.text.strip() for c in r.cells])
                    tables.append(rows)
            except Exception:
                pass
            # images
            try:
                if shape.shape_type == 13 or hasattr(shape, "image"):  # picture
                    img = shape.image
                    blob = img.blob
                    ext = img.ext
                    fname = os.path.join(tmpdir, f"slide{ i }_img_{len(saved_images)}.{ext}")
                    with open(fname, "wb") as f:
                        f.write(blob)
                    saved_images.append(fname)
            except Exception:
                pass
        slides.append({
            "slide_index": i,
            "text": "\n".join(stext),
            "tables": tables,
            "images": saved_images
        })
    return slides

# ---------- OCR ----------
def ocr_image(path):
    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        return ""

# ---------- Heuristic detectors ----------
def detect_numeric_conflicts(slides):
    """
    Robust numeric mismatch detector.
    Works from slide['text'] (not slide['texts']) and splits text into segments.
    """
    mentions = []
    window = 40
    for s in slides:
        txt = s.get("text", "") or ""
        # split into short segments (lines/paragraphs) to create context
        segments = [seg.strip() for seg in re.split(r'[\r\n]+', txt) if seg.strip()]
        if not segments:
            continue
        for seg in segments:
            for m in NUM_RE.finditer(seg):
                start, end = m.span()
                left = seg[max(0, start - window):start].strip()
                right = seg[end:end + window].strip()
                snippet = (left + " " + m.group(0) + " " + right).strip()
                # robust key: join last 6 left words + first 6 right words, remove numbers
                left_words = left.split()[-6:] if left else []
                right_words = right.split()[:6] if right else []
                key = re.sub(r'\d+[\d,\.]*', '', ' '.join(left_words + right_words)).strip()
                try:
                    val = float(m.group(1).replace(",", ""))
                except Exception:
                    val = None
                mentions.append({
                    "slide": s.get("slide_index", s.get("slide", None)),
                    "snippet": snippet,
                    "value": val,
                    "raw": m.group(0),
                    "key": key
                })

    # grouping using fuzzy matching on snippet/key
    groups = []
    used = [False] * len(mentions)
    for i, a in enumerate(mentions):
        if used[i]:
            continue
        grp = [a]
        used[i] = True
        for j in range(i + 1, len(mentions)):
            if used[j]:
                continue
            # compare keys/snippets -- try key first (if present) then fallback to snippet
            score = 0
            if a.get("key") and mentions[j].get("key"):
                score = fuzz.partial_ratio(a["key"][:80], mentions[j]["key"][:80])
            else:
                score = fuzz.partial_ratio(a["snippet"][:80], mentions[j]["snippet"][:80])
            if score > 60:
                grp.append(mentions[j])
                used[j] = True
        if len(grp) > 1:
            vals = [g["value"] for g in grp if g["value"] is not None]
            if len(set(vals)) > 1:
                slides_ref = sorted(list({x["slide"] for x in grp if x["slide"] is not None}))
                groups.append({"group": grp, "slides": slides_ref})
    issues = []
    for g in groups:
        issues.append({
            "type": "numeric_conflict",
            "slides": g["slides"],
            "evidence": g["group"]
        })
    return issues




def detect_percentage_sum_problems(slides):
    issues = []
    for s in slides:
        txt = s["text"]
        # heuristic: if slide has "Breakdown", "split", "composition" keywords, expect percentages to sum to ~100
        if re.search(r'\b(breakdown|split|composition|distribution|share|mix|percentages?)\b', txt, re.IGNORECASE):
            percents = [float(m.group(1)) for m in PERCENT_RE.finditer(txt)]
            if percents and abs(sum(percents) - 100.0) > 3.0:
                issues.append({"type": "percent_sum_mismatch", "slide": s["slide_index"], "sum": sum(percents), "percents": percents, "text_sample": txt[:300]})
    return issues

def detect_timeline_mismatches(slides):
    # naive: collect dates per slide; if a later slide lists a baseline that is after a forecast date, flag
    slide_dates = {}
    for s in slides:
        dts = extract_dates_from_text(s["text"])
        slide_dates[s["slide_index"]] = dts
    issues = []
    # pairwise checks for obvious contradictions: e.g., slide A says "by 2028 revenue will be X", slide B says "2028 baseline revenue was Y"
    for i in range(len(slides)):
        for j in range(i+1, len(slides)):
            di = slide_dates.get(i+1, [])
            dj = slide_dates.get(j+1, [])
            # pick first parsed date if any
            if di and dj:
                try:
                    di0 = dateparser.parse(di[0]["parsed"])
                    dj0 = dateparser.parse(dj[0]["parsed"])
                    # if slide earlier has a date after later slide's date, weird timeline
                    if di0 > dj0 and (j+1) > (i+1):
                        issues.append({"type": "timeline_mismatch",
                                       "slides": [i+1, j+1],
                                       "dates": [di[0], dj[0]],
                                       "note": "earlier slide contains a later date than a later slide"})
                except Exception:
                    pass
    return issues

# ---------- Gemini-based contradiction detection ----------
def init_gemini_client():
    """
    Configure genai and return a GenerativeModel instance.
    Tries multiple env var names for the key.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY / GENAI_API_KEY) environment variable not set.")

    # configure SDK
    genai.configure(api_key=api_key)

    # create and return the model object using the global GEMINI_MODEL name
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        # If strict creation fails, raise a helpful error
        raise RuntimeError(f"Failed to instantiate GenerativeModel('{GEMINI_MODEL}'): {e}")
    return model

def _extract_genai_text(response):
    """
    Helper to extract human text from various response shapes.
    """
    # common: response.text
    if hasattr(response, "text") and response.text:
        return response.text
    # sometimes SDK returns response.output[0].content[0].text
    try:
        out = getattr(response, "output", None)
        if out:
            # output may be list-like
            if isinstance(out, (list, tuple)) and len(out) > 0:
                c0 = out[0]
                # typical nested structure
                try:
                    return c0["content"][0]["text"]
                except Exception:
                    # try attribute access
                    try:
                        return c0.content[0].text
                    except Exception:
                        pass
    except Exception:
        pass
    # candidates variant
    if hasattr(response, "candidates"):
        try:
            cand = response.candidates[0]
            if hasattr(cand, "content"):
                return cand.content[0].text if cand.content else str(cand)
        except Exception:
            pass
    # fallback
    return str(response)

def ask_gemini_for_contradictions(model, slides_chunk):
    """
    Use the provided `model` (GenerativeModel) to ask Gemini to compare slides and
    return structured JSON. Returns list or fallback item on error.
    """
    # build prompt
    prompt_parts = []
    for s in slides_chunk:
        safe_text = s.get("text", "") or ""
        prompt_parts.append(f"Slide {s['slide_index']} text:\n\"\"\"\n{safe_text[:4000]}\n\"\"\"")
    prompt = (
        "You are an assistant that compares short slide texts and finds factual or logical inconsistencies across them. "
        "List each inconsistency found, classify it as one of: numeric_conflict, percent_mismatch, timeline_mismatch, contradictory_claim, other. "
        "For each issue include: type, slide numbers involved, short explanation, and an exact excerpt from each slide that supports the claim.\n\n"
        "Slides:\n\n" + "\n\n".join(prompt_parts) +
        "\n\nOutput: produce a JSON array where each entry has fields: type, slides, explanation, excerpts (map slide->excerpt). If none found, return an empty array []\n"
    )

    # call Gemini (try a couple of invocation styles for compatibility)
    try:
        try:
            response = model.generate_content(prompt)  # positional form used in many examples
        except TypeError:
            # named arg form
            response = model.generate_content(content=prompt)
    except Exception as e:
        # try alternate fallback via genai (some SDK builds expose helper methods)
        try:
            response = genai.generate_text(model=GEMINI_MODEL, input=prompt)
        except Exception as e2:
            # return LLM error as structured single-item list (so caller appends llm_error)
            return [{"type": "llm_error", "error": f"genai call failed: {e} / fallback error: {e2}", "slides": [s["slide_index"] for s in slides_chunk]}]

    # extract textual output
    text = _extract_genai_text(response)

    # try to find a JSON array/object inside the reply
    json_part = None
    m = re.search(r'(\[.*\]|\{.*\})', text, re.S)
    if m:
        candidate = m.group(1)
        try:
            json_part = json.loads(candidate)
        except Exception:
            # try permissive fixes
            try:
                t = candidate.replace("'", '"')
                t = re.sub(r',\s*}', '}', t)
                t = re.sub(r',\s*\]', ']', t)
                json_part = json.loads(t)
            except Exception:
                json_part = None

    if json_part is None:
        # return LLM raw output to let the caller decide (keeps pipeline stable)
        return [{"type": "llm_raw_output", "slides": [s["slide_index"] for s in slides_chunk], "explanation": text}]
    return json_part
# ---------- main orchestration ----------
def analyze_deck(pptx_path, extra_images_dir=None):
    slides = parse_pptx(pptx_path)
    # OCR any images embedded
    for s in slides:
        for imgpath in s["images"]:
            ocr_text = ocr_image(imgpath)
            if ocr_text.strip():
                s["text"] += "\n\n[OCR IMAGE TEXT]\n" + ocr_text

    # OCR any extra images provided by user (treated as pseudo-slides)
    if extra_images_dir and os.path.isdir(extra_images_dir):
        for i, fname in enumerate(sorted(os.listdir(extra_images_dir)), start=len(slides)+1):
            path = os.path.join(extra_images_dir, fname)
            if os.path.isfile(path):
                ocr_text = ocr_image(path)
                slides.append({"slide_index": i, "text": f"[external image {fname}]\n{ocr_text}", "tables": [], "images": [path]})

    report = {"analyzed_at": datetime.utcnow().isoformat() + "Z", "source": pptx_path, "issues": []}

    # run deterministic heuristics
    report["issues"].extend(detect_numeric_conflicts(slides))
    report["issues"].extend(detect_percentage_sum_problems(slides))
    report["issues"].extend(detect_timeline_mismatches(slides))

    # LLM comparisons in chunks
    client = init_gemini_client()
    n = len(slides)
    for i in range(0, n, MAX_SLIDES_PER_GEMINI_REQUEST):
        chunk = slides[i:i+MAX_SLIDES_PER_GEMINI_REQUEST]
        # make minimal dicts
        chunk_mini = [{"slide_index": s["slide_index"], "text": s["text"]} for s in chunk]
        try:
            lres = ask_gemini_for_contradictions(client, chunk_mini)
            # normalize LLM output if necessary
            if isinstance(lres, list):
                for item in lres:
                    # simple normalization
                    if "slides" not in item and "slide" in item:
                        item["slides"] = item.get("slide")
                    report["issues"].append(item)
            else:
                report["issues"].append({"type": "llm_unexpected", "content": lres})
        except Exception as e:
            report["issues"].append({"type": "llm_error", "error": str(e), "slides": [s["slide_index"] for s in chunk]})

    return report

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Detect inconsistencies in a PPTX deck.")
    parser.add_argument("pptx", help="path to .pptx")
    parser.add_argument("--images", help="optional directory of slide images to OCR", default=None)
    parser.add_argument("--out", help="output JSON file", default="inconsistency_report.json")
    args = parser.parse_args()

    if not os.path.isfile(args.pptx):
        print("pptx not found:", args.pptx)
        sys.exit(2)

    print("Parsing PPTX and running heuristics + Gemini ...")
    report = analyze_deck(args.pptx, args.images)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("Done. Report written to", args.out)
    # pretty summary
    print("\nSummary of issues found:")
    for idx, issue in enumerate(report["issues"], start=1):
        itype = issue.get("type")
        slides = issue.get("slides") or issue.get("slide") or issue.get("slide_index")
        explanation = (issue.get("explanation") or issue.get("note") or "")[:200]
        print(f"{idx}. {itype} — slides {slides} — {explanation}")

if __name__ == "__main__":
    main()
