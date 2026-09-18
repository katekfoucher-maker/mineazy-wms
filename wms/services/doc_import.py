"""Parse an uploaded Stock-Movement / delivery document (PDF, CSV or Excel) into
the fields the New Back Order form needs.

Returns::

    {
      "stock_movement_id": str | None,
      "branch": str | None,          # branch code or name, best-effort
      "doc_date": datetime.date | None,
      "lines": [{"sku", "description", "requested_qty", "sent_qty"}],
      "warnings": [str, ...],
    }

An empty / blank "sent" cell means nothing was dispatched (0).
"""
from __future__ import annotations

import csv
import io
import re
from datetime import date, datetime
from typing import Optional

import pandas as pd

_ITEM_KEYS = ("item no", "item_no", "itemno", "sku", "code", "product", "part no")
_REQ_KEYS = ("req. qty", "req qty", "req.qty", "requested", "requested_qty",
             "request qty", "qty req", "order qty", "req")
_SENT_KEYS = ("sent qty", "sent_qty", "sent", "dispatched", "dispatch qty",
              "delivered", "supplied", "issued")
_DESC_KEYS = ("description", "name", "desc", "product name", "item description")
# a plain "how many of each" list (for the split-by-sales tool): much broader than
# a dispatch note's "Req. Qty" - any column that reads like a count of units
_QTY_KEYS = _REQ_KEYS + ("qty", "quantity", "available", "avail", "in stock",
                         "on hand", "units", "amount", "count", "to split",
                         "split qty", "total")

_MOVEMENT_LABEL = re.compile(r"stock\s*movement|movement\s*(no|id|#|number)|document\s*(no|id)|"
                             r"transfer\s*(no|id)", re.I)
_BRANCH_LABEL = re.compile(r"to\s*location|receiving|destination|branch|to\s*branch", re.I)
_DATE_LABEL = re.compile(r"\bdate\b", re.I)
_DATE_RX = re.compile(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(\d{4}-\d{2}-\d{2})")
_SKU_RX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-]{2,}$")


def _norm(v) -> str:
    return re.sub(r"\s+", " ", str(v)).strip().lower()


def _to_int(v) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip().replace(",", "")
    if s == "" or s.lower() in ("nan", "none", "-"):
        return None
    try:
        return int(round(float(s)))
    except ValueError:
        return None


def _parse_date(s: str) -> Optional[date]:
    m = _DATE_RX.search(str(s))
    if not m:
        return None
    tok = m.group(0)
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(tok, fmt).date()
        except ValueError:
            continue
    return None


# a plain commercial invoice (no Item No / SKU column - just a description and a
# quantity, e.g. "1.  Model : 1.1KW-6 polo") has nothing to key the product on,
# so one is manufactured from the description itself
_LEADING_NUM_LABEL_RX = re.compile(r"^\s*\d+\s*[\.\)]\s*model\s*:?\s*", re.I)
_LEADING_LABEL_RX = re.compile(r"^\s*model\s*:?\s*", re.I)


def _derive_sku_and_clean_desc(raw: str) -> tuple[str, str]:
    """('1.  Model : 1.1KW-6 polo', ...) -> ('1.1KW-6POLO', '1.1KW-6 polo') - strip the
    listing-number/"Model:" boilerplate a line like this carries, then compact what's
    left into a SKU (a real code, if the description already has one, passes through
    unchanged)."""
    s = re.sub(r"\s+", " ", str(raw or "")).strip()
    s = _LEADING_NUM_LABEL_RX.sub("", s)
    s = _LEADING_LABEL_RX.sub("", s)
    s = s.strip(" :.-")
    clean_desc = s or str(raw or "").strip()
    sku = re.sub(r"\s+", "", clean_desc).upper()
    return sku or "ITEM", clean_desc


# ======================================================================
# entry point
# ======================================================================
_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
_IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"BM", b"II*\x00", b"MM\x00*")


def parse_document(raw: bytes, filename: str) -> dict:
    name = (filename or "").lower()
    if name.endswith(".pdf") or raw[:5] == b"%PDF-":
        return _parse_pdf(raw)
    if name.endswith(_IMAGE_EXT) or any(raw[:len(m)] == m for m in _IMAGE_MAGIC):
        return _parse_image(raw)
    if name.endswith((".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls")) or not name:
        return _parse_table(raw, filename)
    raise ValueError("Unsupported file type - upload a photo/scan, PDF, CSV or Excel "
                     "export of the document.")


def parse_qty_list(raw: bytes, filename: str) -> dict:
    """Parse a simple 'product + quantity' list for the split-by-sales tool.

    Returns ``{"lines": [{"sku", "description", "qty"}], "warnings": [...]}``.
    Accepts CSV / TSV / Excel with a header naming an item column and any
    count-like column (Qty, Quantity, Available, Units ...); if no header is
    recognised it falls back to "first column = SKU, first numeric column = qty".
    PDFs are read with the dispatch-note parser (its Req. Qty becomes the qty).
    """
    name = (filename or "").lower()
    if name.endswith(".pdf") or raw[:5] == b"%PDF-":
        p = _parse_pdf(raw)
        return {"lines": [{"sku": l["sku"], "description": l.get("description", ""),
                           "qty": l["requested_qty"]} for l in p["lines"]],
                "warnings": p.get("warnings", [])}

    rows = _rows_from_bytes(raw, filename)
    if not rows:
        raise ValueError("The file is empty.")

    sku_c = qty_c = desc_c = header_i = None
    for i, row in enumerate(rows[:40]):
        cells = [_norm(c) for c in row]
        sc = next((j for j, c in enumerate(cells) if any(k in c for k in _ITEM_KEYS)), None)
        qc = next((j for j, c in enumerate(cells)
                   if any(k in c for k in _QTY_KEYS) and j != sc), None)
        if sc is not None and qc is not None:
            sku_c, qty_c, header_i = sc, qc, i
            desc_c = next((j for j, c in enumerate(cells)
                           if any(k in c for k in _DESC_KEYS) and j not in (sc, qc)), None)
            break

    warnings: list[str] = []
    lines: list[dict] = []
    if header_i is None:                       # headerless: col 0 = sku, first numeric col = qty
        warnings.append("No header row recognised; read column 1 as product and the "
                        "first number column as quantity.")
        data = rows
        for row in data:
            if not row:
                continue
            sku = str(row[0]).strip()
            if not sku or sku.lower() in ("nan", "none"):
                continue
            qv = next((_to_int(c) for c in row[1:] if _to_int(c) is not None), None)
            if qv is None or qv <= 0:
                continue
            lines.append({"sku": sku, "description": "", "qty": qv})
    else:
        for row in rows[header_i + 1:]:
            if sku_c >= len(row):
                continue
            sku = str(row[sku_c]).strip()
            if not sku or sku.lower() in ("nan", "none"):
                continue
            qv = _to_int(row[qty_c]) if qty_c < len(row) else None
            if qv is None or qv <= 0:
                continue
            desc = (str(row[desc_c]).strip()
                    if desc_c is not None and desc_c < len(row) else "")
            lines.append({"sku": sku, "description": desc, "qty": qv})

    if not lines:
        raise ValueError("No product / quantity rows found in the file.")
    return {"lines": lines, "warnings": warnings}


# ======================================================================
# CSV / Excel
# ======================================================================
def _rows_from_bytes(raw: bytes, filename: str) -> list[list[str]]:
    name = (filename or "").lower()
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        xl = pd.ExcelFile(io.BytesIO(raw))
        df = xl.parse(xl.sheet_names[0], header=None, dtype=str)
        return df.fillna("").astype(str).values.tolist()
    text = raw.decode("utf-8-sig", errors="replace")
    sniff_delim = "\t" if text.count("\t") > text.count(",") else ","
    return [list(r) for r in csv.reader(io.StringIO(text), delimiter=sniff_delim)]


def _find_header(rows: list[list[str]]) -> tuple[int, dict[str, Optional[int]]]:
    for i, row in enumerate(rows[:40]):
        cells = [_norm(c) for c in row]
        sku_c = next((j for j, c in enumerate(cells) if any(k in c for k in _ITEM_KEYS)), None)
        req_c = next((j for j, c in enumerate(cells) if any(k in c for k in _REQ_KEYS)), None)
        if sku_c is not None and req_c is not None:
            sent_c = next((j for j, c in enumerate(cells) if any(k in c for k in _SENT_KEYS)), None)
            desc_c = next((j for j, c in enumerate(cells) if any(k in c for k in _DESC_KEYS)), None)
            return i, {"sku": sku_c, "requested": req_c, "sent": sent_c, "desc": desc_c}
    # no Item No / SKU column - a plain commercial invoice ("Description" +
    # "QTY", no product code) still has a description and a quantity; derive
    # the SKU from the description instead of giving up
    for i, row in enumerate(rows[:40]):
        cells = [_norm(c) for c in row]
        desc_c = next((j for j, c in enumerate(cells) if any(k in c for k in _DESC_KEYS)), None)
        qty_c = next((j for j, c in enumerate(cells)
                     if any(k in c for k in _QTY_KEYS) and j != desc_c), None)
        if desc_c is not None and qty_c is not None:
            return i, {"sku": None, "requested": qty_c, "sent": None, "desc": desc_c}
    raise ValueError("Could not find a header row with a product column ('Item No' or "
                     "'Description') and a quantity column ('Req. Qty' or 'QTY').")


def _sniff_meta(rows: list[list[str]], header_i: int) -> dict:
    mv = branch = doc_date = None
    scan = rows[:header_i] + rows[header_i + 1: header_i + 4]
    for row in scan:
        cells = [str(c).strip() for c in row]
        joined = " | ".join(cells)
        for j, c in enumerate(cells):
            lc = _norm(c)
            if mv is None and _MOVEMENT_LABEL.search(lc):
                after = c.split(":")[-1].strip() if ":" in c else ""
                m = re.search(r"[A-Za-z0-9\-/]{4,}", after)
                if m:
                    mv = m.group(0)
                else:
                    mv = next((n.strip() for n in cells[j + 1:] if n.strip()), None)
            if branch is None and _BRANCH_LABEL.search(lc):
                after = c.split(":")[-1].strip() if ":" in c else ""
                branch = after or next((n.strip() for n in cells[j + 1:] if n.strip()), None)
            if doc_date is None and _DATE_LABEL.search(lc):
                doc_date = _parse_date(joined)
        if doc_date is None:
            doc_date = _parse_date(joined)
    if mv is None:
        for row in rows[:header_i]:
            for c in row:
                s = str(c).strip()
                if s.isdigit() and len(s) >= 6:
                    mv = s
                    break
            if mv:
                break
    return {"stock_movement_id": mv, "branch": branch, "doc_date": doc_date}


def _parse_table(raw: bytes, filename: str) -> dict:
    rows = _rows_from_bytes(raw, filename)
    if not rows:
        raise ValueError("The file is empty.")
    header_i, cols = _find_header(rows)
    meta = _sniff_meta(rows, header_i)
    lines, warnings = [], []
    derive_sku = cols["sku"] is None       # no Item No column - see _find_header
    for row in rows[header_i + 1:]:
        desc = (str(row[cols["desc"]]).strip()
                if cols["desc"] is not None and cols["desc"] < len(row) else "")
        if derive_sku:
            if not desc:
                continue
            sku, desc = _derive_sku_and_clean_desc(desc)
        else:
            if cols["sku"] >= len(row):
                continue
            sku = str(row[cols["sku"]]).strip()
            if not sku or sku.lower() in ("nan", "none"):
                continue
        if cols["requested"] >= len(row):
            req = None
        elif derive_sku:
            # a plain invoice's qty cell often carries a unit ("120sets", "10set")
            req = _to_int(re.sub(r"[^\d.,]", "", str(row[cols["requested"]])))
        else:
            req = _to_int(row[cols["requested"]])
        if req is None or req <= 0:
            continue
        sent = (_to_int(row[cols["sent"]])
                if (cols["sent"] is not None and cols["sent"] < len(row)) else None)
        sent = req if derive_sku and sent is None else (sent or 0)
        if sent > req:
            warnings.append(f"{sku}: dispatched {sent} > requested {req}; capped.")
            sent = req
        lines.append({"sku": sku, "description": desc, "requested_qty": req, "sent_qty": sent,
                      "sku_derived": derive_sku})
    if not lines:
        raise ValueError("No item rows found under the header.")
    return {**meta, "lines": lines, "warnings": warnings}


# ======================================================================
# shared word-position reconstruction - PDF text and OCR'd image words both
# arrive as a flat list of {"text","x0","x1","top","bottom"} and go through
# the same visual-row grouping and column detection from here on
# ======================================================================
def _words_to_rows(words: list[dict]) -> list[list[dict]]:
    """Group words into visual lines by vertical position. The tolerance scales
    with the words' own height so this works whether ``top``/``bottom`` are PDF
    points (a page ~800 tall) or image pixels (a photo can be 4000+ tall)."""
    if not words:
        return []
    heights = sorted(w["bottom"] - w["top"] for w in words if w["bottom"] > w["top"])
    med_h = heights[len(heights) // 2] if heights else 10.0
    tol = max(3.0, med_h * 0.6)
    rows_by_y: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if rows_by_y and abs(w["top"] - rows_by_y[-1][0]["top"]) <= tol:
            rows_by_y[-1].append(w)
        else:
            rows_by_y.append([w])
    return rows_by_y


def _generic_header(rows_by_y: list[list[dict]]) -> tuple[Optional[int], Optional[dict]]:
    """A plain commercial invoice's header: a Description column and a Qty-ish
    column, nothing else required - there is no Item No / SKU column to key on.
    Scans a several-line window since a header cell often wraps ("QTY" on one
    line, "(set)" on the next) and a photographed header can OCR onto more
    lines than a clean PDF's would."""
    n = len(rows_by_y)
    for i in range(min(n, 40)):
        window = [w for row in rows_by_y[i:i + 4] for w in row]
        joined = _norm(" ".join(w["text"] for w in window))
        if "descri" not in joined or not (("qty" in joined) or ("quantity" in joined)):
            continue
        anchors: dict = {}
        for w in window:
            nt = _norm(w["text"])
            cx = (w["x0"] + w["x1"]) / 2
            if "descri" in nt and "desc" not in anchors:
                anchors["desc"] = cx
            elif ("qty" in nt or "quantity" in nt) and "qty" not in anchors:
                anchors["qty"] = cx
            elif ("price" in nt or "usd" in nt or "amount" in nt) and "price" not in anchors:
                anchors["price"] = cx
        if "desc" in anchors and "qty" in anchors:
            return i, anchors
    return None, None


def _generic_row(line: list[dict], anchors: dict) -> Optional[dict]:
    keys = list(anchors.keys())
    buckets: dict = {k: [] for k in keys}
    for w in sorted(line, key=lambda w: w["x0"]):
        t = w["text"].strip()
        if not t:
            continue
        cx = (w["x0"] + w["x1"]) / 2
        nearest = min(keys, key=lambda k: abs(anchors[k] - cx))
        buckets[nearest].append(t)
    desc_raw = " ".join(buckets.get("desc", [])).strip()
    qty = _to_int(re.sub(r"[^\d.,]", "", " ".join(buckets.get("qty", []))))
    if not desc_raw or qty is None or qty <= 0:
        return None
    sku, desc = _derive_sku_and_clean_desc(desc_raw)
    return {"sku": sku, "description": desc, "requested_qty": qty, "sent_qty": qty,
            "sku_derived": True}


def _extract_lines_from_pages(pages: list[list[list[dict]]]) -> tuple[list[dict], dict, list[str]]:
    """``pages`` is one ``rows_by_y`` per page/image. Tries the dispatch-note
    layout (Item No / Req. Qty / Description / Sent Qty) first, carrying its
    column anchors across pages same as before; if that never matches on any
    page, falls back to the generic description+qty layout instead of failing -
    the plain commercial-invoice shape this exists for has no Item No column."""
    warnings: list[str] = []
    lines: list[dict] = []
    meta = {"stock_movement_id": None, "branch": None, "doc_date": None}
    anchors: Optional[dict] = None
    for pno, rows_by_y in enumerate(pages):
        if not rows_by_y:
            continue
        if pno == 0:
            meta.update(_pdf_meta(rows_by_y))
        hdr_idx, cols = _pdf_header(rows_by_y)
        if cols:
            anchors = cols
        if anchors is None:
            continue
        for line in rows_by_y[(hdr_idx + 1) if hdr_idx is not None else 0:]:
            rec = _pdf_row(line, anchors)
            if not rec:
                continue
            if rec["sent_qty"] is not None and rec["sent_qty"] > rec["requested_qty"]:
                warnings.append(f"{rec['sku']}: dispatched > requested; capped.")
                rec["sent_qty"] = rec["requested_qty"]
            rec["sent_qty"] = rec["sent_qty"] or 0
            lines.append(rec)

    if lines:
        return lines, meta, warnings

    # no Item No / Req. Qty table anywhere - try the plain description+qty
    # invoice shape instead (SKU manufactured from the description)
    anchors = None
    for pno, rows_by_y in enumerate(pages):
        if not rows_by_y:
            continue
        if pno == 0 and not any(meta.values()):
            meta.update(_pdf_meta(rows_by_y))
        hdr_idx, cols = _generic_header(rows_by_y)
        if cols:
            anchors = cols
        if anchors is None:
            continue
        for line in rows_by_y[(hdr_idx + 1) if hdr_idx is not None else 0:]:
            rec = _generic_row(line, anchors)
            if rec:
                lines.append(rec)
    return lines, meta, warnings


# ======================================================================
# PDF  (column reconstruction from word positions)
# ======================================================================
def _pdf_pages(raw: bytes) -> list[list[list[dict]]]:
    """One ``rows_by_y`` per PDF page - the same shape an OCR'd image
    contributes, so a PDF and a set of photos can be combined interchangeably
    in :func:`parse_documents`."""
    try:
        import pdfplumber
    except ImportError:
        raise ValueError("PDF support is not installed on the server "
                         "(pip install pdfplumber). Upload a CSV/Excel export instead.")

    pages: list[list[list[dict]]] = []
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=1.5, y_tolerance=3,
                                       keep_blank_chars=False, use_text_flow=False)
            pages.append(_words_to_rows(words))
    return pages


def _parse_pdf(raw: bytes) -> dict:
    pages = _pdf_pages(raw)
    lines, meta, warnings = _extract_lines_from_pages(pages)

    if not lines:
        raise ValueError("Could not read line items from the PDF. Try a CSV/Excel export "
                         "of the same document.")
    # de-dupe (repeated header rows / page furniture)
    seen, uniq = set(), []
    for ln in lines:
        key = (ln["sku"], ln["requested_qty"])
        if key not in seen:
            seen.add(key)
            uniq.append(ln)
    return {**meta, "lines": uniq, "warnings": warnings}


# ======================================================================
# Image (a photo or scan of a document) - OCR'd with Tesseract, then run
# through the exact same word-position reconstruction as a PDF
# ======================================================================
def _locate_tesseract():
    """Point pytesseract at the Tesseract binary if it isn't already on PATH -
    covers the common Windows installer location; Linux/Mac package managers
    (apt/brew) already put it on PATH, so this is a no-op there."""
    import shutil
    import pytesseract

    if shutil.which(pytesseract.pytesseract.tesseract_cmd or "tesseract"):
        return
    import sys
    if sys.platform == "win32":
        from pathlib import Path
        for candidate in (
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        ):
            if candidate.exists():
                pytesseract.pytesseract.tesseract_cmd = str(candidate)
                return


def _ocr_words(raw: bytes) -> list[dict]:
    """Run Tesseract over one image and return its words as position dicts -
    the same shape ``pdfplumber`` word extraction produces, so an OCR'd photo
    and a PDF page can be combined interchangeably."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        raise ValueError("Photo/scan support is not installed on the server "
                         "(pip install pytesseract pillow, and the Tesseract OCR "
                         "engine itself). Upload a PDF, CSV or Excel export instead.")

    _locate_tesseract()
    try:
        img = Image.open(io.BytesIO(raw))
        img = img.convert("RGB") if img.mode != "RGB" else img
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError:
        raise ValueError("The Tesseract OCR engine isn't installed on the server - "
                         "install it (e.g. 'apt install tesseract-ocr', or the "
                         "UB-Mannheim build on Windows), then try again.")

    words: list[dict] = []
    n = len(data.get("text", []))
    for i in range(n):
        t = (data["text"][i] or "").strip()
        if not t:
            continue
        try:
            if float(data["conf"][i]) < 0:            # -1 = not real text (layout box)
                continue
        except (TypeError, ValueError, KeyError):
            pass
        x, y = data["left"][i], data["top"][i]
        w, h = data["width"][i], data["height"][i]
        words.append({"text": t, "x0": float(x), "x1": float(x + w),
                      "top": float(y), "bottom": float(y + h)})
    return words


def _parse_image(raw: bytes) -> dict:
    words = _ocr_words(raw)
    if not words:
        raise ValueError("Could not read any text from the photo - try a clearer, "
                         "well-lit, straight-on shot, or a PDF/CSV/Excel export instead.")

    rows_by_y = _words_to_rows(words)
    lines, meta, warnings = _extract_lines_from_pages([rows_by_y])
    if not lines:
        preview = " / ".join(
            " ".join(w["text"] for w in sorted(row, key=lambda w: w["x0"]))
            for row in rows_by_y[:15])
        raise ValueError(
            "Read the photo, but couldn't find a Description + Qty (or Item No + "
            "Req. Qty) table in it. If this is one page of a multi-page document, "
            "upload all of its pages/photos together so a header found on one page "
            "can carry over to the others. Line items may need to be entered "
            f"manually otherwise. What was actually read: “{preview[:600]}”")
    seen, uniq = set(), []
    for ln in lines:
        key = (ln["sku"], ln["requested_qty"])
        if key not in seen:
            seen.add(key)
            uniq.append(ln)
    warnings.append("Read from a photo/scan (OCR) - double-check the quantities "
                    "and product names below before confirming.")
    return {**meta, "lines": uniq, "warnings": warnings}


# ======================================================================
# Multiple uploaded files treated as one document - e.g. several photos of
# a multi-page invoice where only the first page repeats the column
# headers. Pages from every file are combined into ONE list before running
# the normal header/column detection, so an anchor found on page 1 carries
# over to headerless continuation pages.
# ======================================================================
def parse_documents(files: list[tuple[bytes, str]]) -> dict:
    files = [f for f in files if f[0]]
    if not files:
        raise ValueError("No file was uploaded.")
    if len(files) == 1:
        raw, filename = files[0]
        return parse_document(raw, filename)

    pages: list[list[list[dict]]] = []
    ocr_used = False
    table_files: list[tuple[bytes, str]] = []
    for raw, filename in files:
        name = (filename or "").lower()
        if name.endswith(".pdf") or raw[:5] == b"%PDF-":
            pages.extend(_pdf_pages(raw))
        elif name.endswith(_IMAGE_EXT) or any(raw[:len(m)] == m for m in _IMAGE_MAGIC):
            words = _ocr_words(raw)
            if not words:
                raise ValueError(f"Could not read any text from '{filename}' - try a "
                                 "clearer, well-lit, straight-on shot.")
            pages.append(_words_to_rows(words))
            ocr_used = True
        else:
            # a spreadsheet/CSV doesn't fit the "consecutive pages of one
            # table" model used here - parse it on its own and merge its lines
            table_files.append((raw, filename))

    lines, meta, warnings = ([], {}, [])
    if pages:
        lines, meta, warnings = _extract_lines_from_pages(pages)
        if not lines:
            raise ValueError(
                "Read all the pages/photos, but couldn't find a Description + Qty "
                "(or Item No + Req. Qty) table across them. Line items may need to "
                "be entered manually.")

    for raw, filename in table_files:
        try:
            extra = _parse_table(raw, filename)
        except ValueError as e:
            warnings.append(f"'{filename}': {e}")
            continue
        lines.extend(extra["lines"])
        warnings.extend(extra.get("warnings", []))
        meta = meta or extra

    if not lines:
        raise ValueError("Could not read line items from the uploaded files.")

    seen, uniq = set(), []
    for ln in lines:
        key = (ln["sku"], ln["requested_qty"])
        if key not in seen:
            seen.add(key)
            uniq.append(ln)
    if ocr_used:
        warnings.append("Read from a photo/scan (OCR) - double-check the quantities "
                        "and product names below before confirming.")
    warnings.append(f"Combined {len(files)} uploaded files/pages into one document.")
    return {**meta, "lines": uniq, "warnings": warnings}


def _pdf_header(rows_by_y: list[list[dict]]) -> tuple[Optional[int], Optional[dict]]:
    for i, line in enumerate(rows_by_y):
        texts = {_norm(w["text"]): w for w in line}
        joined = " ".join(texts)
        if "req." in joined and "description" in joined and ("sent" in joined or "qty" in joined):
            def cx(label):
                w = next((w for t, w in texts.items() if t.startswith(label)), None)
                return (w["x0"] + w["x1"]) / 2 if w else None
            item_x = cx("item")
            req_x = cx("req")
            desc_x = cx("desc")
            sent_x = cx("sent")
            rec_x = cx("rec")
            if item_x is None or req_x is None or desc_x is None:
                continue
            # boundaries = midpoints between successive anchors
            b_item_req = (item_x + req_x) / 2
            b_req_desc = (req_x + desc_x) / 2
            b_desc_sent = (desc_x + (sent_x or desc_x + 200)) / 2
            b_sent_rec = ((sent_x or 1e6) + (rec_x or 1e9)) / 2 if sent_x else 1e9
            return i, {"item_req": b_item_req, "req_desc": b_req_desc,
                       "desc_sent": b_desc_sent, "sent_rec": b_sent_rec}
    return None, None


def _pdf_row(line: list[dict], a: dict) -> Optional[dict]:
    item, req_toks, desc_toks, sent_toks = [], [], [], []
    for w in sorted(line, key=lambda w: w["x0"]):
        c = (w["x0"] + w["x1"]) / 2
        t = w["text"].strip()
        if not t:
            continue
        if c < a["item_req"]:
            item.append(t)
        elif c < a["req_desc"]:
            req_toks.append(t)
        elif c < a["desc_sent"]:
            desc_toks.append(t)
        elif c < a["sent_rec"]:
            sent_toks.append(t)
    sku = "".join(item).strip()
    if not sku or not _SKU_RX.match(sku):
        return None
    req = _to_int(" ".join(req_toks).replace(" ", ""))
    if req is None or req <= 0:
        return None
    sent = _to_int(" ".join(sent_toks).replace(" ", ""))
    return {"sku": sku, "description": " ".join(desc_toks).strip(),
            "requested_qty": req, "sent_qty": sent, "sku_derived": False}


def _pdf_meta(rows_by_y: list[list[dict]]) -> dict:
    mv = branch = doc_date = None
    flat = [w["text"].strip() for line in rows_by_y[:25] for w in line]
    joined = " ".join(flat)
    if not doc_date:
        doc_date = _parse_date(joined)
    for tok in flat:
        if mv is None and tok.isdigit() and len(tok) >= 6:
            mv = tok
    # branch: the "To Location" code/name (same line after the label, or the line below)
    _LABEL_WORDS = {"to", "from", "location", "recieving", "receiving", "sending",
                    "address", "name", "contact", "branch", "destination", "fax", "telephone"}

    def _clean(cand: str) -> Optional[str]:
        cand = cand.strip()
        if not cand or cand.upper() in ("DISTRIBUTION CENTER", "STOCK MOVEMENT"):
            return None
        if cand.lower().startswith(("contact", "telephone", "fax", "name", "comment")):
            return None
        return cand

    for li, line in enumerate(rows_by_y[:25]):
        toks = [w["text"] for w in sorted(line, key=lambda w: w["x0"])]
        lt = _norm(" ".join(toks))
        if not _BRANCH_LABEL.search(lt):
            continue
        rest = [t for t in toks if t.lower().strip(":") not in _LABEL_WORDS]
        branch = _clean(" ".join(rest))
        if not branch:
            for nxt in rows_by_y[li + 1: li + 3]:
                branch = _clean(" ".join(w["text"] for w in nxt))
                if branch:
                    break
        if branch:
            break
    return {"stock_movement_id": mv, "branch": branch, "doc_date": doc_date}
