"""weekly_dispatch_pdf: a branch's order document, one page per as-needed
page break. Regression coverage for a bug where the bounding box drawn
around each page's rows was computed from y-coordinates spanning across a
page break (the box's start point from the old page, its end point from the
new one), producing a bogus - often negative - height. FPDF drew that box
in the wrong place on the page, visually overlapping/cutting through rows
near the break (see the report: a product appeared to print twice around
the page-break boundary for Esigodini's weekly order)."""
import pathlib

import pdfplumber

from wms.exports import pdf as pdf_export


def test_weekly_dispatch_pdf_paginates_without_duplicating_or_corrupting_rows(tmp_path):
    lines = [{"sku": f"SKU{i:04d}", "description": f"Test Product Number {i}",
             "predicted": 10, "on_hand": 0, "requested": 0, "sent": 0, "rec": i}
             for i in range(1, 81)]                  # enough to force several pages
    batch = {"by_branch": [{"code": "ES", "branch": "Esigodini", "doc_id": None,
                           "date": "2026-01-01", "lines": lines,
                           "rec_total": sum(l["rec"] for l in lines)}]}

    path = pdf_export.weekly_dispatch_pdf(batch, out_dir=tmp_path)
    assert pathlib.Path(path).exists()

    with pdfplumber.open(str(path)) as pdf:
        assert len(pdf.pages) > 1               # the whole point of this test
        all_text = ""
        for page in pdf.pages:
            all_text += (page.extract_text() or "") + "\n"
            # exactly one bounding box per page, with a sane (non-negative,
            # not implausibly huge) height - the bug produced a negative or
            # wildly wrong height by mixing y-coordinates from two pages
            rects = page.rects
            assert len(rects) == 1
            assert 0 < rects[0]["height"] < page.height

    skus = [f"SKU{i:04d}" for i in range(1, 81)]
    for sku in skus:
        assert all_text.count(sku) == 1, f"{sku} appears {all_text.count(sku)} times"
    # every page after the first repeats the column headers, so a reader
    # landing mid-document (or the row right after a break) isn't left
    # guessing what each column means
    assert all_text.count("Item No") == len(pdf.pages)
