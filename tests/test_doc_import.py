"""Document import: the dispatch-note-shaped parser (Item No / Req. Qty) is
unchanged; new coverage is the fallback for a plain commercial invoice with no
product-code column (description + qty only, e.g. a supplier's own invoice),
and OCR support for a photo/scan of one."""
import io

import pytest

from wms.models import Product
from wms.services import catalogue, doc_import


def test_derive_sku_strips_listing_number_and_model_label():
    sku, desc = doc_import._derive_sku_and_clean_desc("1.   Model : 1.1KW-6 polo")
    assert sku == "1.1KW-6POLO"
    assert desc == "1.1KW-6 polo"


def test_derive_sku_without_a_model_label_just_compacts():
    sku, desc = doc_import._derive_sku_and_clean_desc("Widget Deluxe")
    assert sku == "WIDGETDELUXE"
    assert desc == "Widget Deluxe"


def test_dispatch_note_csv_still_parses_via_the_item_no_column():
    """The existing, primary path (a real Item No / Req. Qty document) must be
    completely unaffected by the new no-SKU-column fallback."""
    csv = ("Stock Movement,,,SM-1\nTo Location,,Gwanda VID\nDate,,15/07/2026\n\n"
          "Item No,Req. Qty,Description,Sent Qty\n"
          "SFC1269,100,HELMET BLUE,40\n"
          "SFC1274,50,HELMET RED,\n")
    result = doc_import.parse_document(csv.encode(), "doc.csv")
    assert result["stock_movement_id"] == "SM-1"
    lines = {l["sku"]: l for l in result["lines"]}
    assert lines["SFC1269"] == {"sku": "SFC1269", "description": "HELMET BLUE",
                                "requested_qty": 100, "sent_qty": 40, "sku_derived": False}
    assert lines["SFC1274"]["sent_qty"] == 0


def test_commercial_invoice_csv_with_no_sku_column_derives_one():
    """The Hengyou-style layout: Description + QTY, no Item No column at all -
    previously this raised 'Could not find a header row with Item No...'."""
    csv = ("Description,QTY (set),EXW Price (USD),Notes\n"
          "1.   Model : 1.1KW-6 polo,120sets,55USD,120*55=6600\n"
          "1.   Model : 15KW-6polo,10set,332USD,10*332=3320\n")
    result = doc_import.parse_document(csv.encode(), "invoice.csv")
    lines = {l["sku"]: l for l in result["lines"]}
    assert lines["1.1KW-6POLO"] == {"sku": "1.1KW-6POLO", "description": "1.1KW-6 polo",
                                    "requested_qty": 120, "sent_qty": 120, "sku_derived": True}
    assert lines["15KW-6POLO"]["requested_qty"] == 10


def test_unsupported_file_type_still_rejected_clearly():
    with pytest.raises(ValueError, match="Unsupported file type"):
        doc_import.parse_document(b"whatever", "notes.docx")


def test_generic_invoice_lines_are_flagged_as_sku_derived():
    """The route layer uses this flag to know when it should try matching the
    description against the catalogue instead of trusting the SKU outright."""
    csv = "Description,QTY (set)\n1.   Model : Widget,10sets\n"
    result = doc_import.parse_document(csv.encode(), "invoice.csv")
    assert result["lines"][0]["sku_derived"] is True


def test_dispatch_note_lines_are_not_flagged_as_sku_derived():
    csv = ("Item No,Req. Qty,Description,Sent Qty\nSFC1269,100,HELMET BLUE,40\n")
    result = doc_import.parse_document(csv.encode(), "doc.csv")
    assert result["lines"][0]["sku_derived"] is False


def test_match_product_by_description_finds_an_exact_normalized_match():
    products = [Product(sku="MTR-1", name="1.1KW-6 polo"),
                Product(sku="MTR-2", name="15KW-6 polo")]
    match = catalogue.match_product_by_description("1.1KW-6 polo", products)
    assert match.sku == "MTR-1"


def test_match_product_by_description_no_confident_match_returns_none():
    products = [Product(sku="MTR-1", name="1.1KW-6 polo")]
    match = catalogue.match_product_by_description("Completely unrelated widget", products)
    assert match is None


def test_match_product_by_description_empty_input_returns_none():
    products = [Product(sku="MTR-1", name="1.1KW-6 polo")]
    assert catalogue.match_product_by_description("", products) is None


def _tesseract_available() -> bool:
    try:
        import pytesseract
        doc_import._locate_tesseract()
        pytesseract.get_tesseract_version()
        return True
    except Exception:                                     # noqa: BLE001
        return False


@pytest.mark.skipif(not _tesseract_available(), reason="Tesseract OCR not installed")
def test_photo_of_a_commercial_invoice_is_ocr_d_and_parsed():
    """A photographed invoice (no text layer, no Item No column) - the exact
    shape a real supplier invoice snapshot arrives in - goes through OCR and
    then the same no-SKU-column fallback as the CSV case above."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1400, 320), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 26)
        font_b = ImageFont.truetype("arialbd.ttf", 26)
    except Exception:                                       # noqa: BLE001
        font = font_b = ImageFont.load_default()

    d.text((300, 30), "Description", fill="black", font=font_b)
    d.text((750, 30), "QTY", fill="black", font=font_b)
    d.text((730, 65), "(set)", fill="black", font=font_b)
    d.text((950, 30), "EXW Price (USD)", fill="black", font=font_b)
    d.line((280, 110, 1380, 110), fill="black", width=2)
    d.text((300, 150), "1.   Model : 1.1KW-6 polo", fill="black", font=font)
    d.text((750, 150), "120sets", fill="black", font=font)
    d.text((950, 150), "55USD", fill="black", font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    result = doc_import.parse_document(buf.getvalue(), "invoice.png")
    assert any(w.startswith("Read from a photo/scan") for w in result["warnings"])
    lines = {l["sku"]: l for l in result["lines"]}
    assert "1.1KW-6POLO" in lines
    assert lines["1.1KW-6POLO"]["requested_qty"] == 120
    assert lines["1.1KW-6POLO"]["description"] == "1.1KW-6 polo"


def test_parse_documents_single_file_delegates_to_parse_document():
    csv = ("Description,QTY (set)\n1.   Model : Widget,10sets\n")
    result = doc_import.parse_documents([(csv.encode(), "invoice.csv")])
    assert result["lines"][0]["sku"] == "WIDGET"


def test_parse_documents_merges_multiple_table_files():
    """Two separate spreadsheet files uploaded together (not a photo/PDF
    multi-page case) still get their line items merged into one result."""
    csv1 = "Description,QTY (set)\n1.   Model : Widget A,10sets\n"
    csv2 = "Description,QTY (set)\n1.   Model : Widget B,5sets\n"
    result = doc_import.parse_documents([
        (csv1.encode(), "invoice1.csv"),
        (csv2.encode(), "invoice2.csv"),
    ])
    skus = {l["sku"] for l in result["lines"]}
    assert skus == {"WIDGETA", "WIDGETB"}


def _invoice_page_image(header: bool, rows: list[tuple[str, str]]):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1400, 320), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 26)
        font_b = ImageFont.truetype("arialbd.ttf", 26)
    except Exception:                                       # noqa: BLE001
        font = font_b = ImageFont.load_default()

    y = 30
    if header:
        d.text((300, y), "Description", fill="black", font=font_b)
        d.text((750, y), "QTY", fill="black", font=font_b)
        d.line((280, y + 80, 1380, y + 80), fill="black", width=2)
        y += 120
    for desc, qty in rows:
        d.text((300, y), desc, fill="black", font=font)
        d.text((750, y), qty, fill="black", font=font)
        y += 60

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.skipif(not _tesseract_available(), reason="Tesseract OCR not installed")
def test_multi_photo_upload_carries_header_from_page_one_to_headerless_page_two():
    """The real-world failure this exists to fix: a multi-page invoice
    photographed page by page, where only page 1 repeats the column headers -
    page 2 (and beyond) has no header at all and can't be parsed on its own,
    but combining all the photos in one upload lets page 1's header anchor
    carry over."""
    page1 = _invoice_page_image(True, [("1.   Model : 1.1KW-6 polo", "120sets")])
    page2 = _invoice_page_image(False, [("1.   Model : 15KW-6polo", "10set")])

    result = doc_import.parse_documents([
        (page1, "invoice_p1.png"),
        (page2, "invoice_p2.png"),
    ])
    lines = {l["sku"]: l for l in result["lines"]}
    assert lines["1.1KW-6POLO"]["requested_qty"] == 120
    assert lines["15KW-6POLO"]["requested_qty"] == 10
