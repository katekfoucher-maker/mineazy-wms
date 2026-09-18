"""Real sample document 26503244 (Stock Movement: DC -> Belmont Shop, 27/08/2026).

Used by the seed script to populate a realistic catalogue + one transfer document
so the backorder analytics have something to chew on out of the box.
"""
from __future__ import annotations

from datetime import date

DOC_26503244 = {
    "doc_no": "26503244",
    "doc_date": date(2026, 8, 27),
    "from_location": "DC",
    "branch": "Belmont Shop",
    "comment": "stock transfer",
    # (item_no, requested_qty, sent_qty_or_None, description)
    "lines": [
        ("ACGSP001", 700, None, "PANEL FENCE (CLAMP&BOLTS)"),
        ("ANH0025", 10, 10, "WATER PUMP WP-50"),
        ("ANH0026", 10, 10, "WATER PUMP WP-80"),
        ("BMX003", 100, 100, "LUBRIMAX MULTI PURPOSE GREASE MP500G"),
        ("BS1387", 60, 13, "PICK HEAD 2Kgs"),
        ("BSET0017", 150, None, "EAZY TOOLS COMBINATION SPANNER 17MM"),
        ("BSET0019", 150, None, "EAZY TOOLS COMBINATION SPANNER 19MM"),
        ("CFC1065", 1000, 1000, "ELECTRIC CABLE FLEX 1.5MM 3CORE /M"),
        ("CFC1066", 1000, None, "ELECTRIC CABLE FLEX 1.5MMX 4CORE /M"),
        ("CFC1067", 1000, 300, "ELECTRIC CABLE FLEX 2.5MM 3CORE /M"),
        ("CFC1068", 1000, None, "ELECTRIC CABLE FLEX 2.5MM 4CORE /M"),
        ("CFC1069", 1000, None, "ELECTRIC CABLE FLEX 4.MM 4CORE /M"),
        ("CFC6MM4C", 500, None, "ELECTRIC CABLE FLEX 6MM 4C"),
        ("CTBFWM10", 500, None, "FLAT WASHER M10"),
        ("CTBFWM12", 500, None, "FLAT WASHER M12"),
        ("CTBFWM14", 500, None, "FLAT WASHER M14"),
        ("CTBFWM16", 500, None, "FLAT WASHER M16"),
        ("CTBHT1250", 2000, None, "BOLTS HIGH TENSILE WITH NUT M12 X 50MM"),
        ("CTBHT1260", 2000, None, "BOLTS HIGH TENSILE WITH NUT M12 X 60MM"),
        ("CTBHTN16", 500, None, "HT NUTS M16"),
        ("CTBHTN20", 500, None, "HT NUTS M20"),
        ("CTBHTN36", 500, None, "HT NUTS M36"),
        ("CTBMS1240", 7000, None, "BOLTS MILD STEEL WITH NUT M12 X 40-60MM"),
        ("CTBN0010", 300, None, "BOLTS HT WITH NUT M20 X 200"),
        ("GBL1288", 200, 100, "HYSPIN 68 2LTR"),
        ("GLBL2691", 300, 120, "FG-X SUPER SERIES SAE40 5L"),
        ("GLBL2692", 300, 300, "FG-X SUPER SERIES 15W40 5L"),
        ("HQH1837", 60, None, "BOTTOM ROLLER 500MM"),
        ("HQH3T", 50, 5, "JAW CRUSHER SHAFT (150X250)"),
        ("KJ1245", 30, 20, "H/HEAD 6 BEATER"),
        ("JEL1511", 100, None, "ALTERNATOR RECTIFIER 20A"),
        ("KJ1262", 1000, None, "HAMMERMILL PIN 18MM 10-BEATER"),
        ("KJ1263", 1000, None, "HAMMERMILL PIN 18MM 6-BEATER"),
        ("KJ1264", 1000, None, "HAMMERMILL PIN 18MM 8-BEATER"),
        ("KJSN001", 1000, None, "P/BLOCK SN511 CHINESE LONG BASE"),
        ("LC0001", 120, 120, "MUTTON CLOTH FG 500G"),
        ("LC0002", 120, 120, "MUTTON CLOTH FG 1KG"),
        ("LSDPCFWZZ146", 100, 20, "GUMBOOTS SHOVA SZ 6"),
        ("LSDPCFWZZ150", 100, 50, "GUMBOOTS SHOVA SZ 7"),
        ("LSDPCFWZZ160", 100, 100, "GUMBOOTS SHOVA SZ 8"),
        ("LSDPCFWZZ170", 100, 100, "GUMBOOTS SHOVA SZ 9"),
        ("LSDPCFWZZ180", 100, 80, "GUMBOOTS SHOVA SZ 10"),
        ("LSDPCFWZZ190", 100, None, "GUMBOOTS SHOVA SZ 11"),
        ("MCHMMS3060", 100, 10, "CLARITY DEGREASER 2L"),
        ("MCHMMS3059", 120, 60, "CLARITY DEGREASER 500ML"),
        ("MF0067", 20, 5, "BLANTON WEDGES"),
        ("MF1450", 20, 5, "STAMPMILL SHOES RED"),
        ("MF1451", 20, 4, "STAMPMILL DIES"),
        ("MID1352", 500, 500, "MILL BALLS 60MM"),
        ("MID1353", 500, 200, "MILL BALLS 80MM"),
        ("MKMB1061", 30, None, "BUNGA SMALL"),
        ("MKMB1060", 30, 15, "BUNGA BIG"),
        ("MKMB1091", 100, 100, "CLAY POT"),
        ("MOP25C06RBLA00", 20, None, "POLY PIPE 25MM CLASS 6"),
        ("MOP32C06RBLA00", 20, 10, "POLY PIPE 32MM CLASS 6"),
        ("MOP40C06RBLA00", 20, 20, "POLY PIPE 40MM CLASS 6"),
        ("MOP50C06RBLA00", 20, 4, "POLY PIPE 50MM CLASS 6"),
        ("MOP63C06RBLA00", 10, None, "POLY PIPE 63MM CLASS 6"),
        ("WVLY32C06", 20, 20, "POLY PIPE 32MM CLASS6 BLUELINE"),
        ("WVLY40C06", 20, 20, "POLY PIPE 40MM CLASS 6 BLUELINE"),
        ("WVLY50C06", 20, 10, "POLY PIPE 50MM CLASS 6 BLUELINE"),
        ("WVLY63C06", 10, 5, "POLY PIPE 63MM CLASS 6 BLUELINE"),
        ("QUZH1094", 20, 10, "COMPRESSOR 3 PISTON"),
        ("QUZHMS1895", 10, None, "COMPRESSOR 4 PISTON"),
        ("QUZHMS1896", 10, None, "COMPRESSOR 2V"),
        ("RFBS2250", 400, None, "BLACKSHEETING 2MX 250 MICRO"),
        ("RFJS0019", 300, None, "BLACK SHEETING 100 MICRO X 2M"),
        ("SFC1269", 500, 200, "HELMET BLUE"),
        ("SFC1270", 500, 400, "HELMET BLUE W/CLIP"),
        ("SFC1271", 500, 100, "HELMET GREEN"),
        ("SFC1274", 500, None, "HELMET RED"),
        ("SFC1275", 500, None, "HELMET RED W/CLIP"),
        ("SFC1276", 500, 60, "HELMET WHITE"),
        ("SFC1277", 500, None, "HELMET WHITE W/CLIP"),
        ("SFC1278", 500, 100, "HELMET YELLOW"),
        ("SFC1279", 500, 100, "HELMET YELLOW W/CLIP"),
        ("SFCMS1842", 500, None, "HELMET GREEN W/CLIP"),
        ("TGR123ASR", 600, 300, "SAMPLE DISH SMALL"),
        ("TGNS1482", 240, None, "SHOVELS EAZY TOOLS"),
        ("TGNS1483", 240, None, "SHOVELS BLACK (LIGHT)"),
        ("TTR1564", 600, None, "V BELT B1900"),
        ("TTR1574", 600, None, "V BELT B2100"),
        ("TTR1587", 600, None, "V BELT B2540"),
        ("TTR1592", 600, None, "V BELT B2600"),
        ("TTR1673", 600, None, "V BELT B1850/71"),
        ("TTRS6079", 600, None, "V BELT B1750"),
        ("TTRS6083", 600, None, "V BELT B2800"),
        ("TTRS6084", 600, 100, "V BELT B1800"),
        ("VP1548", 800, 800, "TORCH ONE LAMP"),
        ("TTR1585", 600, 600, "V BELT B2500"),
    ],
}


_CATEGORY_RULES = [
    ("V BELT", "V-Belts"),
    ("BOLT", "Fasteners"), ("NUT", "Fasteners"), ("WASHER", "Fasteners"), ("WEDGE", "Fasteners"),
    ("CABLE", "Electrical"), ("RECTIFIER", "Electrical"), ("ALTERNATOR", "Electrical"),
    ("HELMET", "PPE"), ("GUMBOOT", "PPE"), ("MUTTON CLOTH", "PPE"),
    ("POLY PIPE", "Piping"), ("PIPE", "Piping"),
    ("GREASE", "Lubricants"), ("HYSPIN", "Lubricants"), ("SUPER SERIES", "Lubricants"),
    ("DEGREASER", "Chemicals"),
    ("HAMMERMILL", "Milling Spares"), ("MILL BALLS", "Milling Spares"),
    ("STAMPMILL", "Milling Spares"), ("BEATER", "Milling Spares"), ("H/HEAD", "Milling Spares"),
    ("CRUSHER", "Crushing Spares"), ("JAW", "Crushing Spares"), ("BUNGA", "Crushing Spares"),
    ("COMPRESSOR", "Plant & Equipment"), ("WATER PUMP", "Plant & Equipment"), ("PUMP", "Plant & Equipment"),
    ("ROLLER", "Conveyor Spares"), ("P/BLOCK", "Conveyor Spares"), ("PILLOW", "Conveyor Spares"),
    ("SHEETING", "General"), ("FENCE", "General"), ("SHOVEL", "Hand Tools"),
    ("SPANNER", "Hand Tools"), ("PICK HEAD", "Hand Tools"), ("TORCH", "General"),
    ("SAMPLE DISH", "Assay"), ("CLAY POT", "Assay"),
]


def categorise(description: str) -> str:
    up = description.upper()
    for kw, cat in _CATEGORY_RULES:
        if kw in up:
            return cat
    return "General"


# Rough indicative unit prices per category (USD) so value-weighted analytics work.
_CATEGORY_PRICE = {
    "V-Belts": 9.5, "Fasteners": 0.35, "Electrical": 2.1, "PPE": 4.0, "Piping": 6.0,
    "Lubricants": 18.0, "Chemicals": 7.5, "Milling Spares": 22.0, "Crushing Spares": 140.0,
    "Plant & Equipment": 380.0, "Conveyor Spares": 45.0, "Hand Tools": 8.0,
    "Assay": 3.2, "General": 5.0,
}


def indicative_price(description: str) -> float:
    """Deterministic pseudo-price: category base +/- a stable per-item wobble."""
    base = _CATEGORY_PRICE.get(categorise(description), 5.0)
    wobble = 0.8 + (sum(ord(c) for c in description) % 45) / 100.0  # 0.80..1.24
    return round(base * wobble, 2)

