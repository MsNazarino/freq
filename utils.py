"""
utils.py — спільні функції сегментації клієнтів за портфелем транспортних засобів.

Використовується в:
- pipeline.ipynb — повний цикл: дані → ознаки → KMeans → дерево → правила → сегменти;
- score.ipynb   — присвоєння сегментів лише за збереженими правилами (без KMeans і sklearn).

Джерела ознак:
- пальне: колонка fuel_type;
- причепи: kind + body_type (+ total_weight), з погодженими винятками;
- SUV: модель (шаблони з межами слова, частина прив'язана до бренду) + body_type.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. КОНСТАНТИ
# =============================================================================
RANDOM_STATE = 42
REFERENCE_YEAR = 2026          # рік, від якого рахується вік авто
MAX_VEHICLE_AGE = 50           # вік обрізається зверху до цього значення
PLACEHOLDER_YEAR = 1900        # технічне значення року → невідомий рік

LUXURY_VALUE_USD = 60_000            # легковик від цієї вартості → Luxury
MODERN_PREMIUM_MAX_AGE = 9           # premium-авто не старше 9 років → Modern Premium
MODERN_PREMIUM_MIN_VALUE_USD = 30_000  # або premium-авто від $30k → Modern Premium
HEAVY_TRAILER_WEIGHT_KG = 3_500      # причіп важчий за 3.5 т → HEAVY
MULTI_CAR_MIN = 3                    # від 3 легковиків → MULTI_CAR_3PLUS

VALUE_TIER_BINS = [-np.inf, 4_000, 14_000, 30_000, 60_000, np.inf]
VALUE_TIER_LABELS = ["T1", "T2", "T3", "T4", "T5"]

# Ознаки для KMeans (відскейлені в pipeline)
CLUSTER_FEATURES = [
    "LOG_TOTAL_VEHICLE_VALUE",
    "COUNT_LOG1P",
    "AGE_IMP",
    "PREMIUM_SHARE",
    "SUV_SHARE",
    "EV_SHARE",
]

# Ознаки для дерева правил (у зрозумілих одиницях: долари, роки, частки)
TREE_FEATURES = [
    "TOTAL_VEHICLE_VALUE_USD",
    "MAX_VEHICLE_VALUE_USD",
    "PASSENGER_VEHICLE_COUNT",
    "AVG_VEHICLE_AGE",
    "PREMIUM_SHARE",
    "SUV_SHARE",
    "EV_SHARE",
]

# Сегменти, що задаються правилами ієрархії (не деревом)
RULE_SEGMENTS = [
    "COMMERCIAL_VEHICLE_OWNERS",
    "MOTORCYCLE_OWNERS",
    "TRAILER_ONLY_OWNERS",
    "OTHER_NON_PASSENGER",
    "BUSINESS_VEHICLE_OWNERS",
    "MULTI_CAR_3PLUS",
    "LUXURY",
    "MODERN_PREMIUM",
    "EV_ADOPTERS",
]


# =============================================================================
# 2. ДОВІДНИКИ
# =============================================================================
KIND_MAP = {
    "1": "ЛЕГКОВИЙ",
    "2": "ВАНТАЖНИЙ",
    "3": "АВТОБУС",
    "4": "ПРИЧІП",
    "5": "НАПІВПРИЧІП",
    "6": "МОТОЦИКЛ",
    "7": "МОПЕД",
    "9": "КВАДРОЦИКЛ",
}

LUXURY_BRANDS = {
    "BENTLEY", "ROLLS-ROYCE", "MASERATI", "PORSCHE", "FERRARI",
    "LAMBORGHINI", "ASTON MARTIN", "MCLAREN",
}

PREMIUM_BRANDS = {
    "AUDI", "BMW", "MERCEDES-BENZ", "LEXUS", "LAND ROVER", "JAGUAR",
    "VOLVO", "TESLA", "POLESTAR", "DS", "CUPRA", "ZEEKR", "INFINITI",
    "CADILLAC", "ACURA", "GENESIS", "LINCOLN",
}

# Premium-ознака завжди рахується як premium + luxury
PREMIUM_OR_LUXURY_BRANDS = PREMIUM_BRANDS | LUXURY_BRANDS

PASSENGER_KINDS = {"ЛЕГКОВИЙ"}
TRAILER_KINDS = {"ПРИЧІП", "НАПІВПРИЧІП"}
MOTO_KINDS = {"МОТОЦИКЛ", "МОПЕД", "ТРИЦИКЛ", "КВАДРОЦИКЛ", "МОТОРОЛЕР", "СКУТЕР"}
BUS_KEYWORDS = ("АВТОБУС",)

SPECIAL_COMMERCIAL_KINDS = {
    "ЕКСКАВАТОР-НАВАНТАЖУВАЧ",
    "ЕЛЕКТРОНАВАНТАЖУВАЧ ВИЛКОВИЙ",
    "МІНІ-ЕКСКАВАТОР",
    "НАВАНТАЖУВАЧ ВИЛКОВИЙ",
    "АВТОНАВАНТАЖУВАЧ ВИЛКОВИЙ",
    "НАВАНТАЖУВАЧ ТЕЛЕСКОПІЧНИЙ",
    "НАВАНТАЖУВАЧ ФРОНТАЛЬНИЙ",
}
COMMERCIAL_KINDS = {"ВАНТАЖНИЙ"} | SPECIAL_COMMERCIAL_KINDS
# Ключові слова для інших видів спецтехніки, яких немає в списку вище
COMMERCIAL_KEYWORDS = (
    "ВАНТАЖ", "НАВАНТАЖУВАЧ", "ЕКСКАВАТОР", "ТРАКТОР", "БУЛЬДОЗЕР",
    "ГРЕЙДЕР", "КОМБАЙН", "КАТОК", "КРАН", "СПЕЦІАЛ",
)

FUEL_EXACT_MAP = {
    "БЕНЗИН": "PETROL",
    "ДИЗЕЛЬНЕ ПАЛИВО": "DIESEL",
    "ЕЛЕКТРО": "ELECTRIC",
    "ГАЗ": "GAS",
}
FUEL_UNKNOWN_VALUES = {"", ".", "ВІДСУТНЄ", "НЕ ВИЗНАЧЕНО", "NAN", "NONE", "UNKNOWN"}


# =============================================================================
# 3. SUV-ДОВІДНИК
# =============================================================================
# Межі слова: шаблон має стояти окремим словом у назві моделі.
# Після назви допускаються лише пробіл, дефіс, дужка, кома, крапка або кінець рядка.
_START = r"(?:^|[\s(/])"
_END = r"(?=$|[\s\-(),./])"

# Бренди, у яких усі легкові моделі — SUV
SUV_ALL_MODEL_BRANDS = {"LAND ROVER", "JEEP", "HAVAL", "HUMMER"}

# Назви, достатньо унікальні, щоб шукати їх у моделях будь-якого бренду
SUV_GENERIC_MODELS = [
    # Toyota
    r"RAV\s?-?4", r"LAND\s?CRUISER", r"PRADO", r"HIGHLANDER", r"C-?HR", r"FORTUNER",
    r"SEQUOIA", r"4RUNNER", r"FJ\s?CRUISER", r"VENZA", r"BZ4X",
    r"COROLLA\s+CROSS", r"YARIS\s+CROSS",
    # Honda
    r"CR-?V", r"HR-?V", r"ZR-?V", r"PILOT", r"PASSPORT",
    # Nissan
    r"QASHQAI", r"X-?TRAIL", r"ROGUE", r"JUKE", r"MURANO", r"PATHFINDER",
    r"PATROL", r"TERRANO", r"ARIYA", r"KICKS", r"ARMADA",
    # Mitsubishi
    r"OUTLANDER", r"PAJERO(?:\s?SPORT)?", r"ASX", r"ECLIPSE\s+CROSS",
    # Subaru
    r"FORESTER", r"OUTBACK", r"XV", r"CROSSTREK", r"TRIBECA", r"SOLTERRA",
    # Hyundai
    r"TUCSON", r"SANTA\s?-?FE", r"KONA", r"CRETA", r"IX\s?35", r"IX\s?55",
    r"PALISADE", r"VENUE", r"BAYON", r"IONIQ\s?5", r"NEXO", r"GALLOPER", r"TERRACAN",
    # Kia
    r"SPORTAGE", r"SORENTO", r"SELTOS", r"NIRO", r"STONIC", r"TELLURIDE",
    r"MOHAVE", r"EV9",
    # Volkswagen / Skoda / Seat / Cupra
    r"TIGUAN(?:\s?ALLSPACE)?", r"TOUAREG", r"T-?CROSS", r"T-?ROC", r"TAIGO",
    r"ATLAS", r"TERAMONT", r"KODIAQ", r"KAROQ", r"KAMIQ", r"ENYAQ", r"YETI",
    r"ATECA", r"ARONA", r"TARRACO", r"FORMENTOR",
    # Renault / Dacia
    r"DUSTER", r"CAPTUR", r"KOLEOS", r"KADJAR", r"ARKANA", r"AUSTRAL", r"BIGSTER",
    # Citroen / Opel
    r"C[345]\s?AIRCROSS", r"AIRCROSS", r"MOKKA", r"GRANDLAND", r"CROSSLAND",
    r"ANTARA", r"FRONTERA",
    # Ford
    r"KUGA", r"ESCAPE", r"EDGE", r"EXPLORER", r"ECOSPORT", r"PUMA", r"BRONCO",
    r"EXPEDITION", r"MACH-?E",
    # Chevrolet / Dodge
    r"CAPTIVA", r"EQUINOX", r"TAHOE", r"TRAILBLAZER", r"TRAX", r"BLAZER",
    r"SUBURBAN", r"TRAVERSE", r"NIVA", r"DURANGO", r"JOURNEY", r"NITRO",
    # Suzuki
    r"(?:GRAND\s)?VITARA", r"JIMNY", r"S-?CROSS",
    # SsangYong
    r"KORANDO", r"REXTON", r"KYRON", r"ACTYON", r"TIVOLI", r"TORRES",
    # Китайські бренди
    r"TIGGO\s?\d{0,2}", r"COOLRAY", r"MONJARO", r"TUGELLA", r"JOLION", r"HOVER",
    r"ATTO\s?3", r"SONG(?:\s?PLUS)?", r"TANG", r"YUAN(?:\s?PLUS)?",
    # Mercedes-Benz (довгі назви)
    r"GL[ABCEKS]\s?\d{0,3}", r"G-?CLASS", r"G-?KLASSE", r"GELANDEWAGEN",
    # Premium / Luxury
    r"MODEL\s?[XY]", r"CAYENNE", r"MACAN", r"BENTAYGA", r"URUS", r"LEVANTE",
    r"GRECALE", r"CULLINAN", r"DBX", r"ESCALADE", r"NAVIGATOR", r"AVIATOR",
    r"NAUTILUS", r"LYRIQ", r"[FEI]-?PACE", r"XC\s?\d{2}", r"EX\s?[39]0",
    r"QX\s?\d{2}", r"FX\s?\d{2}", r"MDX", r"RDX", r"ZDX", r"GV\s?\d{2}",
]

# Короткі або числові назви — шукаються лише в межах свого бренду
SUV_BRAND_MODELS = {
    ("AUDI",): r"(?:R?S)?Q\s?[2-8](?:\s?SPORTBACK)?|E-?TRON(?!\s*GT)",
    ("BMW",): r"X[1-7]M?|XM|IX[13]?",
    ("LEXUS",): r"(?:UX|NX|RX|GX|LX|RZ|TX)\s?\d{0,3}[HL]?",
    ("MERCEDES-BENZ", "MERCEDES"): r"(?:ML|GL|G)\s?\d{0,3}|EQ[ABC]\s?\d{0,3}",
    ("PEUGEOT",): r"E?-?[2345]008",
    ("MAZDA",): r"CX-?\s?\d{1,2}|TRIBUTE",
    ("VOLKSWAGEN", "VW"): r"ID\.?\s?[456](?:\s?X)?",
    ("ВАЗ", "LADA", "VAZ"): r"NIVA|НИВА|4X4|2121\d?|2131\d?",
    ("УАЗ", "UAZ"): r"PATRIOT|HUNTER|469|315\d{1,2}",
    ("MG",): r"ZS|HS|EHS|MARVEL\s?R",
    ("CHANGAN",): r"CS\s?\d{2}|UNI-?[KT]",
    ("LINCOLN",): r"MK[XCT]|CORSAIR",
    ("CADILLAC",): r"XT[456]|SRX",
}

SUV_BODY_KEYWORDS = r"КРОСОВЕР|ПОЗАШЛЯХОВИК|SUV"


def _wrap_words(pattern):
    """Обгортає шаблон межами слова."""
    return _START + "(?:" + pattern + ")" + _END


SUV_GENERIC_REGEX = re.compile(_wrap_words("|".join(SUV_GENERIC_MODELS)))
SUV_BRAND_REGEX = {
    brands: re.compile(_wrap_words(pattern))
    for brands, pattern in SUV_BRAND_MODELS.items()
}


def detect_suv(brand, model, body):
    """Повертає булеву серію: чи є авто SUV (за моделлю, брендом або кузовом)."""
    is_suv = brand.isin(SUV_ALL_MODEL_BRANDS)
    is_suv |= model.str.contains(SUV_GENERIC_REGEX, na=False)
    for brands, regex in SUV_BRAND_REGEX.items():
        in_brand = brand.isin(brands)
        if in_brand.any():
            is_suv |= in_brand & model.str.contains(regex, na=False)
    is_suv |= body.str.contains(SUV_BODY_KEYWORDS, regex=True, na=False)
    return is_suv


# =============================================================================
# 4. НОРМАЛІЗАЦІЯ
# =============================================================================
def _clean_text(series):
    """Текст у верхньому регістрі без зайвих пробілів; пропуски → 'UNKNOWN'."""
    out = series.astype("object").where(series.notna(), "UNKNOWN")
    out = out.astype(str).str.upper().str.strip()
    out = out.str.replace(r"\s+", " ", regex=True)
    return out.replace("", "UNKNOWN")


def normalize_kind(value):
    """Числові коди kind ('1', '1.0', 1) → назви; текстові значення — як є."""
    if pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip().upper()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit():
        return KIND_MAP.get(text, "UNKNOWN")
    return text if text else "UNKNOWN"


def classify_fuel(value):
    """fuel_type → один із 6 класів: PETROL / DIESEL / ELECTRIC / GAS / HYBRID / UNKNOWN."""
    if pd.isna(value):
        return "UNKNOWN"
    text = re.sub(r"\s+", " ", str(value).strip().upper())
    if text in FUEL_UNKNOWN_VALUES:
        return "UNKNOWN"
    # Будь-яке поєднання видів пального → HYBRID (включно з бензин/газ)
    if " АБО " in text or " ТА " in text or "," in text:
        return "HYBRID"
    return FUEL_EXACT_MAP.get(text, "UNKNOWN")


# =============================================================================
# 5. ПІДГОТОВКА ДАНИХ НА РІВНІ АВТО
# =============================================================================
def load_raw(path, sheet_name=0):
    """Читає вхідний Excel-файл."""
    return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")


def prepare_vehicle_data(raw):
    """
    Додає до кожного ТЗ класи та ознаки.
    Рядок = один транспортний засіб.
    """
    d = raw.copy()

    # --- Текстові поля
    d["KIND_STD"] = d["kind"].map(normalize_kind)
    d["BRAND_STD"] = _clean_text(d["BRAND_STD"])
    d["MODEL_STD"] = _clean_text(d["MODEL_STD"])
    d["BODY_STD"] = _clean_text(d["body_type"])

    # --- Пальне (класифікуємо лише унікальні значення — швидше)
    fuel_raw = d["fuel_type"]
    fuel_map = {v: classify_fuel(v) for v in fuel_raw.dropna().unique()}
    d["FUEL_CLASS"] = fuel_raw.map(fuel_map).fillna("UNKNOWN")

    # --- Рік і вік
    year = pd.to_numeric(d["production_year"], errors="coerce")
    year = year.mask(year.eq(PLACEHOLDER_YEAR))
    d["PRODUCTION_YEAR_CLEAN"] = year
    d["VEHICLE_AGE_CLEAN"] = (REFERENCE_YEAR - year).clip(lower=0, upper=MAX_VEHICLE_AGE)

    d["MARKET_VALUE_USD"] = pd.to_numeric(d["MARKET_VALUE_USD"], errors="coerce")
    weight = pd.to_numeric(d["total_weight"], errors="coerce")

    brand = d["BRAND_STD"]
    model = d["MODEL_STD"]
    body = d["BODY_STD"]
    kind = d["KIND_STD"]

    # --- Погоджені винятки
    van_ignore = body.eq("Н/ПР-АВТОВОЗ") & (
        (brand.eq("OPEL") & model.eq("VIVARO"))
        | (brand.eq("FIAT") & model.eq("DUCATO"))
        | (brand.eq("FORD") & model.eq("TRANSIT CONNECT"))
    )
    forced_light = brand.eq("ОДИСЕЙ") & model.eq("01")
    forced_heavy = brand.eq("KOGEL") & model.eq("JN 24")

    # --- Причепи: kind АБО body_type, потім винятки
    trailer_by_kind = kind.isin(TRAILER_KINDS) | kind.str.startswith(("ПР-", "Н/ПР-"))
    trailer_by_body = body.str.startswith(("ПР-", "Н/ПР-"))
    is_trailer = (trailer_by_kind | trailer_by_body | forced_light | forced_heavy) & ~van_ignore

    is_boat = is_trailer & body.eq("ПР-Д/П ЧОВНІВ")
    is_named_light = is_trailer & body.str.startswith("ПР-ЛЕГКОВИЙ")
    is_semi = is_trailer & (kind.eq("НАПІВПРИЧІП") | body.str.startswith("Н/ПР-"))
    heavy_by_weight = is_trailer & ~is_boat & ~is_named_light & weight.gt(HEAVY_TRAILER_WEIGHT_KG)

    is_heavy = (is_semi | heavy_by_weight | forced_heavy) & ~is_boat & ~forced_light
    is_light = is_trailer & ~is_boat & ~is_heavy

    d["TRAILER_TYPE"] = np.select(
        [is_boat, is_heavy, is_light], ["BOAT", "HEAVY", "LIGHT"], default="NONE"
    )
    d["IS_LIGHT_TRAILER"] = is_light.astype("int8")
    d["IS_BOAT_TRAILER"] = is_boat.astype("int8")
    d["IS_HEAVY_TRAILER"] = is_heavy.astype("int8")
    d["IS_CAMPER_TRAILER"] = (is_trailer & body.eq("ПР-ФУРГОН ЖИТЛОВИЙ")).astype("int8")
    d["TRAILER_WITHOUT_WEIGHT"] = (is_trailer & weight.isna()).astype("int8")

    # --- Класи ТЗ: кожен ТЗ рівно в одному класі (порядок має значення)
    is_passenger = kind.isin(PASSENGER_KINDS) & ~is_trailer
    is_bus = kind.str.contains("|".join(BUS_KEYWORDS), na=False) & ~is_trailer
    is_moto = kind.isin(MOTO_KINDS) & ~is_trailer
    is_commercial = (
        kind.isin(COMMERCIAL_KINDS) | kind.str.contains("|".join(COMMERCIAL_KEYWORDS), na=False)
    ) & ~is_trailer & ~is_passenger & ~is_bus & ~is_moto

    d["VEHICLE_CLASS"] = np.select(
        [is_trailer, is_passenger, is_bus, is_moto, is_commercial],
        ["TRAILER", "PASSENGER", "BUS", "MOTO", "COMMERCIAL"],
        default="OTHER",
    )
    for cls in ["PASSENGER", "COMMERCIAL", "BUS", "MOTO", "TRAILER", "OTHER"]:
        d[f"IS_{cls}"] = d["VEHICLE_CLASS"].eq(cls).astype("int8")

    class_count = d[[f"IS_{c}" for c in ["PASSENGER", "COMMERCIAL", "BUS", "MOTO", "TRAILER", "OTHER"]]].sum(axis=1)
    assert class_count.eq(1).all(), "Кожен ТЗ має належати рівно до одного класу"

    passenger = d["IS_PASSENGER"].eq(1)

    # --- Ознаки легковиків
    d["IS_PREMIUM_BRAND"] = (passenger & brand.isin(PREMIUM_OR_LUXURY_BRANDS)).astype("int8")
    d["IS_EV"] = (passenger & d["FUEL_CLASS"].eq("ELECTRIC")).astype("int8")
    d["IS_HYBRID"] = (passenger & d["FUEL_CLASS"].eq("HYBRID")).astype("int8")
    d["IS_GAS"] = (passenger & d["FUEL_CLASS"].eq("GAS")).astype("int8")
    d["IS_SUV"] = (passenger & detect_suv(brand, model, body)).astype("int8")

    d["PASS_VALUE"] = d["MARKET_VALUE_USD"].where(passenger)
    d["PASS_AGE"] = d["VEHICLE_AGE_CLEAN"].where(passenger)

    d["IS_TIER5"] = (passenger & d["PASS_VALUE"].ge(LUXURY_VALUE_USD)).astype("int8")
    d["IS_MODERN_PREMIUM"] = (
        d["IS_PREMIUM_BRAND"].eq(1)
        & (d["VEHICLE_AGE_CLEAN"].le(MODERN_PREMIUM_MAX_AGE) | d["PASS_VALUE"].ge(MODERN_PREMIUM_MIN_VALUE_USD))
    ).astype("int8")

    tiers = pd.cut(d["PASS_VALUE"], bins=VALUE_TIER_BINS, right=False, labels=VALUE_TIER_LABELS)
    d["VALUE_TIER"] = tiers.astype("object").where(passenger, "NONE")

    return d


# =============================================================================
# 6. АГРЕГАЦІЯ ДО РІВНЯ КЛІЄНТА
# =============================================================================
def aggregate_customers(vehicles, age_median):
    """
    Рядок = один клієнт.
    age_median — медіана середнього віку легковиків (рахується в pipeline і
    зберігається в конфігурації, щоб score заповнював пропуски тим самим значенням).
    """
    # Прапорці int8 переводимо в int64 перед сумуванням: у деяких версіях pandas
    # сума int8 по групі переповнюється (> 127 ТЗ у клієнта → від'ємні значення).
    v = vehicles.copy()
    flag_cols = [col for col in v.columns if col.startswith("IS_")]
    v[flag_cols] = v[flag_cols].astype("int64")

    agg = {
        "TOTAL_VEHICLE_COUNT": ("user_id", "size"),
        "PASSENGER_VEHICLE_COUNT": ("IS_PASSENGER", "sum"),
        "TOTAL_VEHICLE_VALUE_USD": ("PASS_VALUE", "sum"),
        "MAX_VEHICLE_VALUE_USD": ("PASS_VALUE", "max"),
        "AVG_VEHICLE_AGE_RAW": ("PASS_AGE", "mean"),
        "PREMIUM_COUNT": ("IS_PREMIUM_BRAND", "sum"),
        "SUV_COUNT": ("IS_SUV", "sum"),
        "EV_COUNT": ("IS_EV", "sum"),
        "HYBRID_COUNT": ("IS_HYBRID", "sum"),
        "GAS_COUNT": ("IS_GAS", "sum"),
        "TIER5_COUNT": ("IS_TIER5", "sum"),
        "MODERN_PREMIUM_COUNT": ("IS_MODERN_PREMIUM", "sum"),
        "TRAILER_COUNT": ("IS_TRAILER", "sum"),
        "LIGHT_TRAILER_COUNT": ("IS_LIGHT_TRAILER", "sum"),
        "BOAT_TRAILER_COUNT": ("IS_BOAT_TRAILER", "sum"),
        "HEAVY_TRAILER_COUNT": ("IS_HEAVY_TRAILER", "sum"),
        "CAMPER_TRAILER_COUNT": ("IS_CAMPER_TRAILER", "sum"),
        "BUS_COUNT": ("IS_BUS", "sum"),
        "COMMERCIAL_COUNT": ("IS_COMMERCIAL", "sum"),
        "MOTO_COUNT": ("IS_MOTO", "sum"),
        "OTHER_COUNT": ("IS_OTHER", "sum"),
    }
    c = v.groupby("user_id", as_index=False).agg(**agg)

    c["TOTAL_VEHICLE_VALUE_USD"] = c["TOTAL_VEHICLE_VALUE_USD"].fillna(0)
    c["MAX_VEHICLE_VALUE_USD"] = c["MAX_VEHICLE_VALUE_USD"].fillna(0)

    has_passenger = c["PASSENGER_VEHICLE_COUNT"].gt(0)
    c["AGE_UNKNOWN"] = (has_passenger & c["AVG_VEHICLE_AGE_RAW"].isna()).astype("int8")
    c["AVG_VEHICLE_AGE"] = c["AVG_VEHICLE_AGE_RAW"].fillna(age_median)

    # Частки — серед легковиків клієнта
    denominator = c["PASSENGER_VEHICLE_COUNT"].replace(0, np.nan)
    for count_col, share_col in [
        ("PREMIUM_COUNT", "PREMIUM_SHARE"),
        ("SUV_COUNT", "SUV_SHARE"),
        ("EV_COUNT", "EV_SHARE"),
        ("HYBRID_COUNT", "HYBRID_SHARE"),
        ("GAS_COUNT", "GAS_SHARE"),
        ("TIER5_COUNT", "TIER5_SHARE"),
    ]:
        c[share_col] = (c[count_col] / denominator).fillna(0)

    # Частка причепів — від усіх ТЗ клієнта
    c["TRAILER_SHARE"] = c["TRAILER_COUNT"] / c["TOTAL_VEHICLE_COUNT"]
    c["BUSINESS_VEHICLE_COUNT"] = c["COMMERCIAL_COUNT"] + c["BUS_COUNT"] + c["HEAVY_TRAILER_COUNT"]

    # Ознаки для KMeans
    c["LOG_TOTAL_VEHICLE_VALUE"] = np.log1p(c["TOTAL_VEHICLE_VALUE_USD"])
    c["COUNT_LOG1P"] = np.log1p(c["PASSENGER_VEHICLE_COUNT"])
    c["AGE_IMP"] = c["AVG_VEHICLE_AGE"]

    # Бінарні прапорці
    for count_col, flag_col in [
        ("PREMIUM_COUNT", "HAS_PREMIUM"),
        ("SUV_COUNT", "HAS_SUV"),
        ("EV_COUNT", "HAS_EV"),
        ("HYBRID_COUNT", "HAS_HYBRID"),
        ("GAS_COUNT", "HAS_GAS"),
        ("TIER5_COUNT", "HAS_LUXURY_VEHICLE"),
        ("MODERN_PREMIUM_COUNT", "HAS_MODERN_PREMIUM"),
        ("TRAILER_COUNT", "HAS_TRAILER"),
        ("LIGHT_TRAILER_COUNT", "HAS_LIGHT_TRAILER"),
        ("BOAT_TRAILER_COUNT", "HAS_BOAT_TRAILER"),
        ("HEAVY_TRAILER_COUNT", "HAS_HEAVY_TRAILER"),
        ("CAMPER_TRAILER_COUNT", "HAS_CAMPER_TRAILER"),
        ("BUS_COUNT", "HAS_BUS"),
        ("COMMERCIAL_COUNT", "HAS_COMMERCIAL"),
        ("MOTO_COUNT", "HAS_MOTO"),
    ]:
        c[flag_col] = c[count_col].gt(0).astype("int8")
    c["IS_MULTI_VEHICLE"] = c["PASSENGER_VEHICLE_COUNT"].ge(2).astype("int8")

    count_cols = [col for col in c.columns if col.endswith("_COUNT")]
    negative = c[count_cols].lt(0).any()
    assert not negative.any(), f"Від'ємні лічильники: {list(negative[negative].index)}"
    assert c["user_id"].is_unique
    return c


def compute_age_median(vehicles):
    """Медіана середнього віку легковиків по клієнтах (для заповнення пропусків)."""
    passenger_age = vehicles.loc[vehicles["IS_PASSENGER"].eq(1)].groupby("user_id")["PASS_AGE"].mean()
    return float(passenger_age.median())


# =============================================================================
# 7. ІЄРАРХІЯ СЕГМЕНТІВ
# =============================================================================
def rule_masks(c, ev_rule, business_threshold):
    """
    Маски сегментів, що задаються правилами (у порядку пріоритету).
    Повертає список (назва сегмента, маска) і маску популяції дерева.
    """
    no_passenger = c["PASSENGER_VEHICLE_COUNT"].eq(0)
    has_passenger = ~no_passenger
    taken = pd.Series(False, index=c.index)
    masks = []

    def add(name, condition):
        nonlocal taken
        mask = condition & ~taken
        masks.append((name, mask))
        taken = taken | mask

    # Популяція B: без легковиків
    add("COMMERCIAL_VEHICLE_OWNERS", no_passenger & c["BUSINESS_VEHICLE_COUNT"].ge(1))
    add("MOTORCYCLE_OWNERS", no_passenger & c["HAS_MOTO"].eq(1))
    add("TRAILER_ONLY_OWNERS", no_passenger & (c["HAS_LIGHT_TRAILER"].eq(1) | c["HAS_BOAT_TRAILER"].eq(1)))
    add("OTHER_NON_PASSENGER", no_passenger)

    # Популяція A: є легковики
    add("BUSINESS_VEHICLE_OWNERS", has_passenger & c["BUSINESS_VEHICLE_COUNT"].ge(business_threshold))
    add("MULTI_CAR_3PLUS", c["PASSENGER_VEHICLE_COUNT"].ge(MULTI_CAR_MIN))
    add("LUXURY", has_passenger & c["HAS_LUXURY_VEHICLE"].eq(1))
    add("MODERN_PREMIUM", has_passenger & c["HAS_MODERN_PREMIUM"].eq(1))
    if ev_rule:
        add("EV_ADOPTERS", has_passenger & c["EV_SHARE"].eq(1))

    tree_population = ~taken
    return masks, tree_population


def tree_population_mask(c, ev_rule, business_threshold):
    """Клієнти, яким сегмент присвоює дерево (і на яких навчається KMeans)."""
    _, tree_population = rule_masks(c, ev_rule, business_threshold)
    return tree_population


def assign_final_segments(customers, tree_rules, ev_rule, business_threshold):
    """Присвоює кожному клієнту рівно один фінальний сегмент."""
    c = customers.copy()
    masks, tree_population = rule_masks(c, ev_rule, business_threshold)

    c["FINAL_SEGMENT"] = pd.Series(pd.NA, index=c.index, dtype="object")
    c["SEGMENT_SOURCE"] = pd.Series(pd.NA, index=c.index, dtype="object")
    for name, mask in masks:
        c.loc[mask, "FINAL_SEGMENT"] = name
        c.loc[mask, "SEGMENT_SOURCE"] = "RULE"

    tree_segments = apply_rules(c.loc[tree_population], tree_rules)
    c.loc[tree_population, "FINAL_SEGMENT"] = tree_segments
    c.loc[tree_population, "SEGMENT_SOURCE"] = "TREE"

    missing = c["FINAL_SEGMENT"].isna()
    if missing.any():
        print(f"УВАГА: {missing.sum()} клієнтів без сегмента. Приклади:")
        print(c.loc[missing, ["user_id"] + TREE_FEATURES].head(10).to_string())
    assert not missing.any(), "Кожен клієнт має отримати сегмент"
    assert c["user_id"].is_unique
    return c


# =============================================================================
# 8. ПРАВИЛА З ДЕРЕВА
# =============================================================================
def round_threshold(feature, value):
    """Округлює поріг до зрозумілого бізнесу значення."""
    if "VALUE_USD" in feature:
        return float(max(500.0, round(value / 500) * 500))
    if "AGE" in feature:
        return float(max(1.0, round(value)))
    if "COUNT" in feature:
        return float(round(value * 2) / 2)
    # частки
    return float(min(0.95, max(0.05, round(value / 0.05) * 0.05)))


def rules_from_tree(model, feature_names):
    """
    Перетворює sklearn-дерево на список правил:
    - пороги округлюються;
    - сусідні листки з однаковим сегментом зливаються;
    - умови по одній ознаці стискаються до меж (нижня < X <= верхня);
    - суперечливі (недосяжні) після округлення правила відкидаються.
    """
    from sklearn.tree import _tree

    tree = model.tree_
    classes = model.classes_

    def walk(node):
        if tree.feature[node] == _tree.TREE_UNDEFINED:
            return {"kind": "leaf", "segment": str(classes[int(np.argmax(tree.value[node][0]))])}
        feature = feature_names[tree.feature[node]]
        left = walk(tree.children_left[node])
        right = walk(tree.children_right[node])
        if left["kind"] == "leaf" and right["kind"] == "leaf" and left["segment"] == right["segment"]:
            return left
        return {
            "kind": "node",
            "feature": feature,
            "threshold": round_threshold(feature, tree.threshold[node]),
            "left": left,
            "right": right,
        }

    root = walk(0)
    raw_rules = []

    def flatten(node, conditions):
        if node["kind"] == "leaf":
            raw_rules.append({"conditions": conditions, "segment": node["segment"]})
            return
        f, t = node["feature"], node["threshold"]
        flatten(node["left"], conditions + [(f, "<=", t)])
        flatten(node["right"], conditions + [(f, ">", t)])

    flatten(root, [])

    rules = []
    for raw in raw_rules:
        bounds = {}
        for feature, op, threshold in raw["conditions"]:
            low, high = bounds.get(feature, (None, None))
            if op == "<=":
                high = threshold if high is None else min(high, threshold)
            else:
                low = threshold if low is None else max(low, threshold)
            bounds[feature] = (low, high)
        reachable = all(low is None or high is None or low < high for low, high in bounds.values())
        if not reachable:
            continue
        conditions = []
        for feature in feature_names:
            if feature not in bounds:
                continue
            low, high = bounds[feature]
            if low is not None:
                conditions.append({"feature": feature, "operator": ">", "threshold": low})
            if high is not None:
                conditions.append({"feature": feature, "operator": "<=", "threshold": high})
        rules.append({"rule_id": len(rules) + 1, "conditions": conditions, "segment": raw["segment"]})

    return {"version": 2, "feature_names": list(feature_names), "rules": rules}


def apply_rules(frame, rules):
    """Застосовує правила (перше правило, що спрацювало, визначає сегмент)."""
    out = pd.Series(pd.NA, index=frame.index, dtype="object")
    for rule in rules["rules"]:
        mask = pd.Series(True, index=frame.index)
        for cond in rule["conditions"]:
            values = frame[cond["feature"]]
            if cond["operator"] == "<=":
                mask &= values.le(cond["threshold"])
            else:
                mask &= values.gt(cond["threshold"])
        out.loc[mask & out.isna()] = rule["segment"]
    return out


def _fmt(value):
    return f"{int(value):,}".replace(",", " ") if float(value).is_integer() and value >= 100 else f"{value:g}"


def _rule_condition_text(rule):
    parts = []
    by_feature = {}
    for cond in rule["conditions"]:
        by_feature.setdefault(cond["feature"], {})[cond["operator"]] = cond["threshold"]
    for feature, ops in by_feature.items():
        if ">" in ops and "<=" in ops:
            parts.append(f"{_fmt(ops['>'])} < {feature} <= {_fmt(ops['<='])}")
        elif ">" in ops:
            parts.append(f"{feature} > {_fmt(ops['>'])}")
        else:
            parts.append(f"{feature} <= {_fmt(ops['<='])}")
    return " AND ".join(parts) if parts else "завжди"


def _rule_condition_sql(rule):
    parts = [f"{c['feature']} {c['operator']} {c['threshold']:g}" for c in rule["conditions"]]
    return " AND ".join(parts) if parts else "1 = 1"


def hierarchy_text(rules, ev_rule, business_threshold):
    """Повний опис ієрархії сегментів (перше правило, що спрацювало, визначає сегмент)."""
    lines = [
        "Правила перевіряються зверху вниз; перше, що спрацювало, визначає сегмент.",
        "",
        "ПОПУЛЯЦІЯ B (немає легковиків):",
        "  1. BUSINESS_VEHICLE_COUNT >= 1            -> COMMERCIAL_VEHICLE_OWNERS",
        "  2. HAS_MOTO = 1                           -> MOTORCYCLE_OWNERS",
        "  3. HAS_LIGHT_TRAILER = 1 або HAS_BOAT_TRAILER = 1 -> TRAILER_ONLY_OWNERS",
        "  4. інакше                                 -> OTHER_NON_PASSENGER",
        "",
        "ПОПУЛЯЦІЯ A (є легковики):",
        f"  5. BUSINESS_VEHICLE_COUNT >= {business_threshold}            -> BUSINESS_VEHICLE_OWNERS",
        f"  6. PASSENGER_VEHICLE_COUNT >= {MULTI_CAR_MIN}           -> MULTI_CAR_3PLUS",
        f"  7. є легковик >= ${LUXURY_VALUE_USD:,}              -> LUXURY",
        f"  8. є premium-авто віком <= {MODERN_PREMIUM_MAX_AGE} р. або >= ${MODERN_PREMIUM_MIN_VALUE_USD:,} -> MODERN_PREMIUM",
    ]
    step = 9
    if ev_rule:
        lines.append(f"  {step}. EV_SHARE = 1 (усі легковики електро)   -> EV_ADOPTERS")
        step += 1
    lines += ["", f"  {step}. Решта (1-2 легковики) — правила дерева:"]
    for rule in rules["rules"]:
        lines.append(f"     R{rule['rule_id']:02d}: IF {_rule_condition_text(rule)} THEN {rule['segment']}")
    return "\n".join(lines)


def hierarchy_sql(rules, ev_rule, business_threshold):
    """Та сама ієрархія як SQL CASE (на таблиці клієнтів з агрегованими ознаками)."""
    whens = [
        ("PASSENGER_VEHICLE_COUNT = 0 AND BUSINESS_VEHICLE_COUNT >= 1", "COMMERCIAL_VEHICLE_OWNERS"),
        ("PASSENGER_VEHICLE_COUNT = 0 AND HAS_MOTO = 1", "MOTORCYCLE_OWNERS"),
        ("PASSENGER_VEHICLE_COUNT = 0 AND (HAS_LIGHT_TRAILER = 1 OR HAS_BOAT_TRAILER = 1)", "TRAILER_ONLY_OWNERS"),
        ("PASSENGER_VEHICLE_COUNT = 0", "OTHER_NON_PASSENGER"),
        (f"BUSINESS_VEHICLE_COUNT >= {business_threshold}", "BUSINESS_VEHICLE_OWNERS"),
        (f"PASSENGER_VEHICLE_COUNT >= {MULTI_CAR_MIN}", "MULTI_CAR_3PLUS"),
        ("HAS_LUXURY_VEHICLE = 1", "LUXURY"),
        ("HAS_MODERN_PREMIUM = 1", "MODERN_PREMIUM"),
    ]
    if ev_rule:
        whens.append(("EV_SHARE = 1", "EV_ADOPTERS"))
    for rule in rules["rules"]:
        whens.append((_rule_condition_sql(rule), rule["segment"]))

    body = "\n".join(f"    WHEN {cond}\n        THEN '{seg}'" for cond, seg in whens)
    return (
        "-- Вхід: таблиця клієнтів з агрегованими ознаками (utils.aggregate_customers)\n"
        "SELECT\n    user_id,\n    CASE\n"
        f"{body}\n"
        "        ELSE 'UNASSIGNED'\n    END AS FINAL_SEGMENT\n"
        "FROM customers;"
    )


# =============================================================================
# 9. ПРОФІЛІ ТА ВИХІДНІ ДАНІ
# =============================================================================
def profile_table(c, group_col):
    """Профіль груп: розмір, медіани, частки клієнтів з ознакою."""
    total = len(c)
    g = c.groupby(group_col)
    profile = pd.DataFrame({
        "CLIENTS": g.size(),
        "PCT": (g.size() / total * 100).round(2),
        "MED_CARS": g["PASSENGER_VEHICLE_COUNT"].median(),
        "PCT_2PLUS_CARS": (g["PASSENGER_VEHICLE_COUNT"].apply(lambda s: s.ge(2).mean()) * 100).round(1),
        "MED_TOTAL_VALUE": g["TOTAL_VEHICLE_VALUE_USD"].median().round(0),
        "MED_MAX_VALUE": g["MAX_VEHICLE_VALUE_USD"].median().round(0),
        "MED_AGE": g["AVG_VEHICLE_AGE"].median().round(1),
        "MEAN_PREMIUM_SHARE": g["PREMIUM_SHARE"].mean().round(3),
        "MEAN_SUV_SHARE": g["SUV_SHARE"].mean().round(3),
        "MEAN_EV_SHARE": g["EV_SHARE"].mean().round(3),
        "MEAN_HYBRID_SHARE": g["HYBRID_SHARE"].mean().round(3),
        "MEAN_GAS_SHARE": g["GAS_SHARE"].mean().round(3),
        "PCT_HAS_PREMIUM": (g["HAS_PREMIUM"].mean() * 100).round(1),
        "PCT_HAS_SUV": (g["HAS_SUV"].mean() * 100).round(1),
        "PCT_HAS_EV": (g["HAS_EV"].mean() * 100).round(1),
        "PCT_HAS_HYBRID": (g["HAS_HYBRID"].mean() * 100).round(1),
        "PCT_HAS_GAS": (g["HAS_GAS"].mean() * 100).round(1),
    })
    return profile.sort_values("CLIENTS", ascending=False)


OUTPUT_BASE_COLUMNS = [
    "user_id", "FINAL_SEGMENT", "SEGMENT_SOURCE",
    "TOTAL_VEHICLE_COUNT", "PASSENGER_VEHICLE_COUNT", "BUSINESS_VEHICLE_COUNT",
    "TOTAL_VEHICLE_VALUE_USD", "MAX_VEHICLE_VALUE_USD", "AVG_VEHICLE_AGE", "AGE_UNKNOWN",
    "PREMIUM_SHARE", "SUV_SHARE", "EV_SHARE", "HYBRID_SHARE", "GAS_SHARE",
    "TRAILER_COUNT", "TRAILER_SHARE", "HEAVY_TRAILER_COUNT", "BUS_COUNT", "COMMERCIAL_COUNT", "MOTO_COUNT",
]


def output_columns(customers):
    flags = sorted(col for col in customers.columns if col.startswith(("HAS_", "IS_")))
    return [col for col in OUTPUT_BASE_COLUMNS + flags if col in customers.columns]


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# =============================================================================
# 10. ЗВІТ: НАЗВИ, ОПИСИ, ПРАВИЛА СЛОВАМИ
# =============================================================================
SEGMENT_DISPLAY = {
    "LUXURY": "Luxury Vehicle Owners",
    "MODERN_PREMIUM": "Modern Premium Owners",
    "ESTABLISHED_PREMIUM": "Established Premium Owners",
    "ESTABLISHED_PREMIUM_SUV": "Established Premium SUV Owners",
    "MULTI_CAR_3PLUS": "Multi-Car Owners (3+)",
    "MULTI_VEHICLE_HOUSEHOLDS": "Multi-Vehicle Households",
    "MAINSTREAM_SUV": "Mainstream SUV Owners",
    "EV_ADOPTERS": "Electric Vehicle Adopters",
    "CORE_MASS_MARKET_LOWER_MID": "Core Mass Market Lower-Mid",
    "CORE_MASS_MARKET_BUDGET": "Core Mass Market Budget",
    "BUDGET_LEGACY": "Budget Legacy Owners",
    "BUSINESS_VEHICLE_OWNERS": "Business Vehicle Owners",
    "COMMERCIAL_VEHICLE_OWNERS": "Commercial Vehicle Owners",
    "MOTORCYCLE_OWNERS": "Motorcycle Owners",
    "TRAILER_ONLY_OWNERS": "Trailer-Only Owners",
    "OTHER_NON_PASSENGER": "Other (без легкового авто)",
}

SEGMENT_DESCRIPTION = {
    "LUXURY": "Є хоча б одне легкове авто вартістю від 60 000 USD.",
    "MODERN_PREMIUM": "Є авто premium-бренду не старше 9 років або вартістю від 30 000 USD.",
    "ESTABLISHED_PREMIUM": "Вживане авто premium-бренду (не SUV), старше й дешевше за Modern Premium.",
    "ESTABLISHED_PREMIUM_SUV": "Вживаний SUV / кросовер premium-бренду (BMW X5, Audi Q7, Lexus RX тощо).",
    "MULTI_CAR_3PLUS": "Три і більше легкових авто.",
    "MULTI_VEHICLE_HOUSEHOLDS": "Два легкові авто масових брендів або старший premium.",
    "MAINSTREAM_SUV": "Один SUV / кросовер масового бренду.",
    "EV_ADOPTERS": "Електромобіль масового бренду.",
    "CORE_MASS_MARKET_LOWER_MID": "Одне масове авто (не SUV) середньої вартості.",
    "CORE_MASS_MARKET_BUDGET": "Одне недороге масове авто (не SUV).",
    "BUDGET_LEGACY": "Дуже старе й дешеве легкове авто.",
    "BUSINESS_VEHICLE_OWNERS": "Є легкове авто, а також вантажний транспорт, спецтехніка, автобус або важкий причіп.",
    "COMMERCIAL_VEHICLE_OWNERS": "Немає легкового авто; є вантажний транспорт, спецтехніка, автобус або важкий причіп.",
    "MOTORCYCLE_OWNERS": "Немає легкового авто і бізнес-транспорту; є мотоцикл, мопед, трицикл або квадроцикл.",
    "TRAILER_ONLY_OWNERS": "Є лише легкий причіп або причіп для човна.",
    "OTHER_NON_PASSENGER": "Немає легкового авто; інші типи ТЗ.",
}

# Групи сегментів (порядок рядків у звіті)
SEGMENT_GROUPS = {
    "Заможні": ["LUXURY", "MODERN_PREMIUM", "ESTABLISHED_PREMIUM_SUV", "MULTI_CAR_3PLUS"],
    "Established Premium": ["ESTABLISHED_PREMIUM"],
    "Масові": ["CORE_MASS_MARKET_LOWER_MID", "MAINSTREAM_SUV", "MULTI_VEHICLE_HOUSEHOLDS", "EV_ADOPTERS"],
    "Бюджетні": ["CORE_MASS_MARKET_BUDGET", "BUDGET_LEGACY"],
    "Бізнес": ["BUSINESS_VEHICLE_OWNERS", "COMMERCIAL_VEHICLE_OWNERS"],
    "Без легкового авто": ["MOTORCYCLE_OWNERS", "TRAILER_ONLY_OWNERS", "OTHER_NON_PASSENGER"],
}
GROUP_COLORS = {
    "Заможні": "EDE7F6",
    "Established Premium": "F8E5F0",
    "Масові": "E3EEFA",
    "Бюджетні": "FFF6DA",
    "Бізнес": "E6F2E6",
    "Без легкового авто": "EDEDED",
    "Інші": "FFFFFF",
}


def segment_group(segment):
    for group, segments in SEGMENT_GROUPS.items():
        if segment in segments:
            return group
    return "Інші"


def segment_order(segments):
    """Сегменти в порядку груп; невідомі — в кінці."""
    ordered = [s for group in SEGMENT_GROUPS.values() for s in group if s in set(segments)]
    return ordered + sorted(set(segments) - set(ordered))


def _usd(value):
    return f"{int(round(value)):,}".replace(",", " ") + " USD"


def _years(value):
    value = int(round(value))
    if value % 10 == 1 and value % 100 != 11:
        word = "року"
    else:
        word = "років"
    return f"{value} {word}"


_SHARE_WORDS = {
    "PREMIUM_SHARE": ("premium-бренду", "premium-авто"),
    "SUV_SHARE": ("SUV / кросовер", "SUV"),
    "EV_SHARE": ("електромобіль", "електромобілів"),
}


def _share_text(feature, low, high, n_cars):
    """Частка серед 1–2 авто може бути лише 0, 0.5 або 1 — описуємо словами."""
    possible = [0.0, 0.5, 1.0] if n_cars != 1 else [0.0, 1.0]
    fits = [v for v in possible if (low is None or v > low) and (high is None or v <= high)]
    single, plural = _SHARE_WORDS[feature]
    if n_cars == 1:
        if fits == [1.0]:
            return f"авто — {single}" if feature != "PREMIUM_SHARE" else "авто premium-бренду"
        if fits == [0.0]:
            return f"авто не {single}" if feature != "PREMIUM_SHARE" else "авто не premium-бренду"
        return None
    mapping = {
        (0.0,): f"жодного {plural}" if feature != "PREMIUM_SHARE" else "жодного авто premium-бренду",
        (1.0,): f"усі авто — {single}" if feature != "PREMIUM_SHARE" else "усі авто premium-бренду",
        (0.5,): f"одне з двох авто — {single}" if feature != "PREMIUM_SHARE" else "одне з двох авто premium-бренду",
        (0.0, 0.5): f"не всі авто — {single}" if feature != "PREMIUM_SHARE" else "не всі авто premium-бренду",
        (0.5, 1.0): f"хоча б одне авто — {single}" if feature != "PREMIUM_SHARE" else "хоча б одне авто premium-бренду",
    }
    return mapping.get(tuple(fits))


def _range_text(label, low, high, fmt):
    if low is not None and high is not None:
        return f"{label} від {fmt(low)} до {fmt(high)}"
    if low is not None:
        return f"{label} понад {fmt(low)}"
    return f"{label} до {fmt(high)} включно"


def rule_business_text(rule):
    """Умова правила дерева звичайними словами (без назв змінних)."""
    bounds = {}
    for cond in rule["conditions"]:
        low, high = bounds.get(cond["feature"], (None, None))
        if cond["operator"] == ">":
            low = cond["threshold"]
        else:
            high = cond["threshold"]
        bounds[cond["feature"]] = (low, high)

    n_cars = None
    if "PASSENGER_VEHICLE_COUNT" in bounds:
        low, high = bounds["PASSENGER_VEHICLE_COUNT"]
        if high is not None and high < 2:
            n_cars = 1
        elif low is not None and low >= 1:
            n_cars = 2

    parts = []
    if n_cars == 1:
        parts.append("одне легкове авто")
    elif n_cars == 2:
        parts.append("два легкові авто")
    if "TOTAL_VEHICLE_VALUE_USD" in bounds:
        label = "вартість авто" if n_cars == 1 else "сумарна вартість легкових авто"
        parts.append(_range_text(label, *bounds["TOTAL_VEHICLE_VALUE_USD"], _usd))
    if "MAX_VEHICLE_VALUE_USD" in bounds:
        parts.append(_range_text("найдорожче авто", *bounds["MAX_VEHICLE_VALUE_USD"], _usd))
    if "AVG_VEHICLE_AGE" in bounds:
        label = "вік авто" if n_cars == 1 else "середній вік авто"
        parts.append(_range_text(label, *bounds["AVG_VEHICLE_AGE"], _years))
    for feature in ["PREMIUM_SHARE", "SUV_SHARE", "EV_SHARE"]:
        if feature in bounds:
            text = _share_text(feature, *bounds[feature], n_cars)
            if text:
                parts.append(text)
    text = "; ".join(parts) if parts else "усі інші клієнти"
    return text[0].upper() + text[1:] + "."


def hierarchy_business_table(rules, ev_rule, business_threshold):
    """Покрокові правила присвоєння сегмента для звіту (словами)."""
    business_words = (
        "є вантажний транспорт, спецтехніка, автобус або важкий причіп"
        if business_threshold == 1
        else f"є щонайменше {business_threshold} одиниці вантажного транспорту, спецтехніки, автобусів або важких причепів"
    )
    rows = [
        ("1", "Немає легкового авто", business_words[0].upper() + business_words[1:] + ".", "COMMERCIAL_VEHICLE_OWNERS"),
        ("2", "Немає легкового авто", "Є мотоцикл, мопед, трицикл або квадроцикл.", "MOTORCYCLE_OWNERS"),
        ("3", "Немає легкового авто", "Є лише легкий причіп або причіп для човна.", "TRAILER_ONLY_OWNERS"),
        ("4", "Немає легкового авто", "Усі інші клієнти без легкового авто.", "OTHER_NON_PASSENGER"),
        ("5", "Є хоча б одне легкове авто", "Крім легкового, " + business_words + ".", "BUSINESS_VEHICLE_OWNERS"),
        ("6", "Є хоча б одне легкове авто", f"{MULTI_CAR_MIN} і більше легкових авто.", "MULTI_CAR_3PLUS"),
        ("7", "Є хоча б одне легкове авто", f"Є легкове авто вартістю від {_usd(LUXURY_VALUE_USD)}.", "LUXURY"),
        ("8", "Є хоча б одне легкове авто",
         f"Є авто premium-бренду не старше {MODERN_PREMIUM_MAX_AGE} років або вартістю від {_usd(MODERN_PREMIUM_MIN_VALUE_USD)}.",
         "MODERN_PREMIUM"),
    ]
    step = 9
    if ev_rule:
        rows.append((str(step), "Є хоча б одне легкове авто", "Усі легкові авто — електромобілі.", "EV_ADOPTERS"))
        step += 1
    for rule in rules["rules"]:
        rows.append((f"{step}.{rule['rule_id']}", "1–2 легкові авто (решта клієнтів)",
                     rule_business_text(rule), rule["segment"]))
    table = pd.DataFrame(rows, columns=["Крок", "Кого стосується", "Умова", "SEGMENT"])
    table["Сегмент"] = table["SEGMENT"].map(lambda s: SEGMENT_DISPLAY.get(s, s))
    return table


# =============================================================================
# 11. ЗВІТ: СТАТИСТИКА ПО СЕГМЕНТАХ
# =============================================================================
def segment_statistics(segments, vehicles):
    """
    Чотири таблиці з однаковими рядками сегментів:
    main — основні показники; flags — % клієнтів з ознакою;
    cars — структура легкових авто (тип і пальне, по 100%); fleet — структура всіх ТЗ (100%).
    """
    seg = segments[["user_id", "FINAL_SEGMENT"]]
    order = segment_order(seg["FINAL_SEGMENT"].unique())
    c = segments.copy()
    g = c.groupby("FINAL_SEGMENT")
    total = len(c)

    has_cars = c["PASSENGER_VEHICLE_COUNT"].gt(0) & c["AGE_UNKNOWN"].eq(0)
    all_value = vehicles.groupby("user_id")["MARKET_VALUE_USD"].sum().rename("ALL_VALUE")
    c = c.merge(all_value, on="user_id", how="left")
    g = c.groupby("FINAL_SEGMENT")

    main = pd.DataFrame({
        "Група": [segment_group(s) for s in order],
        "Сегмент": [SEGMENT_DISPLAY.get(s, s) for s in order],
        "Опис": [SEGMENT_DESCRIPTION.get(s, "") for s in order],
        "Клієнтів": g.size().reindex(order).values,
        "% бази": (g.size() / total).reindex(order).values,
        "Кількість ТЗ": g["TOTAL_VEHICLE_COUNT"].sum().reindex(order).values,
        "Кількість легкових": g["PASSENGER_VEHICLE_COUNT"].sum().reindex(order).values,
        "Медіана вартості легкових, USD": c.loc[c["PASSENGER_VEHICLE_COUNT"].gt(0)]
            .groupby("FINAL_SEGMENT")["TOTAL_VEHICLE_VALUE_USD"].median().reindex(order).values,
        "Медіана вартості всіх ТЗ, USD": g["ALL_VALUE"].median().reindex(order).values,
        "Медіана віку легкових, років": c.loc[has_cars.values]
            .groupby("FINAL_SEGMENT")["AVG_VEHICLE_AGE"].median().reindex(order).values,
    }, index=order)

    flag_map = {
        "% Premium": "HAS_PREMIUM", "% SUV": "HAS_SUV", "% EV": "HAS_EV", "% Hybrid": "HAS_HYBRID",
        "% Газ": "HAS_GAS", "% Commercial": "HAS_COMMERCIAL", "% Автобус": "HAS_BUS",
        "% Причіп (будь-який)": "HAS_TRAILER", "% Легкий причіп": "HAS_LIGHT_TRAILER",
        "% Важкий причіп": "HAS_HEAVY_TRAILER", "% Причіп для човна": "HAS_BOAT_TRAILER",
        "% Мото": "HAS_MOTO",
    }
    flags = pd.DataFrame({name: g[col].mean().reindex(order) for name, col in flag_map.items()})
    flags.insert(0, "Сегмент", [SEGMENT_DISPLAY.get(s, s) for s in order])

    v = vehicles.merge(seg, on="user_id", how="left")
    cars = v.loc[v["IS_PASSENGER"].eq(1)].copy()
    cars["TYPE"] = np.select(
        [cars["IS_PREMIUM_BRAND"].eq(1) & cars["IS_SUV"].eq(1), cars["IS_PREMIUM_BRAND"].eq(1),
         cars["IS_SUV"].eq(1)],
        ["Premium SUV", "Premium не-SUV", "Масовий SUV"], default="Масовий не-SUV",
    )
    fuel_names = {"ELECTRIC": "EV", "HYBRID": "Hybrid", "PETROL": "Бензин", "DIESEL": "Дизель",
                  "GAS": "Газ", "UNKNOWN": "Невідомо"}
    cars["FUEL"] = cars["FUEL_CLASS"].map(fuel_names)
    car_type = pd.crosstab(cars["FINAL_SEGMENT"], cars["TYPE"], normalize="index")
    car_type = car_type.reindex(columns=["Premium SUV", "Premium не-SUV", "Масовий SUV", "Масовий не-SUV"], fill_value=0)
    car_fuel = pd.crosstab(cars["FINAL_SEGMENT"], cars["FUEL"], normalize="index")
    car_fuel = car_fuel.reindex(columns=list(fuel_names.values()), fill_value=0)
    car_table = pd.concat([car_type, car_fuel], axis=1).reindex(order)
    car_table.insert(0, "Легкових авто", cars.groupby("FINAL_SEGMENT").size().reindex(order).fillna(0).astype(int))
    car_table.insert(0, "Сегмент", [SEGMENT_DISPLAY.get(s, s) for s in order])

    v["FLEET"] = np.select(
        [v["IS_PASSENGER"].eq(1), v["IS_COMMERCIAL"].eq(1), v["IS_BUS"].eq(1), v["IS_MOTO"].eq(1),
         v["IS_LIGHT_TRAILER"].eq(1), v["IS_HEAVY_TRAILER"].eq(1), v["IS_BOAT_TRAILER"].eq(1)],
        ["Легкові", "Вантажні / спецтехніка", "Автобуси", "Мото", "Причепи легкі", "Причепи важкі",
         "Причепи для човнів"],
        default="Інше",
    )
    fleet_cols = ["Легкові", "Вантажні / спецтехніка", "Автобуси", "Мото", "Причепи легкі", "Причепи важкі",
                  "Причепи для човнів", "Інше"]
    fleet = pd.crosstab(v["FINAL_SEGMENT"], v["FLEET"], normalize="index").reindex(columns=fleet_cols, fill_value=0)
    if fleet["Інше"].sum() == 0:
        fleet = fleet.drop(columns="Інше")
    fleet = fleet.reindex(order)
    fleet.insert(0, "Усього ТЗ", v.groupby("FINAL_SEGMENT").size().reindex(order).values)
    fleet.insert(0, "Сегмент", [SEGMENT_DISPLAY.get(s, s) for s in order])

    return {"main": main, "flags": flags, "cars": car_table, "fleet": fleet}


# =============================================================================
# 12. ЗВІТ: ЗАПИС В EXCEL
# =============================================================================
TITLE_FILL = "1F3864"
HEADER_FILL = "5B8FD4"


def _styles():
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    thin = Side(style="thin", color="B4C6E7")
    return {
        "title_font": Font(bold=True, color="FFFFFF", size=16),
        "subtitle_font": Font(bold=True, color="FFFFFF", size=12),
        "header_font": Font(bold=True, color="FFFFFF", size=11),
        "bold": Font(bold=True),
        "title_fill": PatternFill("solid", fgColor=TITLE_FILL),
        "header_fill": PatternFill("solid", fgColor=HEADER_FILL),
        "note_fill": PatternFill("solid", fgColor="E2EFDA"),
        "border": Border(left=thin, right=thin, top=thin, bottom=thin),
        "center": Alignment(horizontal="center", vertical="center", wrap_text=True),
        "left": Alignment(horizontal="left", vertical="center", wrap_text=True),
        "PatternFill": PatternFill,
    }


def write_title(ws, row, col, text, width, size="title"):
    s = _styles()
    ws.merge_cells(start_row=row, start_column=col, end_row=row, end_column=col + width - 1)
    cell = ws.cell(row=row, column=col, value=text)
    cell.font = s["title_font"] if size == "title" else s["subtitle_font"]
    cell.fill = s["title_fill"]
    cell.alignment = s["center"]
    for c in range(col, col + width):
        ws.cell(row=row, column=c).fill = s["title_fill"]
    ws.row_dimensions[row].height = 30 if size == "title" else 22


def write_table(ws, row, col, df, title=None, formats=None, row_fills=None, left_cols=(), height=None,
                header_height=45):
    """
    Пише DataFrame як оформлену таблицю. formats: {колонка: числовий формат Excel}.
    row_fills: список кольорів (hex) для рядків. Повертає номер рядка після таблиці.
    """
    s = _styles()
    formats = formats or {}
    if title:
        write_title(ws, row, col, title, len(df.columns), size="subtitle")
        row += 1
    for j, name in enumerate(df.columns):
        cell = ws.cell(row=row, column=col + j, value=str(name))
        cell.font = s["header_font"]
        cell.fill = s["header_fill"]
        cell.alignment = s["center"]
        cell.border = s["border"]
    ws.row_dimensions[row].height = header_height
    for i, values in enumerate(df.itertuples(index=False)):
        r = row + 1 + i
        fill = s["PatternFill"]("solid", fgColor=row_fills[i]) if row_fills else None
        for j, value in enumerate(values):
            if isinstance(value, (np.floating, float)) and np.isnan(value):
                value = "—"
            elif isinstance(value, np.generic):
                value = value.item()
            cell = ws.cell(row=r, column=col + j, value=value)
            name = df.columns[j]
            if name in formats and value != "—":
                cell.number_format = formats[name]
            cell.alignment = s["left"] if name in left_cols else s["center"]
            cell.border = s["border"]
            if fill:
                cell.fill = fill
        if height:
            ws.row_dimensions[r].height = height
    return row + 1 + len(df)


def write_note(ws, row, col, lines, width):
    s = _styles()
    for k, line in enumerate(lines):
        r = row + k
        ws.merge_cells(start_row=r, start_column=col, end_row=r, end_column=col + width - 1)
        cell = ws.cell(row=r, column=col, value=line)
        cell.font = Font_bold() if k == 0 else s["bold"]
        cell.alignment = s["center"]
        for c in range(col, col + width):
            ws.cell(row=r, column=c).fill = s["note_fill"]
        ws.row_dimensions[r].height = 22
    return row + len(lines)


def Font_bold():
    from openpyxl.styles import Font
    return Font(bold=True, size=12)


def set_widths(ws, widths):
    from openpyxl.utils import get_column_letter
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
