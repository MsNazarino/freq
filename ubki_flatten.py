import ast
import json
import sys
from pathlib import Path

import pandas as pd

# ================== НАЛАШТУВАННЯ ==================
SRC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "uBKI_v3"
N_BATCHES = 9
OUT_DIR = SRC_DIR

# префікс колонок | скільки осіб максимум розкривати в колонки
CONFIG = {
    "Уповноважені особи": ("Уповн", 20),
    "Засновники":         ("Засн",  20),
    "Бенефіціари":        ("Бенеф", 20),
}
DEFAULT_MAX = 20  # для JSON-колонок, яких немає в CONFIG
# ==================================================


def parse_list(v):
    """'[{...}, {...}]' -> list[dict]; порожнє -> []; не вдалося -> None"""
    s = "" if v is None else str(v).strip()
    if not s:
        return []
    data = None
    for loader in (json.loads, ast.literal_eval):
        try:
            data = loader(s)
            break
        except Exception:
            pass
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return None
    return [x if isinstance(x, dict) else {"значення": x} for x in data]


def is_json_col(series):
    s = series.astype(str).str.strip()
    s = s[s != ""]
    return len(s) > 0 and s.str.startswith("[").mean() > 0.9


# ---------- 1. Читаємо й об'єднуємо батчі ----------
frames = []
for i in range(1, N_BATCHES + 1):
    path = SRC_DIR / f"ubki_batch_{i}.csv"
    if not path.exists():
        print(f"[!] немає файлу {path.name}")
        continue
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.insert(0, "_batch", str(i))
    frames.append(df)
    print(f"{path.name}: {len(df)} рядків")

df = pd.concat(frames, ignore_index=True)
df["ЄДРПОУ"] = df["ЄДРПОУ"].str.strip().str.zfill(8)
print(f"\nРазом: {len(df)} рядків, унікальних ЄДРПОУ: {df['ЄДРПОУ'].nunique()}")

# ---------- 2. Розкриваємо JSON-колонки ----------
json_cols = [c for c in df.columns if is_json_col(df[c])]
print("JSON-колонки:", json_cols, "\n")

for col in json_cols:
    pref, max_items = CONFIG.get(col, (col, DEFAULT_MAX))
    parsed = [parse_list(v) for v in df[col].tolist()]

    bad = [p is None for p in parsed]
    if any(bad):
        print(f"[!] {col}: не розпарсилось {sum(bad)} рядків -> сирий текст у '{pref}_raw'")
        df[f"{pref}_raw"] = [v if b else "" for v, b in zip(df[col].tolist(), bad)]
        parsed = [p if p is not None else [] for p in parsed]

    keys = []
    for lst in parsed:
        for d in lst:
            for k in d:
                if k not in keys:
                    keys.append(k)

    counts = [len(lst) for lst in parsed]
    max_n = max(counts) if counts else 0
    cap = min(max_n, max_items)
    over = sum(c > cap for c in counts)

    # --- перші cap осіб в окремі колонки ---
    new = {f"{pref}_кількість": counts}
    for n in range(cap):
        for k in keys:
            new[f"{pref}_{n + 1}_{k}"] = [
                lst[n].get(k, "") if n < len(lst) else "" for lst in parsed
            ]

    pos = df.columns.get_loc(col)
    df = pd.concat(
        [df.iloc[:, :pos], pd.DataFrame(new, index=df.index), df.iloc[:, pos + 1:]],
        axis=1,
    )
    print(
        f"{col}: ключі {keys}, макс. у рядку = {max_n}, в основну таблицю {cap} "
        f"({1 + cap * len(keys)} колонок), обрізано у {over} компаній"
    )

# ---------- 3. Зберігаємо ----------
print(f"\nПідсумок: {df.shape[0]} рядків x {df.shape[1]} колонок")
df.to_csv(OUT_DIR / "ubki_all_flat.csv", index=False, encoding="utf-8-sig")
print("CSV  ->", OUT_DIR / "ubki_all_flat.csv")

if len(df) < 1_048_576 and df.shape[1] <= 16_384:
    try:
        import xlsxwriter  # noqa: F401  (швидше за openpyxl)
        engine = "xlsxwriter"
    except ImportError:
        engine = "openpyxl"
    print(f"Пишу XLSX ({engine}), це може зайняти кілька хвилин...")
    df.to_excel(OUT_DIR / "ubki_all_flat.xlsx", index=False, engine=engine)
    print("XLSX ->", OUT_DIR / "ubki_all_flat.xlsx")
else:
    print("[!] Таблиця більша за ліміти Excel, збережено лише CSV")
