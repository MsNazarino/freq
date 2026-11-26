import ast
import json
import sys
from pathlib import Path

import pandas as pd

# ================== НАЛАШТУВАННЯ ==================
SRC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r".\uBKI_v3")
N_BATCHES = 9
OUT_CSV = SRC_DIR / "ubki_all_flat.csv"
OUT_XLSX = SRC_DIR / "ubki_all_flat.xlsx"

# Короткі префікси для розкритих колонок (решта JSON-колонок знайдеться автоматично)
PREFIX = {
    "Уповноважені особи": "Уповн",
    "Засновники": "Засн",
    "Бенефіціари": "Бенеф",
}
# ==================================================


def parse_list(v):
    """'[{...}, {...}]' -> list[dict]; порожнє -> []; не вдалося -> None"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    s = str(v).strip()
    if not s:
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            data = loader(s)
            break
        except Exception:
            data = None
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return None
    return [x if isinstance(x, dict) else {"значення": x} for x in data]


def is_json_col(series):
    s = series.dropna().astype(str).str.strip()
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
    df.insert(0, "_batch", i)
    frames.append(df)
    print(f"{path.name}: {len(df)} рядків")

df = pd.concat(frames, ignore_index=True)
df["ЄДРПОУ"] = df["ЄДРПОУ"].str.strip().str.zfill(8)
print(f"\nРазом: {len(df)} рядків, унікальних ЄДРПОУ: {df['ЄДРПОУ'].nunique()}")
if "_parser_status" in df:
    print("Статуси парсера:", df["_parser_status"].value_counts().to_dict())

# ---------- 2. Розкриваємо JSON-колонки ----------
json_cols = [c for c in df.columns if is_json_col(df[c])]
print("\nJSON-колонки:", json_cols)

for col in json_cols:
    pref = PREFIX.get(col, col)
    parsed = df[col].map(parse_list)

    bad = parsed.isna()
    if bad.any():
        print(f"[!] {col}: не розпарсилось {bad.sum()} рядків -> збережено в '{pref}_raw'")
        df[f"{pref}_raw"] = df[col].where(bad, "")
        parsed = parsed.map(lambda x: x if isinstance(x, list) else [])

    # ключі в порядку першої появи
    keys = []
    for lst in parsed:
        for d in lst:
            for k in d:
                if k not in keys:
                    keys.append(k)

    max_n = parsed.map(len).max() if len(parsed) else 0
    print(f"{col}: ключі {keys}, макс. записів у рядку = {max_n}")

    new = {f"{pref}_кількість": parsed.map(len)}
    for n in range(max_n):
        for k in keys:
            new[f"{pref}_{n + 1}_{k}"] = parsed.map(
                lambda lst, n=n, k=k: lst[n].get(k, "") if n < len(lst) else ""
            )

    pos = df.columns.get_loc(col)
    left, right = df.iloc[:, :pos], df.iloc[:, pos + 1:]
    df = pd.concat([left, pd.DataFrame(new, index=df.index), right], axis=1)

# ---------- 3. Зберігаємо ----------
print(f"\nПідсумок: {df.shape[0]} рядків x {df.shape[1]} колонок")
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("CSV  ->", OUT_CSV)

if df.shape[1] <= 16384:
    df.to_excel(OUT_XLSX, index=False)  # ЄДРПОУ лишається текстом (нулі не губляться)
    print("XLSX ->", OUT_XLSX)
else:
    print("[!] Забагато колонок для Excel, збережено лише CSV")
