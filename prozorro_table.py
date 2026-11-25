"""
Зведена таблиця: один рядок = учасник у лоті тендера.

  python prozorro_table.py      (після prozorro_export.py)

Результат: prozorro_data/export/summary.csv
"""
import pandas as pd

D = "prozorro_data/export/"
S = {"buyer_edrpou": str, "edrpou": str, "cpv": str, "lot_id": str}

t = pd.read_csv(D + "tenders.csv", dtype=S)
l = pd.read_csv(D + "lots.csv", dtype=S)
it = pd.read_csv(D + "items.csv", dtype=S)
p = pd.read_csv(D + "participants.csv", dtype=S)
for df in (l, it, p):
    df["lot_id"] = df["lot_id"].fillna("")

# учасники (bidder) + переможці (award active); переможці без пропозиції (переговорні) теж потрапляють
bids = p[p.role == "bidder"][["tender_id", "lot_id", "edrpou", "name", "amount", "date"]] \
    .rename(columns={"amount": "bid_amount", "date": "bid_date"})
wins = p[(p.role == "award") & (p.status == "active")][["tender_id", "lot_id", "edrpou", "name", "amount", "date"]] \
    .rename(columns={"name": "win_name", "amount": "win_amount", "date": "win_date"}) \
    .drop_duplicates(["tender_id", "lot_id", "edrpou"])
wins["is_winner"] = True
x = bids.merge(wins, on=["tender_id", "lot_id", "edrpou"], how="outer")
x["is_winner"] = x["is_winner"].fillna(False).astype(bool)
x["name"] = x["name"].fillna(x["win_name"])

# статус і очікувана сума лоту
lots = l.rename(columns={"status": "lot_status", "value_amount": "lot_value"})[
    ["tender_id", "lot_id", "lot_status", "lot_value"]]
x = x.merge(lots, on=["tender_id", "lot_id"], how="left")

# тип товару: CPV першої позиції лоту + сумарна кількість по лоту
cpv = it.groupby(["tender_id", "lot_id"], as_index=False).agg(
    lot_cpv=("cpv", "first"), lot_cpv_description=("cpv_description", "first"),
    quantity=("quantity", "sum"), unit=("unit", "first"))
x = x.merge(cpv, on=["tender_id", "lot_id"], how="left")

# дані тендера
tt = t.rename(columns={"id": "tender_id", "status": "tender_status", "value_amount": "tender_value"})[
    ["tender_id", "tenderID", "procurementMethodType", "mainProcurementCategory", "tender_status",
     "buyer_edrpou", "buyer_name", "tender_value", "date_created", "cpv", "cpv_description"]]
x = x.merge(tt, on="tender_id", how="left")

x["status"] = x["lot_status"].fillna(x["tender_status"])
x["expected_amount"] = x["lot_value"].fillna(x["tender_value"])
x["cpv"] = x["lot_cpv"].fillna(x["cpv"])
x["cpv_description"] = x["lot_cpv_description"].fillna(x["cpv_description"])

out = x[["tenderID", "lot_id", "procurementMethodType", "mainProcurementCategory",
         "buyer_edrpou", "buyer_name",                                  # 1. організатор
         "edrpou", "name", "is_winner",                                 # 2. учасник / переможець
         "status",                                                      # 3. статус (лоту, або тендера)
         "expected_amount", "bid_amount", "win_amount",                 # 4. суми
         "date_created", "bid_date", "win_date",                        # 5. дати
         "cpv", "cpv_description", "quantity", "unit"]]                 # 6. тип товару
out = out.rename(columns={"edrpou": "participant_edrpou", "name": "participant_name"})
out.to_csv(D + "summary.csv", index=False, encoding="utf-8-sig")
print(f"summary.csv: {len(out)} рядків, {out.tenderID.nunique()} тендерів")