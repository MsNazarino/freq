"""
Експорт зібраних тендерів у CSV (остання версія кожного тендера).

  python prozorro_export.py

Результат: prozorro_data/export/tenders.csv, lots.csv, items.csv, participants.csv
"""
import csv, glob, json, os, sqlite3, zlib

CHUNK = 1 << 16
GZ_MAGIC = b"\x1f\x8b\x08"

OUT = "prozorro_data"
RAW = os.path.join(OUT, "raw")
EXP = os.path.join(OUT, "export")


def read_lines(path):
    """Читає .jsonl.gz блок за блоком. Битий блок (після аварійної зупинки)
    пропускає і йде до наступного; недописаний хвіст (збір ще йде) — ігнорує."""
    with open(path, "rb") as f:
        data = f.read()
    mv, n, pos, bad = memoryview(data), len(data), 0, 0
    while pos < n:
        d, out, p = zlib.decompressobj(wbits=31), [], pos
        try:
            while not d.eof and p < n:
                block = mv[p:p + CHUNK]
                out.append(d.decompress(block))
                p += len(block)
        except zlib.error:
            bad += 1
            nxt = data.find(GZ_MAGIC, pos + 1)
            if nxt == -1:
                break
            pos = nxt
            continue
        if not d.eof:          # недописаний останній блок
            break
        for line in b"".join(out).decode("utf-8", "replace").splitlines():
            yield line
        pos = p - len(d.unused_data)
    if bad:
        print(f"  {os.path.basename(path)}: пропущено битих блоків — {bad}")


def org(o):
    o = o or {}
    i = o.get("identifier") or {}
    return i.get("id"), o.get("name") or i.get("legalName")


def money(v):
    v = v or {}
    return v.get("amount"), v.get("currency")


def main():
    os.makedirs(EXP, exist_ok=True)
    db_path = os.path.join(EXP, "_dedup.sqlite")
    if os.path.exists(db_path):
        os.remove(db_path)
    db = sqlite3.connect(db_path)
    db.execute("CREATE TABLE raw(id TEXT PRIMARY KEY, dm TEXT, data BLOB)")

    # 1. Дедуплікація: на кожен id лишається версія з найбільшим dateModified
    for path in sorted(glob.glob(os.path.join(RAW, "*.jsonl.gz"))):
        print("читаю", os.path.basename(path))
        for line in read_lines(path):
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            db.execute("""INSERT INTO raw VALUES (?,?,?)
                          ON CONFLICT(id) DO UPDATE SET dm=excluded.dm, data=excluded.data
                          WHERE excluded.dm >= raw.dm""",
                       (t["id"], t.get("dateModified", ""), zlib.compress(line.encode("utf-8"))))
        db.commit()

    # 2. Розкладання в плоскі таблиці
    files = {n: open(os.path.join(EXP, f"{n}.csv"), "w", newline="", encoding="utf-8-sig")
             for n in ("tenders", "lots", "items", "participants")}
    w = {n: csv.writer(f) for n, f in files.items()}
    w["tenders"].writerow(["id", "tenderID", "procurementMethodType", "mainProcurementCategory", "status",
                           "title", "cpv", "cpv_description", "value_amount", "value_currency",
                           "buyer_edrpou", "buyer_name", "buyer_kind", "buyer_region",
                           "date_created", "dateModified", "tender_start", "tender_end", "n_lots", "n_bids"])
    w["lots"].writerow(["tender_id", "lot_id", "title", "status", "value_amount", "value_currency"])
    w["items"].writerow(["tender_id", "lot_id", "cpv", "cpv_description", "description",
                         "quantity", "unit"])
    w["participants"].writerow(["tender_id", "lot_id", "role", "edrpou", "name",
                                "amount", "currency", "status", "date"])

    n = 0
    for (blob,) in db.execute("SELECT data FROM raw"):
        t = json.loads(zlib.decompress(blob))
        tid = t["id"]
        pe = t.get("procuringEntity") or {}
        cls = ((t.get("items") or [{}])[0].get("classification") or {})
        tp = t.get("tenderPeriod") or {}
        bids = t.get("bids") or []
        lots = t.get("lots") or []

        w["tenders"].writerow([
            tid, t.get("tenderID"), t.get("procurementMethodType"), t.get("mainProcurementCategory"),
            t.get("status"), t.get("title"), cls.get("id"), cls.get("description"),
            *money(t.get("value")), *org(pe), pe.get("kind"), (pe.get("address") or {}).get("region"),
            t.get("dateCreated") or t.get("date"), t.get("dateModified"),
            tp.get("startDate"), tp.get("endDate"), len(lots), len(bids)])

        for lot in lots:
            w["lots"].writerow([tid, lot.get("id"), lot.get("title"), lot.get("status"), *money(lot.get("value"))])

        for it in t.get("items") or []:
            c = it.get("classification") or {}
            w["items"].writerow([tid, it.get("relatedLot"), c.get("id"), c.get("description"),
                                 it.get("description"), it.get("quantity"), (it.get("unit") or {}).get("name")])

        rows = [[tid, None, "buyer", *org(pe), *money(t.get("value")), t.get("status"),
                 t.get("dateCreated") or t.get("date")]]
        for b in bids:
            for tr in b.get("tenderers") or []:
                if b.get("lotValues"):
                    for lv in b["lotValues"]:
                        rows.append([tid, lv.get("relatedLot"), "bidder", *org(tr),
                                     *money(lv.get("value")), lv.get("status") or b.get("status"), b.get("date")])
                else:
                    rows.append([tid, None, "bidder", *org(tr), *money(b.get("value")), b.get("status"),
                                 b.get("date")])
        for a in t.get("awards") or []:
            for s in a.get("suppliers") or []:
                rows.append([tid, a.get("lotID"), "award", *org(s), *money(a.get("value")), a.get("status"),
                             a.get("date")])
        for c in t.get("contracts") or []:
            for s in c.get("suppliers") or []:
                rows.append([tid, None, "contract", *org(s), *money(c.get("value")), c.get("status"),
                             c.get("dateSigned") or c.get("date")])
        w["participants"].writerows(rows)

        n += 1
        if n % 100_000 == 0:
            print(f"  експортовано {n}")

    for f in files.values():
        f.close()
    db.close()
    os.remove(db_path)
    print(f"Готово: {n} унікальних тендерів -> {EXP}")


if __name__ == "__main__":
    main()