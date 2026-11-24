# freq

Оновлюємо сегментацію за новими вимогами бізнесу.
KMeans і мапінг кластерів НЕ перенавчаємо.

=== 1. РІВЕНЬ АВТО ===
a) FUEL_CLASS (5 класів): ELECTRIC / PETROL / DIESEL / HYBRID / UNKNOWN.
   Будь-який мікс пального, включно з бензин/газ (ГБО) → HYBRID.
   Покажи мапінг усіх значень FUEL_STD → FUEL_CLASS з к-стю.
   IS_HYBRID = 1 для HYBRID. IS_EV лишається як було.
b) Причепи: усі kind-причепи → IS_TRAILER = 1.
   TRAILER_TYPE: LIGHT / BOAT / HEAVY.
   LIGHT vs HEAVY — за total_weight (≤ 3.5 т → LIGHT, > 3.5 т → HEAVY)
   + назва kind (напівпричепи, автовози, для фури → HEAVY;
   для човнів → BOAT). Покажи таблицю kind × TRAILER_TYPE зі спірними випадками.
c) IS_BUS = 1 для автобусів. IS_COMMERCIAL = 1 для вантажних і спецтехніки
   (автобуси окремо, не в IS_COMMERCIAL).

=== 2. РІВЕНЬ КЛІЄНТА ===
Нові ознаки:
- HYBRID_SHARE (серед легкових), HAS_HYBRID
- HAS_TRAILER, TRAILER_COUNT, TRAILER_SHARE (від усіх ТЗ клієнта),
  HAS_LIGHT_TRAILER, HAS_BOAT_TRAILER, HAS_HEAVY_TRAILER, HEAVY_TRAILER_COUNT
- HAS_BUS, BUS_COUNT, HAS_COMMERCIAL, COMMERCIAL_COUNT
- BUSINESS_VEHICLE_COUNT = COMMERCIAL_COUNT + BUS_COUNT + HEAVY_TRAILER_COUNT
Ознаки кластеризації (6 SHORT_FEATURES) не змінюються.

=== 3. НОВА ІЄРАРХІЯ (зверху вниз) ===
1. Легкових = 0 (популяція B):
   - BUSINESS_VEHICLE_COUNT ≥ 1 → COMMERCIAL_VEHICLE_OWNERS
   - HAS_MOTO → MOTORCYCLE_OWNERS
   - HAS_LIGHT_TRAILER або HAS_BOAT_TRAILER → TRAILER_ONLY_OWNERS
   - інакше → OTHER_NON_PASSENGER
2. Легкових ≥ 1 і BUSINESS_VEHICLE_COUNT ≥ N → BUSINESS_VEHICLE_OWNERS
   (покажи розміри для N = 1 / 2 / 3, за замовчуванням N = 1)
3. Легкових ≥ 3 → MULTI_CAR_3PLUS
   (одна група; сегмент FLEET_CORPORATE прибираємо повністю)
4. TIER5_SHARE > 0 → LUXURY
5. HAS_MODERN_PREMIUM → MODERN_PREMIUM
6. Решта (1–2 легковики) → поточне дерево depth 5 без змін.
   Перерахуй fidelity на новій популяції дерева. Якщо < 90% —
   перенавчи дерево тільки на клієнтах з 1–2 легковиками і покажи обидва варіанти.
HAS_LUXURY_VEHICLE і всі прапорці зберігаються для всіх клієнтів.

=== 4. ПЕРЕВІРКИ ===
- таблиця розмірів усіх сегментів: було → стало, к-сть і %;
- матриця переходів старий сегмент × новий;
- скільки Luxury і Modern Premium пішли в MULTI_CAR_3PLUS / BUSINESS;
- профілі нових сегментів (BUSINESS, MULTI_CAR_3PLUS): медіани к-сті
  легкових, BUSINESS_VEHICLE_COUNT, вартості, віку; % premium/SUV/EV/hybrid;
- розподіл HAS_HYBRID і HAS_TRAILER по сегментах;
- кожен клієнт має рівно 1 сегмент, без пропусків і дублікатів.

=== 5. ОНОВИ ВСЕ ===
utils.py, pipeline.py, score.py (перевірка 100% збігу), SQL,
текст правил, Excel-звіт (сегменти, правила, прапорці, аркуш "Дані").
Прибери FLEET_CORPORATE з усіх файлів і описів.
