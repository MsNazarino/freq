# %%
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# %%
# ============================================================
# НАЛАШТУВАННЯ
# ============================================================

INPUT_CSV = r"C:\repo\data-science\Nazar Hnatyshyn\CORP\Add ben info and cont to corp sample\corp_client_without_micro.csv"

# Колонка у вхідному CSV, з якої беруться коди для парсингу
CODE_COLUMN = "IDENTIFYCODE"

# Папка для результатів (створиться автоматично)
OUTPUT_DIR = Path(r"C:\repo\data-science\Nazar Hnatyshyn\CORP\Add ben info and cont to corp sample\results")
OUTPUT_FILENAME = "result_corp_client_without_micro.csv"
OUTPUT_CSV = OUTPUT_DIR / OUTPUT_FILENAME

# Якщо створити файл з такою назвою в OUTPUT_DIR, парсер акуратно
# зупиниться після поточної компанії і збереже прогрес.
STOP_FILE = OUTPUT_DIR / "STOP"

BASE_URL = (
    "https://youcontrol.com.ua/"
    "catalog/company_details/{identifycode}/"
)

REQUEST_TIMEOUT = 30     # максимальний час очікування відповіді, секунд
DELAY_MIN = 2            # мінімальна пауза між компаніями, секунд
DELAY_MAX = 5            # максимальна пауза між компаніями, секунд
SAVE_EVERY = 10          # збереження після кожних 10 оброблених компаній

SAVE_RETRIES = 10        # скільки разів пробувати зберегти, якщо файл зайнятий
SAVE_RETRY_DELAY = 3     # пауза між спробами збереження, секунд

BENEFICIARY_LABEL = (
    "Інформація про кінцевого "
    "бенефіціарного власника"
)

RESULT_COLUMNS = [
    "IDENTIFYCODE",
    "BENEFICIARY_NAME",
    "BENEFICIARY_COUNTRY",
    "BENEFICIARY_ADDRESS",
    "OWNERSHIP_TYPE",
    "BENEFICIARY_SHARE",
    "STATUS",
]

# Статуси, при яких сервер обмежує доступ: парсер зупиняється,
# а ці компанії не вважаються обробленими.
STOP_STATUSES = {"HTTP_403", "HTTP_429"}

# Тимчасові помилки: такі компанії будуть повторно спарсені
# при наступному запуску.
RETRY_STATUSES = {"HTTP_403", "HTTP_429", "REQUEST_TIMEOUT", "EMPTY_RESPONSE"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,*/*;q=0.8"
    ),
    "Accept-Language": "uk-UA,uk;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
}

# %%
# ============================================================
# ДОПОМІЖНІ ФУНКЦІЇ
# ============================================================

def clean_text(value):
    """
    Прибирає зайві пробіли, переноси рядків
    і нерозривні пробіли.
    """
    if value is None:
        return None

    value = str(value).replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value).strip()

    return value or None


def normalize_identifycode(value):
    """
    Нормалізує ЄДРПОУ:
    - прибирає пробіли;
    - прибирає .0;
    - залишає тільки цифри;
    - доповнює нулями зліва до 8 символів.
    """
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    value = re.sub(r"\D", "", value)

    if not value:
        return None

    return value.zfill(8)


def is_retry_status(status):
    """
    True, якщо компанію треба спарсити повторно
    (тимчасова помилка або обмеження доступу).
    """
    if pd.isna(status):
        return True

    status = str(status)

    return (
        status in RETRY_STATUSES
        or status.startswith("REQUEST_ERROR")
        or re.fullmatch(r"HTTP_5\d\d", status) is not None
    )


def empty_result(identifycode, status):
    """
    Формує технічний рядок для компанії, яку не вдалося
    успішно розпарсити або в якої немає бенефіціарів.

    Завдяки цьому ЄДРПОУ буде в результаті та не запуститься
    повторно після звичайного перезапуску.
    """
    return {
        "IDENTIFYCODE": identifycode,
        "BENEFICIARY_NAME": None,
        "BENEFICIARY_COUNTRY": None,
        "BENEFICIARY_ADDRESS": None,
        "OWNERSHIP_TYPE": None,
        "BENEFICIARY_SHARE": None,
        "STATUS": status,
    }


def parse_share(value):
    """
    Перетворює текстову частку у float.

    Приклади:
    33.3  -> 33.3
    33,3  -> 33.3
    33.3% -> 33.3
    """
    if not value:
        return None

    match = re.search(r"\d+(?:[.,]\d+)?", str(value))

    if not match:
        return None

    return float(match.group(0).replace(",", "."))


def extract_labeled_value(block, label):
    """
    Дістає значення після заданого підпису.

    Наприклад:

    Тип бенефіціарного володіння:
    Прямий вирішальний вплив
    """
    label_normalized = clean_text(label)

    for paragraph in block.find_all("p"):
        paragraph_text = clean_text(paragraph.get_text(" ", strip=True))

        if not paragraph_text:
            continue

        if label_normalized.lower() not in paragraph_text.lower():
            continue

        value = re.sub(
            rf"^.*?{re.escape(label_normalized)}\s*:?\s*",
            "",
            paragraph_text,
            flags=re.IGNORECASE,
        )

        return clean_text(value)

    return None


def extract_beneficiary_address(info_group, name_tag):
    """
    Дістає адресу з першого абзацу бенефіціара
    та прибирає з нього ПІБ.
    """
    paragraph = name_tag.find_parent("p")

    if not paragraph:
        return None

    full_text = clean_text(paragraph.get_text(" ", strip=True))
    beneficiary_name = clean_text(name_tag.get_text(" ", strip=True))

    if not full_text:
        return None

    if beneficiary_name:
        full_text = full_text.replace(beneficiary_name, "", 1)

    return clean_text(full_text.strip(" ,;:-"))


def extract_country(address):
    """
    Дістає країну як першу частину адреси.
    """
    if not address:
        return None

    return clean_text(address.split(",", 1)[0])

# %%
# ============================================================
# ЧИТАННЯ ВХІДНОГО ФАЙЛУ
# ============================================================

def read_input_codes(path, code_column):
    """
    Читає, нормалізує та дедуплікує коди з колонки code_column.
    """
    input_df = pd.read_csv(
        path,
        dtype={code_column: "string"},
        encoding="utf-8-sig",
    )

    if code_column not in input_df.columns:
        raise ValueError(
            f"У вхідному CSV відсутня колонка {code_column}. "
            f"Наявні колонки: {list(input_df.columns)}"
        )

    identifycodes = (
        input_df[code_column]
        .dropna()
        .map(normalize_identifycode)
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    return identifycodes

# %%
# ============================================================
# ЗАВАНТАЖЕННЯ ПОПЕРЕДНІХ РЕЗУЛЬТАТІВ
# ============================================================

def load_existing_results(path):
    """
    Завантажує попередній результат.

    Рядки з тимчасовими помилками (403/429/timeout/5xx тощо)
    відкидаються, щоб ці компанії спарсились повторно.

    Якщо файл існує, але його не вдалося прочитати, функція
    падає з помилкою: інакше наступне збереження перезаписало б
    весь попередній прогрес порожнім результатом.

    Повертає:
    - список усіх збережених рядків;
    - множину оброблених IDENTIFYCODE.
    """
    path = Path(path)

    if not path.exists():
        return [], set()

    try:
        result_df = pd.read_csv(
            path,
            dtype={"IDENTIFYCODE": "string"},
            encoding="utf-8-sig",
        )
    except pd.errors.EmptyDataError:
        return [], set()

    if "IDENTIFYCODE" not in result_df.columns:
        raise ValueError(
            f"У файлі {path} немає колонки IDENTIFYCODE. "
            "Перевір файл або вкажи інший OUTPUT_CSV."
        )

    result_df = result_df.reindex(columns=RESULT_COLUMNS)

    result_df["IDENTIFYCODE"] = (
        result_df["IDENTIFYCODE"].map(normalize_identifycode)
    )
    result_df = result_df[result_df["IDENTIFYCODE"].notna()]

    retry_mask = result_df["STATUS"].map(is_retry_status)
    retry_count = result_df.loc[retry_mask, "IDENTIFYCODE"].nunique()

    if retry_count:
        print(
            f"Компаній з тимчасовими помилками, "
            f"які буде спарсено повторно: {retry_count}"
        )

    result_df = result_df[~retry_mask].copy()
    result_df = result_df.astype(object).where(result_df.notna(), None)

    results = result_df.to_dict(orient="records")
    processed_codes = set(result_df["IDENTIFYCODE"].tolist())

    return results, processed_codes

# %%
# ============================================================
# ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТУ
# ============================================================

def write_csv_atomic(df, output_path):
    """
    Атомарно записує CSV: спочатку тимчасовий файл,
    потім заміна основного.

    На Windows заміна падає з PermissionError, якщо файл
    відкритий в Excel або його в цей момент сканує антивірус /
    OneDrive. Тому робимо кілька спроб з паузою.
    """
    output_path = Path(output_path)
    temporary_path = output_path.with_name(output_path.name + ".tmp")

    last_error = None

    for attempt in range(1, SAVE_RETRIES + 1):
        try:
            df.to_csv(temporary_path, index=False, encoding="utf-8-sig")
            os.replace(temporary_path, output_path)
            return
        except PermissionError as exc:
            last_error = exc
            print(
                f"Файл зайнятий ({output_path.name}): спроба "
                f"{attempt}/{SAVE_RETRIES}. Закрий його в Excel, якщо відкритий."
            )
            time.sleep(SAVE_RETRY_DELAY)

    raise last_error


def save_results(results, output_path, final=False):
    """
    Зберігає результати в CSV і повертає DataFrame.

    Ніколи не падає через зайнятий файл:
    - проміжне збереження: просто попереджає, дані лишаються
      в пам'яті і збережуться наступного разу;
    - фінальне збереження: пише резервну копію з датою в назві.
    """
    result_df = pd.DataFrame(results, columns=RESULT_COLUMNS)

    if not result_df.empty:
        result_df["IDENTIFYCODE"] = (
            result_df["IDENTIFYCODE"].map(normalize_identifycode)
        )

        result_df = result_df.drop_duplicates(
            subset=[
                "IDENTIFYCODE",
                "BENEFICIARY_NAME",
                "OWNERSHIP_TYPE",
                "BENEFICIARY_SHARE",
                "STATUS",
            ],
            keep="last",
        )

    try:
        write_csv_atomic(result_df, output_path)
        return result_df

    except OSError as exc:
        print(f"Не вдалося зберегти {output_path}: {type(exc).__name__}: {exc}")

    if not final:
        print("Прогрес лишається в пам'яті, спробую на наступному збереженні.")
        return result_df

    output_path = Path(output_path)
    backup_path = output_path.with_name(
        f"{output_path.stem}_backup_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )

    try:
        result_df.to_csv(backup_path, index=False, encoding="utf-8-sig")
        print(
            f"Результат збережено в резервний файл: {backup_path}\n"
            f"Перейменуй його на {output_path.name}, щоб продовжити з нього."
        )
    except OSError as exc:
        print(f"Не вдалося зберегти навіть резервну копію: {exc}")

    return result_df

# %%
# ============================================================
# ПАРСИНГ HTML ОДНІЄЇ КОМПАНІЇ
# ============================================================

def parse_company_page(html, requested_identifycode):
    """
    Дістає з HTML таблицю кінцевих бенефіціарів.

    Повертає список словників:
    один словник на одного бенефіціара.
    """
    soup = BeautifulSoup(html, "html.parser")

    # --------------------------------------------------------
    # Перевірка ЄДРПОУ на сторінці
    # --------------------------------------------------------

    code_tag = soup.select_one("#code-copy")

    page_identifycode = None

    if code_tag:
        page_identifycode = normalize_identifycode(
            code_tag.get_text(" ", strip=True)
        )

    identifycode = page_identifycode or requested_identifycode

    if page_identifycode and page_identifycode != requested_identifycode:
        return [
            empty_result(
                requested_identifycode,
                f"CODE_MISMATCH: {page_identifycode}",
            )
        ]

    # --------------------------------------------------------
    # Загальна секція учасників і бенефіціарів
    # --------------------------------------------------------

    ownership_section = soup.select_one("#catalog-company-beneficiary")

    if not ownership_section:
        return [empty_result(identifycode, "SECTION_NOT_FOUND")]

    # --------------------------------------------------------
    # Пошук конкретного рядка кінцевих бенефіціарів
    # --------------------------------------------------------

    beneficiary_row = None

    for row in ownership_section.select(".seo-table-row"):
        label_column = row.select_one(".seo-table-col-1")

        if not label_column:
            continue

        label_text = clean_text(label_column.get_text(" ", strip=True))

        if label_text and BENEFICIARY_LABEL.lower() in label_text.lower():
            beneficiary_row = row
            break

    if not beneficiary_row:
        return [empty_result(identifycode, "BENEFICIARY_ROW_NOT_FOUND")]

    value_column = beneficiary_row.select_one(".seo-table-col-2")

    if not value_column:
        return [empty_result(identifycode, "BENEFICIARY_VALUE_NOT_FOUND")]

    # --------------------------------------------------------
    # Парсинг окремих бенефіціарів
    # --------------------------------------------------------

    beneficiaries = []

    info_groups = value_column.select(":scope > .info-group")

    # Резервний пошук на випадок зміни вкладеності HTML.
    if not info_groups:
        info_groups = value_column.select(".info-group")

    for info_group in info_groups:
        name_tag = info_group.select_one(".name")

        if not name_tag:
            continue

        beneficiary_name = clean_text(name_tag.get_text(" ", strip=True))

        if not beneficiary_name:
            continue

        beneficiary_address = extract_beneficiary_address(
            info_group=info_group,
            name_tag=name_tag,
        )

        ownership_type = extract_labeled_value(
            block=info_group,
            label="Тип бенефіціарного володіння",
        )

        share_text = extract_labeled_value(
            block=info_group,
            label=(
                "Відсоток частки статутного капіталу "
                "або відсоток права голосу"
            ),
        )

        beneficiaries.append(
            {
                "IDENTIFYCODE": identifycode,
                "BENEFICIARY_NAME": beneficiary_name,
                "BENEFICIARY_COUNTRY": extract_country(beneficiary_address),
                "BENEFICIARY_ADDRESS": beneficiary_address,
                "OWNERSHIP_TYPE": ownership_type,
                "BENEFICIARY_SHARE": parse_share(share_text),
                "STATUS": "DONE",
            }
        )

    # --------------------------------------------------------
    # Рядок знайдений, але структурованих осіб немає
    # --------------------------------------------------------

    if not beneficiaries:
        section_text = clean_text(value_column.get_text(" ", strip=True))

        if section_text:
            status = f"NO_STRUCTURED_BENEFICIARIES: {section_text}"
        else:
            status = "NO_BENEFICIARIES"

        return [empty_result(identifycode, status)]

    return beneficiaries

# %%
# ============================================================
# ЗАВАНТАЖЕННЯ ОДНІЄЇ СТОРІНКИ
# ============================================================

def fetch_and_parse_company(session, identifycode):
    """
    Завантажує сторінку компанії та запускає парсинг.
    """
    url = BASE_URL.format(identifycode=identifycode)

    try:
        response = session.get(
            url,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

    except requests.Timeout:
        return [empty_result(identifycode, "REQUEST_TIMEOUT")]

    except requests.RequestException as exc:
        return [
            empty_result(
                identifycode,
                f"REQUEST_ERROR: {type(exc).__name__}",
            )
        ]

    if response.status_code == 404:
        return [empty_result(identifycode, "NOT_FOUND")]

    if response.status_code != 200:
        return [empty_result(identifycode, f"HTTP_{response.status_code}")]

    if not response.text.strip():
        return [empty_result(identifycode, "EMPTY_RESPONSE")]

    # Помилка парсингу однієї дивної сторінки не має валити
    # весь запуск.
    try:
        return parse_company_page(
            html=response.text,
            requested_identifycode=identifycode,
        )
    except Exception as exc:
        return [
            empty_result(
                identifycode,
                f"PARSE_ERROR: {type(exc).__name__}: {exc}",
            )
        ]

# %%
# ============================================================
# ОСНОВНИЙ ЗАПУСК
# ============================================================

def print_summary(saved_df, processed_in_current_run):
    print(f"\nФайл результату: {OUTPUT_CSV}")
    print(f"Рядків у результаті: {len(saved_df)}")
    print(
        "Компаній оброблено в поточному запуску: "
        f"{processed_in_current_run}"
    )

    if saved_df.empty:
        return

    print("\nСтатистика статусів:")
    print(saved_df["STATUS"].value_counts(dropna=False))

    done_df = saved_df[saved_df["STATUS"] == "DONE"]

    if not done_df.empty:
        share_statistics = (
            done_df.groupby("IDENTIFYCODE")
            .agg(
                BENEFICIARIES_COUNT=("BENEFICIARY_NAME", "count"),
                BENEFICIARY_SHARE_SUM=("BENEFICIARY_SHARE", "sum"),
            )
        )

        print("\nСтатистика успішно розпарсених компаній:")
        print(share_statistics.describe())


def main():
    print("Запуск YouControl parser")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if STOP_FILE.exists():
        STOP_FILE.unlink()

    print(f"Файл результату: {OUTPUT_CSV.resolve()}")
    print(f"Колонка з кодами: {CODE_COLUMN}")

    identifycodes = read_input_codes(INPUT_CSV, CODE_COLUMN)

    results, processed_codes = load_existing_results(OUTPUT_CSV)

    codes_to_process = [
        identifycode
        for identifycode in identifycodes
        if identifycode not in processed_codes
    ]

    print(f"Усього у вхідному файлі: {len(identifycodes)}")
    print(f"Уже оброблено: {len(processed_codes)}")
    print(f"Залишилося обробити: {len(codes_to_process)}")
    print(f"Поточна кількість рядків результату: {len(results)}")

    if not codes_to_process:
        print("Усі компанії вже оброблені")
        return

    print(
        "\nЗупинити: кнопка Interrupt у Jupyter (або Ctrl+C), "
        f"чи створити файл {STOP_FILE}. Прогрес збережеться."
    )

    processed_in_current_run = 0

    try:
        with requests.Session() as session:
            session.headers.update(HEADERS)

            for number, identifycode in enumerate(codes_to_process, start=1):
                print(
                    f"\n[{number}/{len(codes_to_process)}] "
                    f"Парсинг {identifycode}"
                )

                company_results = fetch_and_parse_company(
                    session=session,
                    identifycode=identifycode,
                )

                statuses = sorted(
                    {
                        str(row.get("STATUS", "UNKNOWN"))
                        for row in company_results
                    }
                )

                # Якщо сервер сигналізує про обмеження, не записуємо
                # цю компанію (щоб наступного разу її спарсити)
                # і завершуємо запуск.
                if any(status in STOP_STATUSES for status in statuses):
                    print(
                        f"Отримано обмеження доступу ({', '.join(statuses)}). "
                        "Парсер буде зупинено."
                    )
                    break

                results.extend(company_results)
                processed_in_current_run += 1

                beneficiaries_count = sum(
                    1 for row in company_results if row.get("BENEFICIARY_NAME")
                )

                print(f"Статус: {', '.join(statuses)}")
                print(f"Бенефіціарів: {beneficiaries_count}")

                # Проміжне збереження.
                if processed_in_current_run % SAVE_EVERY == 0:
                    saved_df = save_results(results, OUTPUT_CSV)
                    print(f"Проміжне збереження: {len(saved_df)} рядків")

                # М'яка зупинка через STOP-файл.
                if STOP_FILE.exists():
                    print("Знайдено STOP-файл. Парсер буде зупинено.")
                    STOP_FILE.unlink()
                    break

                # Пауза перед наступною компанією.
                if number < len(codes_to_process):
                    delay = random.uniform(DELAY_MIN, DELAY_MAX)
                    print(f"Пауза: {delay:.1f} сек.")
                    time.sleep(delay)

    except KeyboardInterrupt:
        print("\nПарсер зупинено вручну")

    finally:
        saved_df = save_results(results, OUTPUT_CSV, final=True)
        print_summary(saved_df, processed_in_current_run)


if __name__ == "__main__":
    main()
