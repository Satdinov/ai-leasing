import re
import json
import logging
from typing import Dict, Optional, List
from unidecode import unidecode
from sqlalchemy import inspect
from sqlalchemy.orm import Session
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek

from database import SessionLocal, engine
from models import FileMetadata
from config import Config
import os
from fastapi import HTTPException, status
from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

def normalize_table_name(filename: str) -> str:
    if filename.startswith('._'):
        filename = filename[2:]
    try:
        filename = filename.encode('cp437').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        try:
            filename = filename.encode('utf-8').decode('utf-8')
        except:
            pass
    name = os.path.splitext(filename)[0]
    name = unidecode(name)
    name = re.sub(r'[^\w]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    name = name.lower()
    if len(name) > Config.MAX_TABLE_NAME_LENGTH:
        name = name[:Config.MAX_TABLE_NAME_LENGTH]
    return name

def get_table_names(db: Session) -> List[str]:
    try:
        table_names = db.query(FileMetadata.table_name).filter(FileMetadata.table_name.isnot(None)).distinct().all()
        table_names = [t[0] for t in table_names]
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        return [t for t in table_names if t in existing_tables]
    except Exception as e:
        logger.error(f"Error getting table names: {e}")
        return []

def get_cache_path(model: str) -> str:
    return os.path.join(Config.CACHE_DIR, f"query_cache_{model}.json")

def get_cached_response(question: str, model: str) -> Optional[Dict]:
    cache_path = get_cache_path(model)
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r") as f:
            return json.load(f).get(question)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error reading cache: {e}")
        return None

def cache_response(question: str, model: str, response: Dict):
    cache_path = get_cache_path(model)
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading cache: {e}")
    cache[question] = response
    try:
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except IOError as e:
        logger.error(f"Error writing cache: {e}")

def get_llm(model_name: str):
    models = {
        "gemini": lambda: ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key=os.getenv("AITUNNEL_API_KEY"),
            base_url="https://api.aitunnel.ru/v1/"
        ),
        "chatgpt": lambda: ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("AITUNNEL_API_KEY"),
            base_url="https://api.aitunnel.ru/v1/",
            temperature=0,
        ),
        "deepseek": lambda: ChatDeepSeek(
            model="deepseek-r1",
            api_key=os.getenv("AITUNNEL_API_KEY"),
            base_url="https://api.aitunnel.ru/v1/"
        )
    }
    if model_name not in models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model: {model_name}"
        )
    return models[model_name]()

def build_sql_agent_prefix(db: SQLDatabase, table_names: List[str]) -> str:
    """Формирует префикс для SQL-агента с информацией о схеме и правилами."""
    with SessionLocal() as session:
        file_metadata = session.query(FileMetadata).all()
        file_info = "\n".join(
            f"- {row.file_name} ({row.file_type}) in {row.file_path or 'root'}"
            for row in file_metadata
        )
    table_info = db.get_table_info(table_names)
    prefix = f"""
        You are an AI assistant for a transport company (ООО РАД-ТРАК). You query a SQL database containing data converted from Excel files.
        Database schema: {table_info}
        Available tables: {table_names}
        Available files and their paths (from file_metadata):
        {file_info}
        Key Context:
        - **Revenue**: Cash and non-cash trips.
        - **Expenses**: Diesel, repairs (truck/trailer), driver salaries, trip allowances (repairs, fines, parking), insurance (OSAGO/KASKO, cargo liability), Platon (federal road toll), toll roads.
        - **Mileage**: Monthly mileage per driver (from 'voditeli_probegi_v_2024') and possibly per vehicle (from other tables). Drivers may switch vehicles.
        - **Key Metrics**:
            1. Revenue per km = Revenue / Mileage (rub/km) → higher is better.
            2. Fuel consumption (L/100km) = (Liters of diesel per month / Mileage per month) * 100 → lower is better.
            3. Repair cost per km = Repair costs per month / Mileage per month (rub/km).
            4. Driver salary = Mileage * 15 rub.
        Important Rules:
        1. **Exclude summary rows**: In any table, if a row has 'Автомобиль' or 'Гос. № тягача / прицепа' equal to 'Итого', NULL, or empty string, EXCLUDE it.
        2. **Repair Data**:
            - For repairs in a specific month (e.g., January), use the table named like '01_remont_ianvar_24' (for January 2024) from the folder '3. Ремонт по месяцам'.
            - In repair tables, exclude rows where the repair cost column (e.g., 'Итого', 'Сумма работ', 'Сумма запчастей') is NULL or empty.
            - Each vehicle (truck+trailer) is represented by two consecutive rows: first row is the truck, second is the trailer. 
            - To get total repair cost for a vehicle (truck+trailer) in a month, sum the 'Итого' of the two rows.
        3. **Vehicle Identification**:
            - Always use 'Марка тягача' for truck brand and 'Марка прицепа' for trailer brand. Never use the license plate column for brand.
            - Use the following brand synonyms (use ILIKE with any of these for matching):
                - "Mercedes-Benz": also include "MB", "M-B", "Мерседес", "Мерседес Бенц", "МЕРСЕDES", "МЕРСЕDES-BENZ"
                - "Volvo": also "Вольво"
                - "Shacman": also "Шакман"
                - "Scania": also "Скания"
                - "MAN": also "Ман"
                - "DAF": also "ДАФ"
                - "IVECO": also "Ивеко"
                - "Renault": also "Рено"
                - "HOWO": also "Хово"
                - "LOHR": also "ЛОР", "ЛОХР"
        4. **Driver Identification**:
            - In 'spisok_tekhniki_perevozchika_i_voditelei_ooo_rad_trak', driver full name is in column "Водитель  ФИО".
            - In 'voditeli_probegi_v_2024', driver name is in column "Водитель" (in the format "Фамилия Имя").
            - In 'godovoi_otchet_01_01_31_12_24', driver name is in column "Водитель" (format?).
            - Currently, there is no direct mapping between full name and short name. For queries requiring driver data from multiple tables, use the exact string as found in each table. 
              If a question requires combining data from tables with different driver name formats, focus on one table at a time and inform the user about the limitation.
        5. **Fuel Data**: 
            - Use tables from the folder '2. Заправки'. For example, for January 2024, look for tables with 'ianvar_24' in the name and located in that folder.
            - Sum the 'Кол-во' (quantity) of diesel for the month from all fueling tables to get total liters.
        6. **Mileage Data**:
            - Use 'voditeli_probegi_v_2024' for driver mileage per month. This table does not have vehicle details, only driver name and monthly mileage.
            - For vehicle-specific mileage, we may not have direct data. The user might need to derive it from other sources (like trip logs). Currently, avoid vehicle mileage if not available.
        7. **Financial Data** (revenue, expenses): 
            - Use 'godovoi_otchet_01_01_31_12_24'. Note: each vehicle (truck+trailer) has two rows. 
            - For truck-related financials, use the first row (truck) of a vehicle pair.
            - For trailer-related, use the second row (trailer).
            - For total per vehicle (truck+trailer), sum both rows.
        8. **Query Execution**:
            - Always use ILIKE for string matching (case-insensitive).
            - Validate column names against the schema. If a column does not exist, try the closest match (e.g., if 'Водитель ФИО' is not found, try 'Водитель' or 'Водитель  ФИО').
            - If a query fails, check the schema and correct the column/table name. Retry up to 3 times.
            - When joining tables, use LEFT JOIN to avoid losing rows from the primary table.
            - When asked for a file location, query the 'file_metadata' table.
        Answer in Russian.
        Query Strategy by Question Type:
        **1. Driver Rankings (by revenue or mileage):**
            - Revenue ranking:
                Step 1: From 'godovoi_otchet_01_01_31_12_24', calculate total revenue per driver (sum of revenue for all rows with that driver name). 
                         Note: each driver may have multiple rows (for truck and trailer). Group by driver.
                Step 2: Order by total revenue DESC.
            - Mileage ranking:
                Step 1: From 'voditeli_probegi_v_2024', take the column for the required month (e.g., "Январь" for January) or the annual total ("Общий пробег").
                Step 2: Order by mileage DESC.
        **2. Largest/Smallest Repair in a Month (e.g., January 2024):**
            - Use table '01_remont_ianvar_24' (if exists).
            - If the table has multiple months, filter by the relevant month column (e.g., "Сумма работ - Январь 2024"). But ideally, the table is for January.
            - For each vehicle (truck and trailer are separate), take the repair cost from the 'Итого' column (or if not exists, sum 'Сумма работ' and 'Сумма запчастей').
            - But note: the user wants per vehicle (truck+trailer). So group by the vehicle identifier (e.g., the pair of rows with the same index) and sum the two rows.
            - However, in the absence of a clear group, you might have to treat truck and trailer separately? 
              The user's example question: "У кого самый большой ремонт за январь?" - likely meaning per driver or per vehicle. 
              Since repairs are tied to a vehicle, and a vehicle is assigned to a driver, we can use the driver as the grouping factor? 
            - Alternatively: the table '01_remont_ianvar_24' has a column "Гос. № тягача / прицепа". The two rows for the same vehicle have the same index (like 1.0, then the next row is 1.0 for trailer). 
              So group by the index (before the decimal) and sum the repair costs for the two rows. Then the driver is the one in the first row (for the truck) of that group.
            - Steps:
                - Exclude rows with "Гос. № тягача / прицепа" = 'Итого', NULL, or empty.
                - For each unique index (the integer part of the number in the first column, e.g., 1, 2, ...), sum the 'Итого' for the two rows (truck and trailer) to get the total repair cost for that vehicle.
                - Then, for each group, take the driver from the truck row (the first row of the pair).
                - Then rank by the total repair cost.
        **3. Fuel Consumption by Truck:**
            - Step 1: Get total fuel consumption (liters) for each truck in a month:
                - From all fueling tables in '2. Заправки' for the month, we need to aggregate by truck. But the fueling tables don't always have the truck identifier? 
                - In the example table `01_rosneft_ianvar_24`, there is no truck identifier. This is a problem.
            - Alternative: 
                - The annual report table (`godovoi_otchet_01_01_31_12_24`) has fuel expenses by vehicle (by truck and trailer). But it has fuel in rubles, not liters.
                - Without liters data, we cannot compute liters/100km.
            - Since the problem is complex and data might be incomplete, we might avoid this until we have better data. 
            - However, the user provided an example table for Rosneft. Let's reexamine: 
                The table has columns: Дата, Кол-во (quantity), ... 
                But it doesn't have the vehicle or driver. So we cannot attribute fuel to a specific truck.
            - This is a data limitation. We must inform the user.
            - If the user insists, we might have to use an alternative: 
                - In the annual report table, there are columns 'роснефть' and 'татнефть' which are fuel expenses in rubles. And we have the price per liter? 
                - But we don't have the price per liter in the table. 
            - Conclusion: without liters per truck, we cannot compute this metric. Skip until we have proper data.
        **4. Repair Cost per km for a Vehicle Brand (e.g., Mercedes):**
            - Step 1: Get the total repair cost for the brand (for a given time period, e.g., January 2024):
                - Use the repair table for the month (e.g., '01_remont_ianvar_24').
                - Filter rows by 'Марка тягача' (for trucks) or 'Марка прицепа' (for trailers) using the brand synonyms.
                - For Mercedes trucks: where "Марка тягача" ILIKE any of the Mercedes synonyms.
                - Sum the repair costs (Итого) for both rows (truck and trailer) of each vehicle that has a Mercedes truck.
            - Step 2: Get the total mileage for these vehicles in the same month:
                - Problem: the mileage table is per driver, not per vehicle. And we don't have a direct link from vehicle to driver in the repair table for that month? 
                - In the repair table, there is a column "Гос. № тягача / прицепа" and also a driver column. 
                - For the vehicles filtered in step 1, take the driver name from the repair table.
                - Then, from the mileage table ('voditeli_probegi_v_2024'), get the mileage for that driver in the month.
                - But: one driver might drive multiple vehicles? And one vehicle might be driven by multiple drivers? 
                - This is an approximation: assume the driver in the repair table is the main driver for that vehicle in the month.
            - Step 3: Repair cost per km = (total repair cost) / (total mileage for the driver in the month).
            - Note: This is an estimate and might not be precise.
        Given the complexity and data limitations, always inform the user about assumptions and limitations.
        Additional Notes:
        - When generating SQL, always use double quotes for column and table names if they contain spaces or special characters.
        - For month names in Russian, use the Russian month name in the column (e.g., "Январь").
        - For the 'voditeli_probegi_v_2024' table, the columns are named by month in Russian.
        Example Queries:
        **Example 1: Driver revenue ranking for January 2024**
        ```sql
        SELECT "Водитель", SUM("Итого") AS total_revenue
        FROM godovoi_otchet_01_01_31_12_24
        WHERE ...  -- if there is a month column? But the table is for the whole year.
        GROUP BY "Водитель"
        ORDER BY total_revenue DESC;
        ```
        But the annual report table doesn't have a month breakdown. So we cannot get January alone from it.
        This indicates a data model limitation. We might need to use other tables for monthly revenue.
        **Alternative for revenue**: The user might have trip tables (like '01_volvo_r_876_kha'). But they are not provided in the schema. 
        Given the provided tables, we might not be able to answer all questions. Focus on what's available.
        We must adapt to the available data.
        Final note: The assistant should be cautious and inform the user when data is missing or assumptions are made.
        """
    return prefix

def normalize_brand_name(brand: str) -> str:
    """Нормализует название бренда для поиска в базе."""
    brand = brand.lower().strip()
    brand_map = {
        "mercedes-benz": ["mercedes-benz", "mercedes", "m-b", "mb", "мерседес", "мерседес бенц", "mercedes benz", "мерседес-бенц", "мерсeдес"],
        "volvo": ["volvo", "вольво"],
        "shacman": ["shacman", "шакман"],
        "scania": ["scania", "скания"],
        "man": ["man", "ман"],
        "daf": ["daf", "даф"],
        "iveco": ["iveco", "ивеко"],
        "renault": ["renault", "рено"],
        "howo": ["howo", "хово"],
        "lohr": ["lohr", "лор", "лохр"]
    }
    brand_normalized = unidecode(brand).lower()
    for canonical, synonyms in brand_map.items():
        if brand in [s.lower() for s in synonyms] or brand_normalized in [unidecode(s).lower() for s in synonyms]:
            return canonical
    return brand

def modify_query_for_brands(query: str, table_name: str) -> str:
    """Модифицирует SQL-запрос для учета синонимов брендов."""
    if "Марка тягача" not in query or table_name not in query:
        return query
    brand_pattern = r"ILIKE\s*'%([^']+)%'"
    match = re.search(brand_pattern, query, re.IGNORECASE)
    if not match:
        return query
    brand = match.group(1)
    normalized_brand = normalize_brand_name(brand)
    if normalized_brand == brand:
        return query
    synonyms = next((synonyms for canonical, synonyms in [
        ("МЕRCEDES-BENZ", ["mercedes-benz", "mercedes", "m-b", "mb", "мерседес", "мерседес бенц", "mercedes benz", "мерседес-бенц", "мерсeдес"]),
        ("volvo", ["volvo", "вольво"]),
        ("shacman", ["shacman", "шакман"]),
        ("scania", ["scania", "скания"]),
        ("man", ["man", "ман"]),
        ("daf", ["daf", "даф"]),
        ("iveco", ["iveco", "ивеко"]),
        ("renault", ["renault", "рено"]),
        ("howo", ["howo", "хово"]),
        ("lohr", ["lohr", "лор", "лохр"])
    ] if normalized_brand == canonical), [])
    if not synonyms:
        return query
    conditions = [f""""Марка тягача" ILIKE '%{synonym}%'""" for synonym in synonyms]
    new_condition = " OR ".join(conditions)
    logger.info(f"Original brand: {brand}, Normalized: {normalized_brand}, Conditions: {new_condition}")
    return re.sub(brand_pattern, new_condition, query, flags=re.IGNORECASE)


class CustomAgentExecutor(Runnable):
    """Обертка для AgentExecutor, модифицирующая SQL-запросы перед выполнением."""

    def __init__(self, agent, table_name: str):
        self.agent = agent
        self.table_name = table_name

    def invoke(self, input, config=None, **kwargs):
        modified_input = modify_query_for_brands(input, self.table_name)
        logger.info(f"Original query: {input}")
        logger.info(f"Modified query: {modified_input}")
        return self.agent.invoke(modified_input, config, **kwargs)

    def __getattr__(self, name):
        return getattr(self.agent, name)


def create_sql_agent_wrapper(llm, db, table_names: List[str]):
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=True,
        include_tables=table_names,
        prefix=build_sql_agent_prefix(db, table_names),
        max_iterations=15,
        handle_parsing_errors=True
    )

    # Возвращаем обертку для агента
    return CustomAgentExecutor(agent, "spisok_tekhniki_perevozchika_i_voditelei_ooo_rad_trak")
