import json, sqlite3, pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT = DATA_DIR / "train_data.csv"

def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return pd.DataFrame(items)

def read_csv(file_path):
    return pd.read_csv(file_path)

def read_sqlite(file_path, table="incidents"):
    conn = sqlite3.connect(str(file_path))
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df

def harmonize_and_export():
    json_path = DATA_DIR / "source_api.json"
    csv_path = DATA_DIR / "source_csv.csv"
    db_path = DATA_DIR / "source_db.sqlite"

    parts = []
    if json_path.exists():
        parts.append(read_json(json_path))
    if csv_path.exists():
        parts.append(read_csv(csv_path))
    if db_path.exists():
        parts.append(read_sqlite(db_path))

    if not parts:
        raise RuntimeError("No data sources found in data/")

    df = pd.concat(parts, ignore_index=True, sort=False)

    if 'description' in df.columns and 'text' not in df.columns:
        df = df.rename(columns={'description': 'text'})
    if 'headline' in df.columns and 'summary' not in df.columns:
        df = df.rename(columns={'headline': 'summary'})

    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str).str.strip()
    df['summary'] = df.get('summary', pd.Series([""]*len(df))).astype(str).str.strip()

    df = df.drop_duplicates(subset=['text'])

    df[['text','summary']].to_csv(OUT, index=False, encoding='utf-8')
    print(f"Exported {len(df)} rows to {OUT}")

if __name__ == "__main__":
    harmonize_and_export()