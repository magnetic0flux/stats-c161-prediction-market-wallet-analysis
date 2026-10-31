"""Full-scope course-database query, translated from the supplied R workflow."""

from pathlib import Path
from datetime import datetime, timezone
import getpass
import pandas as pd
PROJECT_DIR = Path(__file__).resolve().parents[1]

def require_columns(frame, required, label):
    missing = sorted(set(required) - set(frame))
    if missing:
        raise ValueError(f'{label} is missing columns: {missing}')

def new_project_output(prefix, project_dir=PROJECT_DIR):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    return Path(project_dir) / '04_results' / 'python_import' / f'{prefix}_{stamp}'

def make_output(output_dir, prefix, project_dir=PROJECT_DIR):
    path = Path(output_dir) if output_dir is not None else new_project_output(prefix, project_dir)
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()

SETTLEMENT_SQL = """
WITH requested(market) AS (SELECT unnest(%s::text[]))
SELECT p.market, (m.slug IS NOT NULL) AS metadata_found,
       m.title AS market_question, m.condition_id, m.event_slug,
       e.title AS event_title, m.start_date, m.end_date, m.active, m.closed,
       m.token_ids::text AS token_ids, m.outcomes::text AS outcomes,
       (r.market_slug IS NOT NULL) AS has_resolution, r.resolved_yes,
       CASE WHEN r.resolved_yes IS NULL THEN NULL
            WHEN jsonb_array_length(m.outcomes) <> 2 THEN NULL
            WHEN r.resolved_yes THEN m.outcomes ->> 0
            ELSE m.outcomes ->> 1 END AS winning_outcome,
       r.price_at_close, r.resolved_at, r.uma_resolution_status,
       CURRENT_TIMESTAMP AS extracted_at
FROM requested AS p
LEFT JOIN public.markets AS m ON m.slug = p.market
LEFT JOIN public.events AS e ON e.slug = m.event_slug
LEFT JOIN public.resolutions AS r ON r.market_slug = m.slug
ORDER BY p.market
"""

def extract_market_settlements(market_file=None, output_dir=None, project_dir=PROJECT_DIR,
        host='aws-0-us-west-2.pooler.supabase.com', port=6543, dbname='postgres',
        user='polymarketdatabase.merbqatlxbrcvtbnuaqe', password=None, batch_size=1000):
    """Optional SELECT-only export, same database/tables/scope as the supplied R.

    Invoked explicitly, never during import or pilot calculation. Passwords are
    prompted locally and are not logged/saved. No wallet trades are fetched.
    """
    try:
        import psycopg
    except ImportError as e:
        raise ImportError('Optional DB step: pip install "psycopg[binary]>=3.1,<4"') from e
    index = pd.read_csv(market_file or Path(project_dir) / 'data/markets_in_yuzhe_files.csv', dtype='string')
    require_columns(index, ['market', 'n_trades'], 'market index')
    if index.market.isna().any() or index.market.str.strip().eq('').any():
        raise ValueError('Missing/empty market slugs.')
    slugs = sorted(set(index.market))
    if not slugs or batch_size < 1:
        raise ValueError('Empty market list or invalid batch size.')
    password = password or getpass.getpass('Enter the course PostgreSQL password: ')
    if not password:
        raise ValueError('No password entered; no database query was run.')
    destination = make_output(output_dir, 'market_results', project_dir)
    output, partial = destination / 'market_results.csv', destination / 'market_results.csv.part'
    count = 0
    with psycopg.connect(host=host, port=port, dbname=dbname, user=user, password=password,
                         sslmode='require', connect_timeout=15, options='-c timezone=UTC') as connection:
        password = None
        connection.read_only = True
        with connection.cursor() as cursor:
            for start in range(0, len(slugs), batch_size):
                batch = slugs[start:start + batch_size]
                # Pass values separately. No slug is interpolated into SQL text.
                cursor.execute(SETTLEMENT_SQL, (batch,))
                frame = pd.DataFrame(cursor.fetchall(), columns=[c.name for c in cursor.description])
                if len(frame) != len(batch) or frame.market.duplicated().any() or set(frame.market) != set(batch):
                    raise ValueError('Database join returned unexpected market keys; inspect the partial export.')
                frame.to_csv(partial, mode='w' if start == 0 else 'a', header=start == 0, index=False)
                count += len(frame)
                print(f'Metadata exported: {count:,}/{len(slugs):,}', flush=True)
    partial.replace(output)
    return output
