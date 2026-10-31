"""Optional full-scope refresh using the same course PostgreSQL method as R.

Run locally, enter the course password at the hidden prompt:
    python refresh_all_market_results.py

Queries exactly the 78,543 markets in the seven supplied trade files, in 79
batches of at most 1,000 slugs. It does not limit the export to pilot markets,
drop unresolved markets or filter settlement dates to the trading week.
"""
from pathlib import Path
import argparse
import pandas as pd
from full_scope_query import extract_market_settlements


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--market-list', type=Path, default=root / 'data/markets_in_yuzhe_files.csv')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    requested = pd.read_csv(args.market_list)
    assert len(requested) == 78543 and requested.market.nunique() == 78543
    print('Requesting all 78,543 observed markets. This fetches metadata/results, not more trades.')
    output = extract_market_settlements(market_file=args.market_list, output_dir=args.output_dir,
                                       project_dir=root, batch_size=1000)
    exported = pd.read_csv(output)
    assert len(exported) == len(requested) and set(exported.market) == set(requested.market)
    print('Full export verified:', output)
