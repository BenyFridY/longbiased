"""
Fetch real CDI series from Banco Central API and recompute strategy returns.

BCB SGS series:
  - 12 = CDI daily (% per day, decimal — already daily rate, not annualized)
"""
import json
import sys
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT_CSV = ROOT / 'outputs' / 'results' / 'cdi_real_bcb.csv'
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def fetch_series(code: int, start: str, end: str) -> pd.DataFrame:
    """Fetch a BCB SGS series. start/end in DD/MM/YYYY format."""
    url = (
        f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados'
        f'?formato=json&dataInicial={urllib.parse.quote(start)}'
        f'&dataFinal={urllib.parse.quote(end)}'
    )
    print(f'Fetching: {url}')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode('utf-8'))
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
    df['valor'] = df['valor'].astype(float)
    return df[['date', 'valor']].sort_values('date').reset_index(drop=True)


def main():
    cdi = fetch_series(12, '01/01/2022', '04/05/2026')
    print(f'Got {len(cdi)} observations of CDI daily rate')
    print(f'Date range: {cdi.date.min().date()} to {cdi.date.max().date()}')
    # Series 12 reports the daily CDI rate as a percentage (e.g. 0.04452 = 0.04452%)
    # Convert to fractional daily rate
    cdi['cdi_daily'] = cdi['valor'] / 100.0
    cdi['cdi_annualized_pct'] = ((1 + cdi['cdi_daily']) ** 252 - 1) * 100  # business day convention

    print()
    print('Annual averages of CDI:')
    cdi['year'] = cdi['date'].dt.year
    yearly = cdi.groupby('year').agg(
        days=('cdi_daily', 'count'),
        cum_pct=('cdi_daily', lambda x: ((1 + x).prod() - 1) * 100),
        annualized=('cdi_annualized_pct', 'mean'),
    )
    print(yearly.round(2).to_string())

    print()
    print('Cumulative CDI from 2022-01-07 to 2026-04-24:')
    mask = (cdi['date'] >= '2022-01-07') & (cdi['date'] <= '2026-04-24')
    sub = cdi[mask]
    cum = (1 + sub['cdi_daily']).prod() - 1
    days = len(sub)
    years = days / 252
    cagr = (1 + cum) ** (1 / years) - 1
    print(f'  Cumulative: {cum*100:+.2f}%')
    print(f'  Business days: {days}  (years: {years:.2f})')
    print(f'  CAGR: {cagr*100:+.2f}%')
    print(f'  Constant 13% baseline gives:  {((1.13)**years - 1)*100:+.2f}%  (CAGR 13.00%)')

    # Save to CSV
    cdi[['date', 'cdi_daily']].to_csv(OUT_CSV, index=False)
    print(f'\nSaved: {OUT_CSV}')


if __name__ == '__main__':
    main()
