"""Refresh USDBRL daily series through today (2026-06-09).

outputs/results/usd_brl.csv (yfinance dump) ends 2026-04-27. This fetches a
fresh daily series and writes outputs/results/usd_brl_2026_06_09.csv with
columns date,usdbrl. Tries yfinance first, falls back to BCB SGS series 1
(PTAX venda; same API pattern as src/features/macro/cdi_rates.py).
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'outputs/results'


def via_yfinance():
    import yfinance as yf
    px = yf.download('USDBRL=X', start='2021-12-01', progress=False, auto_adjust=False)
    if px is None or len(px) == 0:
        return None
    close = px['Close']
    if hasattr(close, 'columns'):  # multi-index ticker columns
        close = close.iloc[:, 0]
    out = close.reset_index()
    out.columns = ['date', 'usdbrl']
    return out


def via_bcb():
    import requests
    url = ('https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados'
           '?formato=json&dataInicial=01/12/2021')
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df['date'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
    df['usdbrl'] = df['valor'].astype(float)
    return df[['date', 'usdbrl']]


def main():
    out = None
    try:
        out = via_yfinance()
        src = 'yfinance USDBRL=X'
    except Exception as e:
        print(f"yfinance failed: {e}")
    if out is None or len(out) == 0:
        out = via_bcb()
        src = 'BCB SGS 1 (PTAX venda)'
    out = out.dropna().sort_values('date').reset_index(drop=True)
    path = OUT / 'usd_brl_2026_06_09.csv'
    out.to_csv(path, index=False)
    print(f"source: {src} | rows {len(out)} | {out['date'].min().date()} -> "
          f"{out['date'].max().date()} | last {out['usdbrl'].iloc[-1]:.4f}")
    print(f"saved: {path}")


if __name__ == '__main__':
    main()
