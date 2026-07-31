"""
Daily production pipeline — single entry point.
Model: E1 D7 + H1 (no-short), 32 features, with risk controls.
Backtest 4.28y OOS (daily DD, 4 bps BRL, BAGS=160, 5-seed): CAGR +57%, Sortino 3.41, DD -6.68%.

Runs the 3 steps in order:
  1. fetch_raw_data.py          — incremental fetch from 12+ APIs
                                  (Binance spot/futures/funding, yfinance, FRED,
                                   BGeometrics, bitcoin-data.com [reserve-risk,
                                   puell-multiple], DefiLlama, CoinMetrics, BQ Messari)
  2. bootstrap_from_original.py — hybrid dataset: enhanced base + build_features
                                  for new days + V36 feature backfill
  3. generate_signal.py         — generate today's allocation signal +
                                  apply risk controls (kill switch / acc derisk / PSI)

Usage:
    python scripts/production/run_daily.py             # normal daily run
    python scripts/production/run_daily.py --retrain   # force model retrain (required
                                                        # after any config change)
    python scripts/production/run_daily.py --full      # full data rebuild + signal
"""
import sys, subprocess, logging, argparse
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

PROD_DIR = Path(__file__).parent
ROOT = PROD_DIR.parent.parent


def run_step(name, script, extra_args=None):
    """Run a production script and check for errors."""
    args = [sys.executable, str(PROD_DIR / script)]
    if extra_args:
        args.extend(extra_args)

    log.info(f"{'='*50}")
    log.info(f"  {name}")
    log.info(f"{'='*50}")

    try:
        result = subprocess.run(args, cwd=str(ROOT), timeout=600)
    except subprocess.TimeoutExpired:
        log.error(f"  TIMEOUT: {script} exceeded 600s (likely a slow/hung external API)")
        return False
    if result.returncode != 0:
        log.error(f"  FAILED: {script} exited with code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Daily production pipeline')
    parser.add_argument('--retrain', action='store_true', help='Force model retrain')
    parser.add_argument('--full', action='store_true', help='Full data rebuild')
    args = parser.parse_args()

    log.info(f"\n{'#'*50}")
    log.info(f"  DAILY PIPELINE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"{'#'*50}\n")

    # Step 1: Fetch raw data
    fetch_args = ['--full'] if args.full else []
    if not run_step("Step 1/3: Fetch raw data", "fetch_raw_data.py", fetch_args):
        log.error("Pipeline aborted at step 1")
        sys.exit(1)

    # Step 2: Bootstrap dataset
    if not run_step("Step 2/3: Bootstrap dataset", "bootstrap_from_original.py"):
        log.error("Pipeline aborted at step 2")
        sys.exit(1)

    # Step 3: Generate signal
    signal_args = ['--retrain'] if args.retrain else []
    if not run_step("Step 3/3: Generate signal", "generate_signal.py", signal_args):
        log.error("Pipeline aborted at step 3")
        sys.exit(1)

    log.info(f"\n{'#'*50}")
    log.info(f"  PIPELINE COMPLETE")
    log.info(f"{'#'*50}")


if __name__ == '__main__':
    main()
