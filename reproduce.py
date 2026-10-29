"""Generate the manuscript's numerical results and figures."""
from pathlib import Path
import argparse, os, subprocess, sys
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['all','data','results','figures'],default='all',nargs='?')
    args=parser.parse_args()
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if args.action=='data':
        subprocess.run([sys.executable,'scripts/build_data.py'],cwd=ROOT,env=env,check=True)
    if args.action in ('all','results'):
        subprocess.run([sys.executable,'scripts/compute_results.py'],cwd=ROOT,env=env,check=True)
    if args.action in ('all','figures'):
        subprocess.run([sys.executable,'scripts/plot_figures.py'],cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()
