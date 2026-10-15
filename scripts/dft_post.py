from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("st2m.postproc.dft_post", run_name="__main__")
