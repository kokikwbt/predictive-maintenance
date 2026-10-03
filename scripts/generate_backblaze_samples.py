"""Save compact Backblaze example plots from an already downloaded quarter."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import matplotlib
    matplotlib.use("Agg")
    from pdmdata.datasets.backblaze.viz import save_samples

    for sample in save_samples():
        print(f"{sample['image']}: {sample['observations']} observations, {sample['bytes']:,} bytes")


if __name__ == "__main__":
    main()
