import ndjson
from utils.process_ndjson_quickdraw import Drawing
from tqdm import tqdm
import argparse
import os
import re
import pandas as pd
import numpy as np
from joblib import Parallel, delayed


def process_ndjson(filepath: str, out_path: str, num_samples=20, store_svg=True, raw=True):
    category = filepath[filepath.rfind("/") + 1:-7]
    category = re.sub('[^0-9a-zA-Z]+', '-', category)
    print("Reading :", filepath, "  --  ", category)
    try:
        with open(filepath) as f:
            data = ndjson.load(f)
    except Exception as e:
        print(f"\n\n - - - Error while opening a file {filepath} - - - \n\n")
        print(str(e))
        print("\n------------------\n")
        return
    for i_sketch in range(len(data[:num_samples])):
        drawing = Drawing.from_drawing_data(data[i_sketch]['drawing'], raw_ndjson=raw, apply_rdp=False, pad=True)
        drawing.dataset_dump(path=out_path + f"{category}_{i_sketch}.npz", side=288,
                             line_diameter=np.random.randint(low=4, high=8))
        if store_svg:
            drawing.write_svg(path=out_path[:-1] + f"_svg/{category}_{i_sketch}.svg")
    print(f"Done {category}: {num_samples} / {len(data)}")


def collect_dataset_csv(folder):
    coll = [{"filename": f, "category": f[:f.rfind("_")]}
           for f in os.listdir(folder)
           if f.endswith(".npz")]
    df = pd.DataFrame(coll)
    print(df.head())
    df.to_csv(folder + "dataset.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get dataset from quickdraw .ndjson files")
    parser.add_argument("--input", "-i", default="/media/ivan/DATA/QUICKDRAW_DIR/quickdraw_ndjson_raw/")
    parser.add_argument("--output", "-o", default="/home/ivan/datasets/quickdraw_tiny288/dataset/")
    parser.add_argument("-n", type=int, default=200, help="number of samples to draw per category")
    parser.add_argument("--svg", default=True, action="store_true")
    parser.add_argument("--no-svg", dest="svg", action="store_false")
    parser.add_argument("--raw", default=True, action="store_true")
    parser.add_argument("--jobs", "-j", type=int, default=4, help="Number of jobs in parallel")
    args = parser.parse_args()

    do_raw = args.raw
    input_dir = args.input
    assert (os.path.isdir(input_dir))
    output_dir = args.output
    if not (output_dir.endswith("/")):
        output_dir += "/"
    if not (os.path.exists(output_dir)):
        print("Creating directory")
        os.mkdir(output_dir)
        if args.svg:
            os.mkdir(output_dir[:-1] + "_svg/")

    if ("_raw" in input_dir) and not do_raw:
        print("\n\n Your data path has RAW in it, but no --raw flag provided. Are you sure?")
        input("Press any key to continue")

    n_samples = args.n
    do_svg = args.svg
    n_categories = len([f for f in os.listdir(input_dir) if f.endswith(".ndjson")])
    i = 1

    Parallel(n_jobs=args.jobs)(delayed(process_ndjson)(input_dir + file, output_dir, n_samples) for file in os.listdir(input_dir) if file.endswith("ndjson"))
    collect_dataset_csv(output_dir)
