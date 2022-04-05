import ndjson
from utils.process_ndjson_quickdraw import Drawing
from tqdm import tqdm
import argparse
import os
import re
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get dataset from quickdraw .ndjson files")
    parser.add_argument("--input", "-i", default="/home/ivan/datasets/quickdraw_ndjson_raw/")
    parser.add_argument("--output", "-o", default="/home/ivan/datasets/quickdraw_svg_raw/")
    parser.add_argument("-n", type=int, default=2000, help="number of samples to draw per category")
    args = parser.parse_args()

    input_path = args.input
    assert (os.path.isdir(input_path))
    output_path = args.output
    if not (os.path.exists(output_path)):
        print("Creating directory")
        os.mkdir(output_path)
    n_samples = args.n

    for ndj in [f for f in os.listdir(input_path) if f.endswith(".ndjson")]:
        print(ndj)
        # category = ndj[ndj.find("simplified_")+11:-7]
        category = ndj[:-7]
        category = re.sub('[^0-9a-zA-Z]+', '-', category)
        try:
            with open(input_path + ndj) as f:
                data = ndjson.load(f)
        except:
            print(f"\n\n - - - Error while opening a file {ndj} - - - \n\n")
            continue
        print(f"{category}: {len(data)}")
        if not os.path.exists(output_path+category+"/"):
            print("Creating subfolder")
            os.mkdir(output_path+category+"/")
        for i_sketch in tqdm(range(len(data[:n_samples]))):
            drawing = Drawing.from_drawing_data(data[i_sketch]['drawing'], raw_ndjson=True)
            drawing.write_svg(path=output_path+category+"/"+f"{category}_{i_sketch}.svg")
    ttt = [{"filename": f, "category": f[:f.rfind("_")]}
           for f in os.listdir(output_path)
           if f.endswith(".npz")]
    df = pd.DataFrame(ttt)
    print(df.head())
    df.to_csv(output_path+"dataset.csv", index=False)
