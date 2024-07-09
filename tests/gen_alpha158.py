from KunTestUtil import gen_data
import pandas as pd
import argparse
import os
import qlib
import subprocess
import yaml
import numpy as np
from qlib.utils import init_instance_by_config

#based on qlib commit a7d5a9b500de5df053e32abf00f6a679546636eb
def main(tmppath: str, qlibroot: str, outdir: str):
    num_stock = 8
    num_time = 260
    print("Generating data")
    dopen, dclose, dhigh, dlow, dvol, damount = gen_data.gen_stock_data2(
        0.5, 100, num_stock, num_time, 0.03 if num_time > 1000 else 0.05, "float32")
    np.savez_compressed(os.path.join(outdir, "input.npz"), dopen = dopen, dclose = dclose, dhigh = dhigh, dlow = dlow, dvol = dvol, damount = damount)
    watch_list = []
    csvpath = os.path.join(tmppath, "csv")
    datapath = os.path.join(tmppath, "data")
    os.makedirs(csvpath, exist_ok=True)
    for i in range(1, num_stock+1):
        code = f"a{i:03d}"
        watch_list.append(code)
    for idx, tmp in enumerate(watch_list):
        df = pd.DataFrame(
            columns=["symbol", "date", "close", "open", "high", "low", "vwap", "volume"])
        df["symbol"] = [tmp] * num_time
        df["date"] = pd.date_range(start="2008-06-01", periods=num_time)
        df["close"] = dclose[idx]
        df["open"] = dopen[idx]
        df["high"] = dhigh[idx]
        df["low"] = dlow[idx]
        df["vwap"] = damount[idx] / dvol[idx]
        df["volume"] = dvol[idx]
        if idx == 0:
            print(df)
        df.to_csv(os.path.join(csvpath, f"{tmp}.csv"))
    print("Transforming data to qlib")
    subprocess.run(["python", f"{qlibroot}/scripts/dump_bin.py", "dump_all", "--csv_path", csvpath, "--qlib_dir", datapath, "--include_fields",
                   "close,open,high,low,vwap,volume", "--symbol_field_name", "symbol", "--date_field_name", "date"], check=True)
    os.rename(os.path.join(datapath, "instruments", "all.txt"), os.path.join(datapath, "instruments", "csi300.txt"))
    qlib.init(provider_uri=datapath)
    config_file = os.path.join(qlibroot, "examples", "benchmarks", "Linear", "workflow_config_linear_Alpha158.yaml")
    print("Using", config_file)
    with open(config_file) as f:
        config = yaml.safe_load(f)
    print("Computing alpha158")
    config["data_handler_config"]["infer_processors"].pop(0) # drop RobustZScoreNorm
    config["data_handler_config"]["infer_processors"].pop(0) # drop Fillna
    dataset = init_instance_by_config(config["task"]["dataset"])
    train: pd.DataFrame = dataset.prepare("train")
    print(train)
    data = dict([(c, train[c].to_numpy().reshape(num_time, num_stock)) for c in train.columns])
    del data["LABEL0"]
    print(train["KLEN"].to_numpy())
    print(data["KLEN"])
    np.savez_compressed(os.path.join(outdir, "alpha158.npz"), **data)
    print(list(data.keys()))
    print("Input data written to", os.path.join(outdir, "input.npz"))
    print("alpha158 data written to", os.path.join(outdir, "alpha158.npz"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Generate input and reference output for alpha158 based on qlib commit a7d5a9b500de5df053e32abf00f6a679546636eb")
    parser.add_argument("--tmp", required=True, type=str,
                        help="The path to a temp dir. It should be non existing or empty")
    parser.add_argument("--qlib", required=True, type=str,
                        help="The path to root dir of qlib source code")
    parser.add_argument("--out", required=True, type=str,
                        help="The path to a dir for outputs")
    args = parser.parse_args()
    main(args.tmp, args.qlib, args.out)
