from glob import glob
import os
import pandas as pd
import argparse

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_files_path", type=str, help="the folder where the result file paths are stored in")
    parser.add_argument("--hyperparam_json_path", type=str, help="specify path to json that stores all hyperparams")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()

    hyperparams_dict = glob(os.path.join(args.hyperparam_json_path, '*.json'), recursive=True)[0]
    hyperparams = pd.read_json(hyperparams_dict).loc[:,["alpha",
                                          "beta",
                                          "k",
                                          "temperature",
                                          "l",
                                          "keyword_prompt",
                                          "prompt",
                                          "end_penalty"]]
    print(hyperparams)
    print(type(hyperparams))

    latex_hyperparams = hyperparams.to_latex(index=False,
                                             header=True)
    print(latex_hyperparams)

    result_files = []

    for f in glob(os.path.join(args.result_files_path, '*.csv'), recursive=True):

        result_files.append(pd.read_csv(f).set_index(['Dataset', 'Model']))

    result_table = pd.concat(result_files, axis=0).applymap(lambda x: x*100).drop(columns=["Mean_NLG_M"])

    # take json and automatically include hyperparams in caption

    latex_results = result_table.to_latex()
    print(latex_results)