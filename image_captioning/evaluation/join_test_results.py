from glob import glob
import os
import pandas as pd
import numpy as np
import argparse
import sys

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_files_path", type=str, help="the folder where the result file paths are stored in")
    parser.add_argument("--hyperparam_json_path", type=str, help="specify path to json that stores all hyperparams")
    return parser.parse_args()

def make_max_cells_bold(val, max_val, second_max_val):
    if val == max_val:
        return '\textbf{' + str(val) + '}'
    elif val == second_max_val:
        return '\textbf{' + str(val) + '}'
    else:
        return val


if __name__ == '__main__':
    args = parse_config()

    hyperparams_dict = glob(os.path.join(args.hyperparam_json_path, '*.json'), recursive=True)[0]
    hyperparams = pd.read_json(hyperparams_dict).loc[0:1,["alpha",
                                          "beta",
                                          "k",
                                          "temperature",
                                          "l",
                                          "keyword_prompt",
                                          "prompt",
                                          "end_penalty"]]

    latex_hyperparams = hyperparams.to_latex(index=False,
                                             header=True)
    print(latex_hyperparams)

    result_files = []

    for f in glob(os.path.join(args.result_files_path, '**/*.csv'), recursive=True):

        result_files.append(pd.read_csv(f).set_index(['Dataset', 'Model']))
        
    result_table = pd.concat(result_files, axis=0).applymap(lambda x: x*100).drop(columns=["Mean_NLG_M"])
    
    # add supervised SOTA results for clotho and AudioCaps

    SOTA = pd.DataFrame({'Dataset':["AudioCaps", "clotho"],
                         'Model':['Supervised SOTA', 'Supervised SOTA'],
                         'Bleu_1':[70.7, 60.1],
                         'Bleu_2':[0, 0],
                         'Bleu_3':[0, 0],
                         'Bleu_4':[28.3, 18.0],
                         'METEOR':[25.0, 18.5],
                         'ROUGE_L':[50.7, 40.0],
                         'CIDEr':[78.7, 48.8],
                         'SPICE':[18.2, 13.3],
                         'SPIDEr':[48.5, 31.0]}).set_index(['Dataset', 'Model'])
    
    ablation_table = pd.concat([result_table, SOTA], axis=0).reset_index()

    groups = [
        ablation_table["Model"].str.contains('Baseline'),
        ablation_table["Model"].str.contains('AudioCLIP'),
        ablation_table["Model"].str.contains('LAION'),
        ablation_table["Model"].str.contains('MAGIC_WavCaps'),
        ablation_table["Model"].str.contains('WavCaps'),
        ablation_table["Model"].str.contains('SOTA')
    ]

    group_names = ['Baseline', 'MAGIC AudioCLIP', 'MAGIC LAION', 'MAGIC WavCaps', 'WavCaps', 'SOTA']

    
    ablation_table["Group"] = pd.Series(np.select(groups, group_names))
    ablation_table["Dataset"] = np.where(ablation_table["Dataset"] == "clotho",
                                         "Clotho",
                                         ablation_table["Dataset"])


    ablation_table = ablation_table.groupby(["Dataset", "Group"]).apply(lambda x: x.sort_values('Bleu_1'))

    """
    metric_cols = ablation_table.iloc[:, 1:]    
    ac = metric_cols.iloc[:len(metric_cols)//2]
    clotho = metric_cols.iloc[len(metric_cols)//2:]

    for col in ac.columns:
        max_val = ac[col].max()
        second_max_val = ac[col][ac[col] != max_val].max()
        ac[col] = ac[col].apply(lambda x: make_max_cells_bold(x, max_val, second_max_val))
        
        max_val = clotho[col].max()
        second_max_val = clotho[col][clotho[col] != max_val].max()
        clotho[col] = clotho[col].apply(lambda x: make_max_cells_bold(x, max_val, second_max_val))

    ablation_table = pd.concat([ac, clotho], axis=0)
    """

    ablation_table = ablation_table.reset_index(level=[0,1,2], drop=True).drop(columns=['Group', 'Dataset'])


    index = [""] * ablation_table.shape[0]
    index[0] = "AudioCaps"
    index[len(index)//2] = "Clotho"
    ablation_table.index = index
    ablation_table.index.name = "Dataset"

    ablation_table.columns = ablation_table.columns.str.replace('_', ' ')
    ablation_table["Model"] = ablation_table["Model"].str.replace('_', ' ')

    latex_results = ablation_table.to_latex()
    
    with open('ablation_table.txt', 'w') as f:
        f.write(latex_results)
    # take json and automatically include hyperparams in caption

