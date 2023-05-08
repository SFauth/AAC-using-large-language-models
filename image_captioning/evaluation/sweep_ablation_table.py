from glob import glob
import os
import pandas as pd
import numpy as np
import argparse
import json


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_files_path", type=str, help="the folder where the result file paths are stored in")
    parser.add_argument("--hyperparam_json_path", type=str, help="specify path to json that stores all hyperparams")
    parser.add_argument("--hyperparam", type=str, help="specify hyperparam name to create ablation table for")
    parser.add_argument("--caption", type=str, help="specify caption for LaTeX table")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()

    # go into validation/validation_performance and get NLG metrics
    # go into output_jsons/validation_performance and get l / beta

    result_file_list = glob(os.path.join(args.result_files_path, '**/*.csv'), recursive=True)
    index_NLG_metrics = []
    result_files = []

    for file in result_file_list:
        
        file_name = os.path.split(file)[-1]
        if args.hyperparam in file_name:
            #print(file_name)
            result_files.append(pd.read_csv(file))
            index_NLG_metrics.append(file_name.split('_MAGIC')[0])
    
    result_table = pd.concat(result_files, axis=0).applymap(lambda x: x*100).drop(columns=["Mean_NLG_M", "Dataset", "Model"])

    result_table["timestamp"] = index_NLG_metrics

    #print(result_table.shape)

    hyperparams_dicts_list_list = glob(os.path.join(args.hyperparam_json_path, '*.json'), recursive=True)

    #print(hyperparams_dicts_list_list[0])
    
    hyperparams = []
    index_hyperparams = []

    for hyperparam_dicts_list in hyperparams_dicts_list_list:


        file_name = os.path.split(hyperparam_dicts_list)[-1]
        if args.hyperparam in file_name:
            index_hyperparams.append(file_name.split('_MAGIC')[0])
            f = open(hyperparam_dicts_list)
            dict_list = json.load(f)
            hyperparams.append(dict_list[0][args.hyperparam])


    hyperparams = pd.DataFrame([hyperparams, index_hyperparams]).transpose()
    hyperparams.columns = [str(args.hyperparam), "timestamp"]

    sweep_table = pd.merge(hyperparams, result_table, on='timestamp', how='left').\
            drop_duplicates(subset=[args.hyperparam]).\
            sort_values(args.hyperparam).\
            drop(columns=["timestamp"])



    latex_table = sweep_table.to_latex(index=False,
                                     caption=args.caption)
    
    path_name = "../evaluation/tables/" + "ablation_table_" + args.hyperparam + ".txt" 

    with open(path_name, 'w') as f:
        f.write(latex_table)

    #hyperparams = pd.Series(hyperparams)

    #hyperparams["timestamp"]= index_hyperparams

    #print(hyperparams)

    #print(result_table.to_latex(caption=args.caption))

    