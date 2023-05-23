from glob import glob
import os
import pandas as pd
import numpy as np
import argparse
import json
from plotnine import ggplot, aes, geom_line, geom_point, xlab, ylab, scale_x_continuous, scale_y_continuous

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_files_path", type=str, help="the folder where the result file paths are stored in")
    parser.add_argument("--hyperparam_json_path", type=str, help="specify path to json that stores all hyperparams")
    parser.add_argument("--hyperparam", type=str, help="specify hyperparam name to create ablation table for")
    parser.add_argument("--caption", type=str, help="specify caption for LaTeX table")
    return parser.parse_args()

def ablation_plot(sweep_table,
                  hyperparam,
                  save_path,
                  truncation_val=None):
    """
    Function that creates an ablation plot for the specified hyperparam, based on the data specified in sweep table.
    For a better interpretable plot, specify a hyperparameter value, of which bigger values are dropped.
    """
    # change table to long format 

    sweep_table_long = pd.melt(sweep_table,
                                id_vars=hyperparam,
                                var_name="Metric")
    
    # TRUNCATE VALUES!

    if truncation_val != None:
        sweep_table_long = sweep_table_long[sweep_table_long[hyperparam] <= truncation_val]

    plot = (

    ggplot(sweep_table_long)

    + aes(x=hyperparam, y="value", color="Metric")  

    + geom_line()

    + geom_point(shape='x', size=2) 
    
    + ylab("Score")

    + scale_x_continuous(breaks=round(sweep_table_long[hyperparam], 1))

    + scale_y_continuous(breaks=np.linspace(5, 45, 9))
    )

    plot.save(save_path, dpi=320)

    print("{} ablation plot saved in {}".format(hyperparam, save_path))


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
    
    result_table = pd.concat(result_files, axis=0).\
        applymap(lambda x: x*100).\
        drop(columns=["Dataset", "Model"])

    result_table["timestamp"] = index_NLG_metrics


    hyperparams_dicts_list_list = glob(os.path.join(args.hyperparam_json_path, '*.json'), recursive=True)

    #print(hyperparams_dicts_list_list[0])
    
    beta = []
    l = []
    index_hyperparams = []

    for hyperparam_dicts_list in hyperparams_dicts_list_list:


        file_name = os.path.split(hyperparam_dicts_list)[-1]
        if args.hyperparam in file_name:
            index_hyperparams.append(file_name.split('_MAGIC')[0])
            f = open(hyperparam_dicts_list)
            dict_list = json.load(f)
            beta.append(dict_list[0]["beta"])
            l.append(dict_list[0]["l"])
            # also append beta! then just select based on args.hyperparam later!

    hyperparams = pd.DataFrame([beta, l]).transpose()

    hyperparams = pd.concat([hyperparams, pd.Series(index_hyperparams)], axis=1)

    hyperparams.columns = ["beta", "l", "timestamp"]

    #print("Hyperparams df: {}".format(hyperparams))
    #print("result_table df: {}".format(result_table))

    beta_val_value = hyperparams["beta"].mode()[0]

    l_val_value = hyperparams["l"].mode()[0]

    sweep_table = pd.merge(hyperparams, result_table, on='timestamp', how='left').\
            sort_values(args.hyperparam).\
            drop(columns=["timestamp"]).\
            rename(columns={"Mean_NLG_M":"Mean Score"})
    
    sweep_table.columns = sweep_table.columns.str.replace('_', ' ')

    if args.hyperparam == "beta":
        sweep_table = sweep_table[sweep_table["l"] == l_val_value]
        sweep_table = sweep_table.drop_duplicates().drop(columns=["l", "Bleu 2", "Bleu 3"]) #.drop(columns=["l"])

        if "validation" in args.caption:

            save_path="../evaluation/plots/beta_ablation_validation_plot.png"

        else:

            save_path="../evaluation/plots/beta_ablation_plot.png"

        ablation_plot(sweep_table=sweep_table,
                    hyperparam=args.hyperparam,
                    save_path=save_path,
                    truncation_val=2)
            

    elif args.hyperparam == "l":
        sweep_table = sweep_table[sweep_table["beta"] == beta_val_value]
        sweep_table = sweep_table.drop_duplicates().drop(columns=["beta", "Bleu 2", "Bleu 3"])#, "Bleu_2", "Bleu_3"])

        ablation_plot(sweep_table=sweep_table,
                      hyperparam=args.hyperparam,
                      save_path="../evaluation/plots/l_ablation_plot.png")

    import sys
    sys.exit()

    latex_table = sweep_table.to_latex(index=False,
                                     caption=args.caption)
    
    
    if "AudioCaps" in args.caption:

        dataset_prefix = "AudioCaps_test_"

    elif "Clotho" in args.caption:

        dataset_prefix = "Clotho_test_"

    else:
        print("No familiar dataset selected!")
    
    path_name = "../evaluation/tables/" + "ablation_table_" + dataset_prefix + args.hyperparam + ".txt" 

    with open(path_name, 'w') as f:
        f.write(latex_table)

    #hyperparams = pd.Series(hyperparams)

    #hyperparams["timestamp"]= index_hyperparams

    #print(hyperparams)

    #print(result_table.to_latex(caption=args.caption))

    