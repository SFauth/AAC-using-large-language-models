from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
import os
import glob
from json import encoder
from datetime import datetime
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file_path", type=str, help="the folder where the result file paths are stored in")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_config()

    file_pattern = os.path.join(args.result_file_path, '*.json')
    result_files = glob.glob(file_pattern)

    result_dicts = []

    for result_file in result_files:
        # calculate metrics and store 
        cocoEval = COCOEvalCap(result_file)
        cocoEval.evaluate()
    # add to cocoEval.eval dict all hyperparams (alpha, beta, prompt, ...). include them in the dict in args.result_file_path

        res_dict = cocoEval.eval

        all_res_dict = json.load(open(result_file))

    # all_res_dict: list of dicts

        keys_to_remove = ["captions", "sound_name", "split", "prediction"]
        for k in keys_to_remove:
            del all_res_dict[0][k]

        for hyperparam in all_res_dict[0].keys():
            res_dict[str(hyperparam)] = all_res_dict[0][str(hyperparam)]
        
        result_dicts.append(res_dict)

    filename = "test_" + datetime.today().strftime('%Y_%m_%d_%H_%M_%S') + ".json"
    with open (filename, "w", encoding="utf-8") as f:
        json.dump(result_dicts, f, ensure_ascii=False, indent=4)

    #for metric, score in cocoEval.eval.items():
     #   print ('%s: %.3f'%(metric, score))