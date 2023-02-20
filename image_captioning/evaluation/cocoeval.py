from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from json import encoder
from datetime import datetime
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file_path", type=str, help="the result file to evaluate")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_config()
    cocoEval = COCOEvalCap(args.result_file_path)
    cocoEval.evaluate()
    # add to cocoEval.eval dict all hyperparams (alpha, beta, prompt, ...). include them in the dict in args.result_file_path

    res_dict = cocoEval.eval
    print("type res dict: " + str(type(res_dict)))
    all_res_dict = json.load(open(args.result_file_path))

    # all_res_dict: list of dicts

    for hyperparam in ["prompt", "alpha", "beta"]:
        res_dict[str(hyperparam)] = all_res_dict[0][str(hyperparam)]

    filename = "test_" + datetime.today().strftime('%Y_%m_%d_%H_%M_%S') + ".json"
    with open (filename, "w", encoding="utf-8") as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=4)

    #for metric, score in cocoEval.eval.items():
     #   print ('%s: %.3f'%(metric, score))