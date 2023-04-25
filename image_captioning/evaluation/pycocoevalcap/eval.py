__author__ = 'tylin'
from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice
import json
import ipdb
import pandas as pd

class COCOEvalCap:
    def __init__(self, path):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}

        data = json.load(open(path))
        imgids = [item['sound_name'] for item in data]
        self.ground_truth, self.prediction = {}, {}
        for item in data:
            self.ground_truth[item['sound_name']] = [{'caption': i} for i in item['captions']]
            self.prediction[item['sound_name']] = [{'caption': item['prediction']}]
        self.params = {'image_id': imgids}

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.ground_truth[imgId]
            res[imgId] = self.prediction[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print ('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]



class COCOEvalCap_list:

    def __init__(self, one_res_dict_list):

        """
        one_res_dict: list of result dictionarys of all samples in the batch
        """

        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}

        data = one_res_dict_list
        imgids = [item['sound_name'] for item in data]
        self.ground_truth, self.prediction = {}, {}
        for item in data:
            self.ground_truth[item['sound_name']] = [{'caption': i} for i in item['captions']]
            self.prediction[item['sound_name']] = [{'caption': item['prediction']}]
        self.params = {'image_id': imgids}


    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}

        for imgId in imgIds:
            gts[imgId] = self.ground_truth[imgId]
            res[imgId] = self.prediction[imgId]

        # =================================================
        # Set up scorers
        # =================================================

        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================

        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================

        self.sample_metrics = []
        self.final_metrics = []

        for scorer, method in scorers:

            # score (for whole run): [B1, B2, B3, B4]
            # scores: B1(sample_1, sample_2 ,..., sample_N), B2(sample_1, sample_2, ..., sample_N)

            score, scores = scorer.compute_score(gts, res)


            if type(method) != list:

                if method == "SPICE":

                    current_metric = [scores[metric_nr]["All"]["f"] for metric_nr in range(len(scores))]
                    self.sample_metrics.append({method:current_metric})
                    current_metric = {method:score}
                    self.final_metrics.append(current_metric)
                    
                else:
                    current_metric = {method:score}
                    self.final_metrics.append(current_metric)
                    self.sample_metrics.append({method:scores})


            else:    
                self.final_metrics.append(dict(zip(method, score)))

                sample_metrics = {method[metric_nr]: scores[metric_nr] for metric_nr in range(len(scores))}
                self.sample_metrics.append(sample_metrics)   

            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)

            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)

        self.setEvalImgs()


        cider_and_spice = [metric for metric in self.final_metrics if "CIDEr" in metric.keys() or "SPICE" in metric.keys()]
        SPIDEr = (cider_and_spice[0]["CIDEr"] + cider_and_spice[1]["SPICE"]) / 2

        self.final_metrics.append({"SPIDEr":SPIDEr})

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]



class COCOEvalCap_obs:

    def __init__(self, one_res_dict):

        """
        one_res_dict: result dictionary of current sample
        """

        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}

        self.ground_truth, self.prediction = {}, {}
        self.ground_truth[one_res_dict["sound_name"]] = [{'caption': i} for i in one_res_dict['captions']]
        self.prediction[one_res_dict['sound_name']] = [{'caption': one_res_dict['prediction']}]

        imgids = one_res_dict['sound_name'] # list of sound names

        self.params = {'image_id': imgids}

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}

        gts[imgIds] = self.ground_truth[imgIds]
        res[imgIds] = self.prediction[imgIds]

        # =================================================
        # Set up scorers
        # =================================================

        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================

        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================

        self.metrics = []

        for scorer, method in scorers:

            score, scores = scorer.compute_score(gts, res)

            if type(method) != list:
                current_metric = {method:score}
                self.metrics.append(current_metric)

            else:    
                self.metrics.append(dict(zip(method, score)))

            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)

            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)

        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
