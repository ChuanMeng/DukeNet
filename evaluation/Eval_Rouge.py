import sys
import glob
import json
import os
import time
from evaluation.Rouge import *
import codecs
import nltk


def rounder(num):
    return round(num, 2)


def rouge_max_over_ground_truths(prediction, ground_truths):
    scores_for_rouge1 = []
    scores_for_rouge2 = []
    scores_for_rougel = []
    for ground_truth in ground_truths:
        score = cal_rouge([prediction], [ground_truth])
        scores_for_rouge1.append(score[0])
        scores_for_rouge2.append(score[1])
        scores_for_rougel.append(score[2])
    return max(scores_for_rouge1), max(scores_for_rouge2), max(scores_for_rougel)


def eval_rouge_file(run_file, ref_file, tokenizer=None, detokenizer=None):
    run_dict = {}
    with codecs.open(run_file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t', 3)
            assert len(temp) == 4
            run_dict[temp[1]+'##<>##'+temp[2]] = temp[3] #assume it's already tokenized as it is for the groundtruth
    ref_dict = {}
    with codecs.open(ref_file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t', 3)
            assert len(temp) == 4
            tokenized = temp[3]
            if tokenizer is not None:
                tokenized = detokenizer(tokenizer(temp[3]))
            if temp[1] in ref_dict:
                ref_dict[temp[1]].append(tokenized)
            else:
                ref_dict[temp[1]] = [tokenized]

    run = []
    ref = []
    for id in run_dict:
        run.append(run_dict[id])  # [text1,text2,...]
        ref.append(ref_dict[id.split('##<>##')[0]])  # # [[text1],[text2],...]
    return eval_rouge(run, ref)


def cal_rouge(run, ref):
    x = rouge(run, ref)
    return x['rouge_1/f_score'] * 100, x['rouge_2/f_score'] * 100, x['rouge_l/f_score'] * 100


def eval_rouge(run, ref):
    rouge_1 = rouge_2 = rouge_l = total = 0
    assert len(run) == len(ref), "the length of predicted span and ground_truths span should be same"

    for i, pre in enumerate(run):
        rouge_result = rouge_max_over_ground_truths(pre, ref[i])
        rouge_1 += rouge_result[0]
        rouge_2 += rouge_result[1]
        rouge_l += rouge_result[2]
        total += 1

    rouge_1 = rouge_1 / total
    rouge_2 = rouge_2 / total
    rouge_l = rouge_l / total

    return {'ROUGE_1_F1': rounder(rouge_1), 'ROUGE_2_F1': rounder(rouge_2), 'ROUGE_L_F1': rounder(rouge_l)}


