import codecs
from nltk.translate.meteor_score import *


def rounder(num):
    return round(num, 2)


def eval_meteor_file(run_file, ref_file, tokenizer=None, detokenizer=None):
    run_dict = {}
    with codecs.open(run_file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t', 3)
            assert len(temp) == 4
            run_dict[temp[1]+'##<>##'+temp[2]] = temp[3]
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


        meteor = 0.
    for id in run_dict:  # [text1(,text2)] vs text
        meteor += meteor_score(ref_dict[id.split('##<>##')[0]], run_dict[id])
    return {'METEOR': rounder(meteor*100/len(run_dict))}

