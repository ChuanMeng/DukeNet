import codecs
from nltk.translate.bleu_score import sentence_bleu


def rounder(num):
    return round(num, 2)


def eval_bleu_file(run_file, ref_file, tokenizer=None, detokenizer=None):
    run_dict = {}
    with codecs.open(run_file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t', 3)
            assert len(temp) == 4
            run_dict[temp[1]+'##<>##'+temp[2]] = temp[3].split(' ') #assume it's already tokenized as it is for the groundtruth

    ref_dict = {}
    with codecs.open(ref_file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t', 3)
            assert len(temp) == 4
            tokenized = temp[3]
            if tokenizer is not None:
                tokenized = detokenizer(tokenizer(temp[3])).split(' ')
            else:
                tokenized = tokenized.split(' ')
            if temp[1] in ref_dict:
                ref_dict[temp[1]].append(tokenized)
            else:
                ref_dict[temp[1]] = [tokenized]

    bleu1 = 0.
    bleu2 = 0.
    bleu3 = 0.
    bleu4 = 0.


    for id in run_dict:
        bleu1 += sentence_bleu(ref_dict[id.split('##<>##')[0]], run_dict[id], weights=[1, 0, 0, 0])
        bleu2 += sentence_bleu(ref_dict[id.split('##<>##')[0]], run_dict[id], weights=[0.5, 0.5, 0, 0])
        bleu3 += sentence_bleu(ref_dict[id.split('##<>##')[0]], run_dict[id], weights=[1/3, 1/3, 1/3, 0])
        bleu4 += sentence_bleu(ref_dict[id.split('##<>##')[0]], run_dict[id], weights=[0.25, 0.25, 0.25, 0.25])

    return {'BLEU-1': rounder(bleu1*100/len(run_dict)), 'BLEU-2': rounder(bleu2*100/len(run_dict)), 'BLEU-3': rounder(bleu3*100/len(run_dict)), 'BLEU-4': rounder(bleu4*100/len(run_dict))}


