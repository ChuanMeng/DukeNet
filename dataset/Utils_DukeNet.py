import sys
import os
import nltk
import codecs
from sys import *
import random
from transformers import BertTokenizer
from tqdm import tqdm

def bert_tokenizer():
    t = BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case=True)  # do_lower_case Whether to lower case the input.
    return t.tokenize, t.vocab, t.ids_to_tokens


def bert_detokenizer():
    def detokenizer(tokens):
        return ' '.join(tokens).replace(' ##', '').strip()
    return detokenizer


def nltk_tokenizer():
    def tokenizer(sent):
        return nltk.word_tokenize(sent.lower())
    return tokenizer


def nltk_detokenizer():
    def detokenizer(tokens):
        return ' '.join(tokens)
    return detokenizer


def load_answer(file, tokenizer):
    print("load_answer")
    answer = []
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t', 3)

            assert len(temp) == 4,"all_previous_query_id;all_previous_query_id;all_previous_query_id	current_query_id	background_id;background_id 	response_content"
            if len(temp[0]) < 1:
                temp[0] = []
            else:
                temp[0] = temp[0].split(';')
            temp[2] = temp[2].split(';')
            temp[3] = tokenizer(temp[3])
            answer.append(temp)
    return answer


def load_passage(file, pool, tokenizer):  # background_id	background_content
    print("load_passage")
    poolset = set()
    for k in pool:
        poolset.update(pool[k])

    passage = dict()
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t', 1)
            assert len(temp) == 2, "load_passage"
            if temp[0] in poolset:
                passage[temp[0]] = ' [SEP] '.join([' '.join(tokenizer(sent)) for sent in nltk.sent_tokenize(temp[1])]).split(' ')  # list的形式
    print("passage:{}, poolset:{}".format(len(passage), len(poolset)))
    return passage  # {background_id1:background_content, background_id2:background_content}


def load_pool(file, topk=None):  # current_query_id Q0 background_id rank relevance_score model_name
    print("load_pool")
    pool = {}
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split(' ')
            assert len(temp) == 6, "load_pool"
            if temp[0] not in pool:
                pool[temp[0]] = [temp[2]]  # {“current_query_id”:[background_id1]}
            else:
                pool[temp[0]].append(temp[2])  # {“current_query_id”:[background_id1,background_id2,background_id3...]}
    return pool


def load_qrel(file):
    print("load_qrel")
    qrel = dict()
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split(' ')
            assert len(temp) == 4, "load_qrel"
            if int(temp[3]) > 0:
                qrel[temp[0]] = temp[2]  # {current_query_id:background_id1, current_query_id2:background_id2........}
    return qrel


def load_query(file, tokenizer):  # query_id	query_content
    print("load_query")
    query = dict()
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t',1)
            assert len(temp) == 2, "load_query"
            query[temp[0]] = tokenizer(temp[1])  # {1_1:[query_tokens],}
    return query


def load_split(dataset, file):
    train = set()
    dev = set()
    if dataset == "wizard_of_wikipedia":
        test_seen = set()
        test_unseen = set()
        with codecs.open(file, encoding='utf-8') as f:
            for line in f:
                temp = line.strip('\n').strip('\r').split('\t')
                assert len(temp) == 2, "query_id train/dev/test_seen/test_unseen"
                if temp[1] == 'train':
                    train.add(temp[0])
                elif temp[1] == 'dev':
                    dev.add(temp[0])
                elif temp[1] == 'test_seen':
                    test_seen.add(temp[0])
                elif temp[1] == 'test_unseen':
                    test_unseen.add(temp[0])

        return train, dev, test_seen, test_unseen

    elif dataset == "holl_e":
        test = set()
        with codecs.open(file, encoding='utf-8') as f:
            for line in f:
                temp = line.strip('\n').strip('\r').split('\t')
                assert len(temp) == 2, "query_id train/dev/test"
                if temp[1] == 'train':
                    train.add(temp[0])
                elif temp[1] == 'dev':
                    dev.add(temp[0])
                elif temp[1] == 'test':
                    test.add(temp[0])
        return train, dev, test

def split_data(dataset, split_file, samples):
    print("split_data:", dataset)
    train_samples = list()
    dev_samples = list()

    if dataset == "wizard_of_wikipedia":
        train, dev, test_seen, test_unseen = load_split(dataset, split_file)
        test_seen_samples = list()
        test_unseen_samples = list()
        for sample in samples:
            if sample['query_id'] in train:
                train_samples.append(sample)
            elif sample['query_id'] in dev:
                dev_samples.append(sample)
            elif sample['query_id'] in test_seen:
                test_seen_samples.append(sample)
            elif sample['query_id'] in test_unseen:
                test_unseen_samples.append(sample)
        return train_samples, dev_samples, test_seen_samples, test_unseen_samples

    elif dataset == "holl_e":
        train, dev, test = load_split(dataset, split_file)
        test_samples = list()
        for sample in samples:
            if sample['query_id'] in train:
                train_samples.append(sample)
            elif sample['query_id'] in dev:
                dev_samples.append(sample)
            elif sample['query_id'] in test:
                test_samples.append(sample)
        return train_samples, dev_samples, test_samples


def load_default(dataset, answer_file, passage_file, pool_file, qrel_file, query_file, tokenizer, topk=None, randoms=1):
    random.seed(1)
    answer = load_answer(answer_file, tokenizer)  # [[all_previous_query_ids],current_query_id,[background_ids],[response_tokens]]
    pool = load_pool(pool_file, topk)  # {“current_query_id1”:[background_id1,background_id2,background_id3...]，“current_query_id2”:[background_id1,background_id2,background_id3...]}
    query = load_query(query_file, tokenizer)  # {current_query_id_1:[query_tokens],current_query_id_2:[query_tokens]}
    passage = load_passage(passage_file, pool, tokenizer)  # {background_id1:[background_tokens], [background_id2:[background_tokens]}
    average_pool = 0
    samples = []
    for i in tqdm(range(len(answer))):
        for j in range(randoms):
            c_id, q_id, knowledge_shifting_id, response = answer[i]  # c_id is a lis，q_id is string，p_id is a list，ans is a list

            if len(c_id) == 0:
                if dataset == "wizard_of_wikipedia":
                    knowledge_tracking_pool = ["K_0"]
                    knowledge_tracking_id = ["K_0"]
                elif dataset == "holl_e":
                    knowledge_tracking_pool = ["k_2872"]
                    knowledge_tracking_id = ["k_2872"]

            else:
                previous_c_id, previous_q_id, knowledge_tracking_id, previous_response = answer[i-1]
                knowledge_tracking_pool = pool[previous_q_id]

                for p in knowledge_tracking_id:  # label knowledge sentence id
                    if p not in knowledge_tracking_pool:
                        raise Exception("label tracking knowledge is not in tracking knowledge pool")

                j = knowledge_tracking_pool.index(knowledge_tracking_id[0])
                if j == 0:
                    pass
                else:
                    knowledge_tracking_pool[0], knowledge_tracking_pool[j] = knowledge_tracking_pool[j], knowledge_tracking_pool[0]


            knowledge_shifting_pool = pool[q_id]

            average_pool += len(knowledge_shifting_pool)

            for p in knowledge_shifting_id:  # label knowledge sentence id
                if p not in knowledge_shifting_pool:
                    raise Exception("label shifting knowledge is not in knowledge shifting pool")

            # we want the correct knowledge to always be in index 0
            i = knowledge_shifting_pool.index(knowledge_shifting_id[0])
            if i == 0:
                pass
            else:
                knowledge_shifting_pool[0], knowledge_shifting_pool[i] = knowledge_shifting_pool[i], knowledge_shifting_pool[0]

            sample = dict()
            sample['context_id'] = c_id  # list ：[previous utterance]
            sample['query_id'] = q_id  # string ：current query
            sample['response'] = response  # list

            sample['tracking_knowledge_pool'] = knowledge_tracking_pool  # list
            sample['shifting_knowledge_pool'] = knowledge_shifting_pool  # list

            sample['tracking_knowledge_label'] = knowledge_tracking_id  # list
            sample['shifting_knowledge_label'] = knowledge_shifting_id


            sample['answer_file'] = answer_file
            sample['passage_file'] = passage_file
            sample['pool_file'] = pool_file
            sample['query_file'] = query_file

            samples.append(sample)  # [{example1},{example2},{example3}...]

    print("average knowledge pool:", average_pool/len(samples))
    print('total eamples:', len(samples))

    return samples, query, passage











