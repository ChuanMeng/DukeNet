import sys
sys.path.append('./')
from DukeNet.Dataset import *
from torch import optim
from DukeNet.CumulativeTrainer import *
import torch.backends.cudnn as cudnn
import argparse
from DukeNet.Model import *
from dataset.Utils_DukeNet import *
from transformers import get_constant_schedule
import os
import time


def train(args):
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    print("torch_version:{}".format(torch.__version__))
    print("CUDA_version:{}".format(torch.version.cuda))
    print("cudnn_version:{}".format(cudnn.version()))

    init_seed(123456)

    data_path = args.base_data_path+args.dataset+'/'

    tokenizer, vocab2id, id2vocab = bert_tokenizer()
    detokenizer = bert_detokenizer()

    print('Vocabulary size', len(vocab2id))

    if os.path.exists(data_path + 'train_DukeNet.pkl'):
        query = torch.load(data_path + 'query_DukeNet.pkl')
        train_samples = torch.load(data_path + 'train_DukeNet.pkl')
        passage = torch.load(data_path + 'passage_DukeNet.pkl')
        print("The number of train_samples:", len(train_samples))
    else:
        samples, query, passage = load_default(args.dataset, args.datasetdata_path + args.dataset + '.answer',
                                                                   data_path + args.dataset + '.passage',
                                                                   data_path + args.dataset + '.pool',
                                                                   data_path + args.dataset + '.qrel',
                                                                   data_path + args.dataset + '.query',
                                                                   tokenizer)

        if args.dataset == "wizard_of_wikipedia":
            train_samples, dev_samples, test_seen_samples, test_unseen_samples = split_data(args.dataset, data_path + args.dataset + '.split', samples)
            print("The number of test_seen_samples:", len(test_seen_samples))
            print("The number of test_unseen_samples:", len(test_unseen_samples))
            torch.save(test_seen_samples, data_path + 'test_seen_DukeNet.pkl')
            torch.save(test_unseen_samples, data_path + 'test_unseen_DukeNet.pkl')

        elif args.dataset == "holl_e":
            train_samples, dev_samples, test_samples, = split_data(args.dataset, data_path + args.dataset + '.split', samples)
            print("The number of test_samples:", len(test_samples))
            torch.save(test_samples, data_path + 'test_DukeNet.pkl')

        print("The number of train_samples:", len(train_samples))
        print("The number of dev_samples:", len(dev_samples))
        torch.save(query, data_path + 'query_DukeNet.pkl')
        torch.save(passage, data_path + 'passage_DukeNet.pkl')
        torch.save(train_samples, data_path + 'train_DukeNet.pkl')
        torch.save(dev_samples, data_path + 'dev_DukeNet.pkl')


    model = DukeNet(vocab2id, id2vocab, args)
    saved_model_path = os.path.join(args.base_output_path + args.name + "/", 'model/')

    if args.resume is True:
        print("Reading checkpoints...")

        with open(saved_model_path + "checkpoints.json", 'r', encoding='utf-8') as r:
            checkpoints = json.load(r)
        last_epoch = checkpoints["time"][-1]

        fuse_dict = torch.load(os.path.join(saved_model_path, '.'.join([str(last_epoch), 'pkl'])))
        model.load_state_dict(fuse_dict["model"])
        print('Loading success, last_epoch is {}'.format(last_epoch))


    else:
        init_params(model, "enc")
        freeze_params(model, "enc")

        last_epoch = -1

        if not os.path.exists(saved_model_path):
            os.makedirs(saved_model_path)

        with open(saved_model_path + "checkpoints.json", 'w', encoding='utf-8') as w:
            checkpoints = {"time": []}
            json.dump(checkpoints, w)

    # construct an optimizer object
    model_optimizer = optim.Adam(model.parameters(), args.lr) # model.parameters() Returns an iterator over module parameters.This is typically passed to an optimizer.
    model_scheduler = get_constant_schedule(model_optimizer)

    if args.resume is True:
        model_scheduler.load_state_dict(fuse_dict["scheduler"])
        print('Loading scheduler, last_scheduler is', fuse_dict["scheduler"])

    trainer = CumulativeTrainer(args.name, model, tokenizer, detokenizer, args.local_rank, accumulation_steps=args.accumulation_steps)
    model_optimizer.zero_grad()  # Clears the gradients of all optimized torch.Tensor s.

    for i in range(last_epoch+1, args.epoches):
        if i==5:
            unfreeze_params(model, "enc")
            args.train_batch_size = 2
            args.accumulation_steps = 16

        train_dataset = Dataset(args.mode, train_samples, query, passage, vocab2id,
                                    args.max_knowledge_pool_when_train, args.max_knowledge_pool_when_inference, args.context_len, args.knowledge_sentence_len,
                                    args.max_dec_length)
        trainer.train_epoch('train', train_dataset, collate_fn, args.train_batch_size, i, model_optimizer, model_scheduler)
        del train_dataset

        trainer.serialize(i, model_scheduler, saved_model_path=saved_model_path)



def inference(args):
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    print("torch_version:{}".format(torch.__version__))
    print("CUDA_version:{}".format(torch.version.cuda))
    print("cudnn_version:{}".format(cudnn.version()))

    init_seed(123456)

    data_path = args.base_data_path + args.dataset + '/'


    tokenizer, vocab2id, id2vocab = bert_tokenizer()
    detokenizer = bert_detokenizer()

    print('Vocabulary size', len(vocab2id))

    if os.path.exists(data_path + 'dev_DukeNet.pkl'):
        query = torch.load(data_path + 'query_DukeNet.pkl')
        passage = torch.load(data_path + 'passage_DukeNet.pkl')
        dev_samples = torch.load(data_path + 'dev_DukeNet.pkl')
        print("The number of dev_samples:", len(dev_samples))

        if args.dataset == "wizard_of_wikipedia":
            test_seen_samples = torch.load(data_path + 'test_seen_DukeNet.pkl')
            test_unseen_samples = torch.load(data_path + 'test_unseen_DukeNet.pkl')
            print("The number of test_seen_samples:", len(test_seen_samples))
            print("The number of test_unseen_samples:", len(test_unseen_samples))
        elif args.dataset == "holl_e":
            test_samples = torch.load(data_path + 'test_DukeNet.pkl')
            print("The number of test_samples:", len(test_samples))


    else:
        samples, query, passage = load_default(args.dataset, data_path + args.dataset + '.answer',
                                                                   data_path + args.dataset + '.passage',
                                                                   data_path + args.dataset + '.pool',
                                                                   data_path + args.dataset + '.qrel',
                                                                   data_path + args.dataset + '.query',
                                                                   tokenizer)

        if args.dataset == "wizard_of_wikipedia":
            train_samples, dev_samples, test_seen_samples, test_unseen_samples = split_data(args.dataset, data_path + args.dataset + '.split', samples)
            print("The number of test_seen_samples:", len(test_seen_samples))
            print("The number of test_unseen_samples:", len(test_unseen_samples))
            torch.save(test_seen_samples, data_path + 'test_seen_DukeNet.pkl')
            torch.save(test_unseen_samples, data_path + 'test_unseen_DukeNet.pkl')

        elif args.dataset == "holl_e":
            train_samples, dev_samples, test_samples, = split_data(args.dataset, data_path + args.dataset + '.split', samples)
            print("The number of test_samples:", len(test_samples))
            torch.save(test_samples, data_path + 'test_DukeNet.pkl')

        print("The number of train_samples:", len(train_samples))
        print("The number of dev_samples:", len(dev_samples))
        torch.save(query, data_path + 'query_DukeNet.pkl')
        torch.save(passage, data_path + 'passage_DukeNet.pkl')
        torch.save(train_samples, data_path + 'train_DukeNet.pkl')
        torch.save(dev_samples, data_path + 'dev_DukeNet.pkl')


    if args.dataset == "wizard_of_wikipedia":
        dev_dataset = Dataset(args.mode, dev_samples, query, passage, vocab2id, args.max_knowledge_pool_when_train, args.max_knowledge_pool_when_inference,
                                args.context_len, args.knowledge_sentence_len, args.max_dec_length)

        test_seen_dataset = Dataset(args.mode, test_seen_samples, query, passage, vocab2id, args.max_knowledge_pool_when_train, args.max_knowledge_pool_when_inference,
                                args.context_len, args.knowledge_sentence_len, args.max_dec_length)

        test_unseen_dataset = Dataset(args.mode, test_unseen_samples, query, passage, vocab2id, args.max_knowledge_pool_when_train, args.max_knowledge_pool_when_inference,
                                 args.context_len, args.knowledge_sentence_len, args.max_dec_length)

    elif args.dataset == "holl_e":
        test_dataset = Dataset(args.mode, test_samples, query, passage, vocab2id, args.max_knowledge_pool_when_train,
                               args.max_knowledge_pool_when_inference,
                               args.context_len, args.knowledge_sentence_len, args.max_dec_length)

    saved_model_path = os.path.join(args.base_output_path + args.name + "/", 'model/')


    def inference(dataset, epoch=None):
        file =saved_model_path + str(epoch) + '.pkl'
        if os.path.exists(file):
            model = DukeNet(vocab2id, id2vocab, args)

            model.load_state_dict(torch.load(file)["model"])
            trainer = CumulativeTrainer(args.name, model, tokenizer, detokenizer, None)

            if dataset == "wizard_of_wikipedia":
                print('inference {}'.format("dev_dataset"))
                trainer.test('inference', dev_dataset, collate_fn, args.inference_batch_size, 'dev', str(epoch), output_path=args.base_output_path + args.name+"/")
                print('inference {}'.format("test_seen_dataset"))
                trainer.test('inference', test_seen_dataset, collate_fn, args.inference_batch_size, 'test_seen', str(epoch), output_path=args.base_output_path + args.name+"/")
                print('inference {}'.format("test_unseen_dataset"))
                trainer.test('inference', test_unseen_dataset, collate_fn, args.inference_batch_size, 'test_unseen', str(epoch), output_path=args.base_output_path + args.name+"/")

            elif dataset == "holl_e":
                print('inference {}'.format("test_dataset"))
                trainer.test('inference', test_dataset, collate_fn, args.inference_batch_size, 'test', str(epoch), output_path=args.base_output_path + args.name+"/")


    if not os.path.exists(saved_model_path+"finished_inference.json"):
        finished_inference = {"time": []}
        w = open(saved_model_path+"finished_inference.json", 'w', encoding='utf-8')
        json.dump(finished_inference, w)
        w.close()

    if args.appoint_epoch != -1:
        print('Start inference at epoch', args.appoint_epoch)
        inference(args.dataset, args.appoint_epoch)

        r = open(saved_model_path+"finished_inference.json", 'r', encoding='utf-8')
        finished_inference = json.load(r)
        r.close()

        finished_inference["time"].append(args.appoint_epoch)
        w = open(saved_model_path + "finished_inference.json", 'w', encoding='utf-8')
        json.dump(finished_inference, w)
        w.close()
        print("finished epoch {} inference".format(args.appoint_epoch))
        exit()

    while True:
        with open(saved_model_path + "checkpoints.json", 'r', encoding='utf-8') as r:
            checkpoints = json.load(r)

        r = open(saved_model_path + "finished_inference.json", 'r', encoding='utf-8')
        finished_inference = json.load(r)
        r.close()

        if len(checkpoints["time"]) == 0:
            print('Inference_mode: wait train finish the first epoch...')
            time.sleep(300)
        else:
            for i in checkpoints["time"]:  # i is the index of epoch
                if i in finished_inference["time"]:
                    print("epoch {} already has been inferenced, skip it".format(i))
                    pass
                else:
                    print('Start inference at epoch', i)
                    inference(args.dataset, i)

                    r = open(saved_model_path + "finished_inference.json", 'r', encoding='utf-8')
                    finished_inference = json.load(r)
                    r.close()

                    finished_inference["time"].append(i)

                    w = open(saved_model_path+"finished_inference.json", 'w', encoding='utf-8')
                    json.dump(finished_inference, w)
                    w.close()
                    print("finished epoch {} inference".format(i))

            print("Inference_mode: current all model checkpoints are completed...")
            print("Inference_mode: finished %d modes" % len(finished_inference["time"]))
            if len(finished_inference["time"]) == args.epoches:
                print("All inference is ended")
                break
            else:
                print('Inference_mode: wait train finish the next epoch...')
                time.sleep(300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)

    parser.add_argument("--name", type=str, default='DukeNet')
    parser.add_argument("--base_output_path", type=str, default='output/')
    parser.add_argument("--base_data_path", type=str, default='datasets/')
    parser.add_argument("--dataset", type=str, default='wizard_of_wikipedia')
    parser.add_argument("--GPU", type=int, default=0)

    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--epoches", type=int, default=10)
    parser.add_argument("--accumulation_steps", type=int, default=1) # with BERT should increase
    parser.add_argument("--lr", type=float, default=0.00002)
    parser.add_argument("--train_batch_size", type=int, default=64) # with BERT should reduce
    parser.add_argument("--inference_batch_size", type=int, default=25)
    parser.add_argument("--appoint_epoch", type=int, default=-1)

    parser.add_argument("--context_len", type=int, default=50)
    parser.add_argument("--knowledge_sentence_len", type=int, default=34)
    parser.add_argument("--max_knowledge_pool_when_train", type=int, default=32)
    parser.add_argument("--max_knowledge_pool_when_inference", type=int, default=128)
    parser.add_argument("--max_dec_length", type=int, default=61)

    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embedding_dropout", type=float, default=0.1)
    parser.add_argument("--embedding_size", type=int, default=768)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=5)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--ffn_size", type=int, default=768)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    if args.mode == 'inference':
        inference(args)
    elif args.mode == 'train':
        train(args)
    else:
        Exception("no ther mode")