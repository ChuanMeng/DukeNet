import sys
sys.path.append('./')
from torch.utils.data import DataLoader
from evaluation.Eval_Rouge import *
from evaluation.Eval_Bleu import *
from evaluation.Eval_Meteor import *
from evaluation.Eval_F1 import *
from evaluation.Eval_Distinct import *
from torch.utils.data.distributed import DistributedSampler
from DukeNet.Utils import *
import json
import os


def rounder(num, places):
    return round(num, places)


def train_embedding(model):
    for name, param in model.named_parameters():
        if 'embedding' in name:
            print('requires_grad', name, param.size())
            param.requires_grad = True


def init_params(model, escape=None):
    for name, param in model.named_parameters():  # (string, Parameter) – Tuple containing the name and parameter
        if escape is not None and escape in name:
            print('no_init', name, param.size())
            continue
        print('init', name, param.size())
        if param.data.dim() > 1:
            xavier_uniform_(param.data)


def freeze_params(model, freeze=None):
    for name, param in model.named_parameters():  # (string, Parameter) – Tuple containing the name and parameter
        if freeze is not None and freeze in name:
            param.requires_grad = False
            print('freeze_params', name, param.size())


def unfreeze_params(model, unfreeze=None):
    for name, param in model.named_parameters():  # (string, Parameter) – Tuple containing the name and parameter
        if unfreeze is not None and unfreeze in name:
            param.requires_grad = True
            print('unfreeze_params', name, param.size())


class CumulativeTrainer(object):
    def __init__(self, name, model, tokenizer, detokenizer, local_rank=None, accumulation_steps=None):
        super(CumulativeTrainer, self).__init__()
        self.local_rank = local_rank
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer
        self.name = name


        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model

        self.eval_model = self.model

        self.accumulation_steps = accumulation_steps
        self.accumulation_count = 0


    def train_batch(self, epoch, data, method, optimizer, scheduler=None):
        self.accumulation_count += 1
        loss_pri_tracking, loss_pos_tracking, loss_shifting, loss_g, pri_tracking_acc, pos_tracking_acc, shifting_acc = self.model(data, method=method)


        loss = loss_pri_tracking+ loss_pos_tracking + loss_shifting + loss_g
        loss = loss/self.accumulation_steps

        loss.backward()

        if self.accumulation_count % self.accumulation_steps == 0:
            # The norm is computed over all gradients together,
            # as if they were concatenated into a single vector.
            # return is Total norm of the parameters (viewed as a single vector).

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.4)


            # torch.optim.Adam.step()
            # Performs a single optimization step.
            optimizer.step()
            if scheduler is not None:
                # Learning rate scheduling should be applied after optimizer’s update
                scheduler.step()
            optimizer.zero_grad()

        return loss_pri_tracking.cpu().item(), loss_pos_tracking.cpu().item(), loss_shifting.cpu().item(), \
               loss_g.cpu().item(), \
               pri_tracking_acc.cpu().item(), pos_tracking_acc.cpu().item(), shifting_acc.cpu().item()

    def serialize(self, epoch, scheduler, saved_model_path):

        fuse_dict = {"model": self.eval_model.state_dict(), "scheduler": scheduler.state_dict()}

        torch.save(fuse_dict, os.path.join(saved_model_path, '.'.join([str(epoch), 'pkl'])))
        print("Saved epoch {} model".format(epoch))

        with open(saved_model_path + "checkpoints.json", 'r', encoding='utf-8') as r:
            checkpoints = json.load(r)

        checkpoints["time"].append(epoch)

        with open(saved_model_path + "checkpoints.json", 'w', encoding='utf-8') as w:
            json.dump(checkpoints, w)


    def train_epoch(self, method, train_dataset, train_collate_fn, batch_size, epoch, optimizer, scheduler=None):
        self.model.train()  # Sets the module in training mode；
        if torch.cuda.is_available():
            train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=batch_size, shuffle=True)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=batch_size, shuffle=True)

        start_time = time.time()
        count_batch = 0


        accu_loss_pri_tracking = 0.
        accu_loss_pos_tracking = 0.
        accu_loss_shifting = 0.
        accu_loss_g = 0.
        accu_pri_tracking_acc = 0.
        accu_pos_tracking_acc = 0.
        accu_shifting_acc = 0.
        step = 0


        for j, data in enumerate(train_loader, 0):
            if torch.cuda.is_available():
                data_cuda = dict()
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data_cuda[key] = value.cuda()
                    else:
                        data_cuda[key] = value
                data = data_cuda
            count_batch += 1

            loss_pri_tracking, loss_pos_tracking, loss_shifting, loss_g, pri_tracking_acc, pos_tracking_acc, shifting_acc = self.train_batch(epoch, data, method=method, optimizer=optimizer, scheduler=scheduler)

            accu_loss_pri_tracking += loss_pri_tracking
            accu_loss_pos_tracking += loss_pos_tracking
            accu_loss_shifting += loss_shifting
            accu_loss_g += loss_g

            accu_pri_tracking_acc += pri_tracking_acc
            accu_pos_tracking_acc += pos_tracking_acc
            accu_shifting_acc += shifting_acc
            step +=1

            if j >= 0 and j % 100 == 0:
                elapsed_time = time.time() - start_time
                print('Training: %s' % self.name)
                #print(scheduler.state_dict())
                if scheduler is not None:
                    # Calculates the learning rate at batch index.
                    print(
                        'Epoch:{}, Step:{}, Batch:{}, Pri_Tra_Loss:{}, Pos_Tra_Loss:{}, Shi_Loss:{}, Gen_Loss:{}, Pri_Tra_ACC:{}, Pos_Tra_ACC:{}, Shi_ACC:{}, Time:{}, LR:{}'.format(
                            epoch, scheduler.state_dict()['_step_count'], count_batch,
                            rounder(accu_loss_pri_tracking / step, 4),
                            rounder(accu_loss_pos_tracking / step, 4),
                            rounder(accu_loss_shifting / step, 4),
                            rounder(accu_loss_g / step, 4),
                            rounder(accu_pri_tracking_acc / step, 4),
                            rounder(accu_pos_tracking_acc / step, 4),
                            rounder(accu_shifting_acc/ step, 4),
                            rounder(elapsed_time, 2),
                            scheduler.get_lr()))
                else:
                    print(
                        'Epoch:{}, Step:{}, Batch:{}, Pri_Tra_Loss:{}, Pos_Tra_Loss:{}, Shi_Loss:{}, Gen_Loss:{}, Pri_Tra_ACC:{}, Pos_Tra_ACC:{}, Shi_ACC:{}, Time:{}'.format(
                            epoch, scheduler.state_dict()['_step_count'], count_batch,
                            rounder(accu_loss_pri_tracking / step, 4),
                            rounder(accu_loss_pos_tracking / step, 4),
                            rounder(accu_loss_shifting / step, 4),
                            rounder(accu_loss_g / step, 4),
                            rounder(accu_pri_tracking_acc / step, 4),
                            rounder(accu_pos_tracking_acc / step, 4),
                            rounder(accu_shifting_acc / step, 4),
                            rounder(elapsed_time, 2)))

                accu_loss_pri_tracking = 0.
                accu_loss_pos_tracking = 0.
                accu_loss_shifting = 0.
                accu_loss_g = 0.
                accu_pri_tracking_acc = 0.
                accu_pos_tracking_acc = 0.
                accu_shifting_acc = 0.
                step = 0


                sys.stdout.flush()

            del loss_pri_tracking
            del loss_pos_tracking
            del loss_shifting
            del loss_g
            del pri_tracking_acc
            del pos_tracking_acc
            del shifting_acc


        sys.stdout.flush()

    def predict(self, method, dataset, collate_fn, batch_size, epoch, output_path):
        #  changes the forward() behaviour of the module it is called upon. eg, it disables dropout and has batch norm use the entire population statistics
        self.eval_model.eval()

        with torch.no_grad():
            test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

            accumulative_tracking_acc = 0
            accumulative_shifting_acc = 0
            accumulative_example = 0
            systems = []
            ref_path=None
            for k, data in enumerate(test_loader, 0):
                print("doing {} / total {} in {}".format(k+1, len(test_loader), epoch))
                if torch.cuda.is_available():
                    data_cuda=dict()
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            data_cuda[key] = value.cuda()
                        else:
                            data_cuda[key] = value
                    data = data_cuda

                indices, tracking_acc_one_batch, shifting_acc_one_batch, batch_size, shifting_pred_one_batch = self.eval_model(data, method=method)  # [batch, max_decoder_len]
                #indices, tracking_acc_one_batch, shifting_acc_one_batch, batch_size = self.eval_model(data, method=method)  # [batch, max_decoder_len]
                sents = self.eval_model.to_sentence(data, indices)  # [[tokens],[tokens]...batch个]

                accumulative_tracking_acc += tracking_acc_one_batch
                accumulative_shifting_acc += shifting_acc_one_batch
                accumulative_example += batch_size


                for i in range(len(data['id'])):
                    id = data['id'][i].item()
                    shifting_pred_one_example = shifting_pred_one_batch[i].item()
                    #systems.append([';'.join(dataset.context_id(id)), dataset.query_id(id), ';'.join(dataset.passage_id(id)), self.detokenizer(sents[i])])
                    systems.append([';'.join(dataset.context_id(id)), dataset.query_id(id), dataset.knowledge_pool(id)[shifting_pred_one_example], self.detokenizer(sents[i])])

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            output_path = os.path.join(output_path, str(epoch) + '.txt')

            file = codecs.open(output_path, "w", "utf-8")
            for i in range(len(systems)):
                file.write('\t'.join(systems[i])+os.linesep)
            file.close()
        return output_path, dataset.answer_file, {"t_acc": rounder(100*(accumulative_tracking_acc/accumulative_example), 2)}, {"s_acc": rounder(100*(accumulative_shifting_acc/accumulative_example), 2)}

    def test(self, method, dataset, collate_fn, batch_size, dataset_name, epoch, output_path):
        #  disables tracking of gradients in autograd.
        # In this mode, the result of every computation will have requires_grad=False, even when the inputs have requires_grad=True.
        with torch.no_grad():
            run_file, answer_file, tracking_acc, shifting_acc = self.predict(method, dataset, collate_fn, batch_size, dataset_name+"_"+epoch, output_path)

        print("Start auotimatic evaluation")

        print("TRACKING_KNOW_ACC", tracking_acc)
        print("SHIFTING_KNOW_ACC", shifting_acc)

        f1 = eval_f1_file(run_file, answer_file, self.tokenizer, self.detokenizer)
        print("F1", f1)

        bleus = eval_bleu_file(run_file, answer_file, self.tokenizer, self.detokenizer)
        print("BLEU", bleus)

        rouges = eval_rouge_file(run_file, answer_file, self.tokenizer, self.detokenizer)
        print("ROUGE", rouges)

        meteors = eval_meteor_file(run_file, answer_file, self.tokenizer, self.detokenizer)
        print("METEOR", meteors)

        distinct = eval_distinct_file(run_file, answer_file, self.tokenizer, self.detokenizer)
        print("DISTINCT", distinct)

        metric_output = {**f1, **bleus, **rouges, **distinct, **meteors, **tracking_acc, **shifting_acc}
        print({epoch+"_"+dataset_name: metric_output})

        try:
            with open(os.path.join(output_path, dataset_name + "_result.json"), 'r', encoding='utf-8') as r:
                result_log = json.load(r)
            result_log[epoch + "_" + dataset_name] = metric_output
            with open(os.path.join(output_path, dataset_name + "_result.json"), 'w', encoding='utf-8') as w:
                json.dump(result_log, w)

        except FileNotFoundError:
            with open(os.path.join(output_path, dataset_name + "_result.json"), 'w', encoding='utf-8') as w:
                result_log={}
                result_log[epoch+"_"+dataset_name] = metric_output
                json.dump(result_log, w)

        return None


if __name__ == '__main__':

    from transformers import BertTokenizer
    from evaluation.Eval_Multi_acc import *

    def bert_tokenizer():
        t = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True)  # do_lower_case Whether to lower case the input.
        return t.tokenize, t.vocab, t.ids_to_tokens

    def bert_detokenizer():
        def detokenizer(tokens):
            return ' '.join(tokens).replace(' ##', '').strip()
        return detokenizer


    tokenizer, vocab2id, id2vocab = bert_tokenizer()
    detokenizer = bert_detokenizer()

    dataset_name="test"
    run_file = "output/DukeNet_Holl_E/test_d_4.txt"
    answer_file ="datasets/holl_e/holl_e.multi_answer"


    multi_acc = eval_multi_acc_file(run_file, answer_file)
    print(multi_acc)


    f1 = eval_f1_file(run_file, answer_file, tokenizer, detokenizer)
    print("F1", f1)

    bleus = eval_bleu_file(run_file, answer_file, tokenizer, detokenizer)
    print("BLEU", bleus)

    rouges = eval_rouge_file(run_file, answer_file, tokenizer, detokenizer)
    print("ROUGE", rouges)

    meteors = eval_meteor_file(run_file, answer_file, tokenizer, detokenizer)
    print("METEOR", meteors)

    distinct = eval_distinct_file(run_file, answer_file, tokenizer, detokenizer)
    print("DISTINCT", distinct)

