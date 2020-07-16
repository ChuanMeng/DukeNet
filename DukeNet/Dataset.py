from torch.utils.data import Dataset
from DukeNet.Utils import *
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class Dataset(Dataset):

    def __init__(self, mode, samples, query, passage, vocab2id, max_knowledge_pool_when_train=None, max_knowledge_pool_when_inference=None, context_len=None, knowledge_sentence_len=None, max_dec_length=None, n=1E10):  # 1e10=1E10
        super(Dataset, self).__init__()

        self.knowledge_sentence_len = knowledge_sentence_len
        self.max_knowledge_pool_when_train = max_knowledge_pool_when_train
        self.max_knowledge_pool_when_inference = max_knowledge_pool_when_inference
        self.context_len = context_len
        self.max_dec_length = max_dec_length

        self.mode = mode

        self.samples = samples
        self.query = query
        self.passage = passage

        self.answer_file = samples[0]['answer_file']

        self.vocab2id = vocab2id
        self.n = n

        self.sample_tensor = []
        self.load()


    def load(self):
        for id in range(len(self.samples)):
            sample = self.samples[id]
            id_tensor = torch.tensor([id]).long()

            if len(sample['context_id']) == 0:
                context_x = []
                context_y = []
            else:
                context_x = self.query[sample['context_id'][-2]]
                context_y = self.query[sample['context_id'][-1]]

            query = self.query[sample['query_id']]

            contexts = []
            for u in [context_x, context_y, query]:
                u_=[CLS_WORD]+u+[SEP_WORD]
                if len(u_)>self.context_len:
                    u_=[CLS_WORD] + u_[-(self.context_len-1):]
                elif len(u_)<self.context_len:
                    u_ = u_ +[PAD_WORD] * (self.context_len - len(u_))
                assert len(u_) == self.context_len
                contexts.append(u_)

            contexts_tensor = [torch.tensor([self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in c], requires_grad=False).long() for c in contexts]

            contexts_tensor = torch.stack(contexts_tensor)


            if self.mode == "train" and len(sample['tracking_knowledge_pool']) > self.max_knowledge_pool_when_train:
                keepers = 1 + np.random.choice(len(sample['tracking_knowledge_pool']) - 1, self.max_knowledge_pool_when_train, False)
                # correct answer is always the first one
                keepers[0] = 0
                temp_sample_tracking_knowledge_pool = [sample['tracking_knowledge_pool'][i] for i in keepers]
            else:
                temp_sample_tracking_knowledge_pool = sample['tracking_knowledge_pool'].copy()

            if self.mode == "train" and len(sample['shifting_knowledge_pool']) > self.max_knowledge_pool_when_train:
                keepers = 1 + np.random.choice(len(sample['shifting_knowledge_pool']) - 1, self.max_knowledge_pool_when_train, False)
                # correct answer is always the first one
                keepers[0] = 0
                temp_sample_shifting_knowledge_pool = [sample['shifting_knowledge_pool'][i] for i in keepers]
            else:
                temp_sample_shifting_knowledge_pool = sample['shifting_knowledge_pool'].copy()

            # text [[tokens],[tokens],...]
            tracking_knowledge_text_list = []
            for pid in temp_sample_tracking_knowledge_pool:
                p=[CLS_WORD]+self.passage[pid]+[SEP_WORD]
                if len(p)>self.knowledge_sentence_len:
                    p=p[:self.knowledge_sentence_len-1]+[SEP_WORD]
                elif len(p)<self.knowledge_sentence_len:
                    p = p +[PAD_WORD] * (self.knowledge_sentence_len - len(p))
                assert len(p) == self.knowledge_sentence_len
                tracking_knowledge_text_list.append(p)

            shifting_knowledge_text_list = []
            for pid in temp_sample_shifting_knowledge_pool:
                p = [CLS_WORD] + self.passage[pid] + [SEP_WORD]
                if len(p) > self.knowledge_sentence_len:
                    p = p[:self.knowledge_sentence_len - 1] + [SEP_WORD]
                elif len(p) < self.knowledge_sentence_len:
                    p = p + [PAD_WORD] * (self.knowledge_sentence_len - len(p))
                assert len(p)==self.knowledge_sentence_len
                shifting_knowledge_text_list.append(p)

            if self.mode == "train":
                max_knowledge_pool=self.max_knowledge_pool_when_train
            elif self.mode == "inference":
                max_knowledge_pool = self.max_knowledge_pool_when_inference
            else:
                Exception("no ther mode")

            tracking_ck_mask_one_example = torch.zeros(max_knowledge_pool)
            tracking_ck_mask_one_example[:len(tracking_knowledge_text_list)] = 1
            tracking_ck_mask_one_example = tracking_ck_mask_one_example == 1

            shifting_ck_mask_one_example = torch.zeros(max_knowledge_pool)
            shifting_ck_mask_one_example[:len(shifting_knowledge_text_list)] = 1
            shifting_ck_mask_one_example = shifting_ck_mask_one_example == 1

            while len(tracking_knowledge_text_list) < max_knowledge_pool:
                tracking_knowledge_text_list.append([CLS_WORD] + [SEP_WORD] + [PAD_WORD] * (self.knowledge_sentence_len - 2))

            while len(shifting_knowledge_text_list) < max_knowledge_pool:
                shifting_knowledge_text_list.append([CLS_WORD] + [SEP_WORD] + [PAD_WORD] * (self.knowledge_sentence_len - 2))

            assert len(tracking_knowledge_text_list) == len(shifting_knowledge_text_list) == max_knowledge_pool


            # index tensor: [passage1tokensidstensor,passage2tokensidstensor,passage3tokensidstensor,...]
            tracking_knowledge_tensor = [torch.tensor([self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in p], requires_grad=False).long() for p in tracking_knowledge_text_list]
            tracking_knowledge_tensor = torch.stack(tracking_knowledge_tensor)  # size(num_passage, passage_len)

            shifting_knowledge_tensor = [torch.tensor([self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in p], requires_grad=False).long() for p in shifting_knowledge_text_list]
            shifting_knowledge_tensor = torch.stack(shifting_knowledge_tensor)  # size(num_passage, passage_len)


            assert temp_sample_tracking_knowledge_pool.index(sample['tracking_knowledge_label'][0]) == 0
            assert temp_sample_shifting_knowledge_pool.index(sample['shifting_knowledge_label'][0]) == 0

            tracking_knowledge_label = [torch.tensor([temp_sample_tracking_knowledge_pool.index(pid)], requires_grad=False).long() for pid in sample['tracking_knowledge_label']]
            shifting_knowledge_label = [torch.tensor([temp_sample_shifting_knowledge_pool.index(pid)], requires_grad=False).long() for pid in sample['shifting_knowledge_label']]

            response = (sample['response']+[EOS_WORD])[:self.max_dec_length]
            response_tensor = torch.tensor([self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in response], requires_grad=False).long()

            self.sample_tensor.append(
                [id_tensor, contexts_tensor, tracking_knowledge_tensor, shifting_knowledge_tensor,
                 tracking_knowledge_label, shifting_knowledge_label, response_tensor, tracking_ck_mask_one_example,
                 shifting_ck_mask_one_example])

            self.len = id + 1

            if id >= self.n:
                break

    def __getitem__(self, index):
        sample = self.sample_tensor[index]
        return [sample[0], sample[1], sample[2], sample[3], sample[4][random.randint(0, len(sample[4]) - 1)], sample[5][random.randint(0, len(sample[5]) - 1)], sample[6],
                sample[7], sample[8]]

    def __len__(self):
        return self.len


    def context_id(self, id):
        return self.samples[id]['context_id']  # list

    def query_id(self, id):
        return self.samples[id]['query_id']  # string

    def passage_id(self, id):
        return self.samples[id]['shifting_knowledge_label']  # list

    def knowledge_pool(self, id):
        return self.samples[id]['shifting_knowledge_pool']  # list

def collate_fn(data):

    id, contexts, tracking_knowledge_pool, shifting_knowledge_pool, tracking_knowledge_label, shifting_knowledge_label, response, tracking_ck_mask_one_example, shifting_ck_mask_one_example= zip(
        *data)

    return {'id': torch.cat(id),
            'contexts': torch.stack(contexts), # (batch, 3, context_len)
            'response': pad_sequence(response, batch_first=True),  # list of tensor
            'knowledge_tracking_label': torch.cat(tracking_knowledge_label),  # batch
            'knowledge_shifting_label': torch.cat(shifting_knowledge_label),  # batch
            'knowledge_tracking_pool': torch.stack(tracking_knowledge_pool),  # (batch, num_passage, passage_len)
            'knowledge_shifting_pool': torch.stack(shifting_knowledge_pool),  # (batch, num_passage, passage_len)
            'tracking_ck_mask': torch.stack(tracking_ck_mask_one_example),  # (batch, num_passage)
            'shifting_ck_mask': torch.stack(shifting_ck_mask_one_example),  # (batch, num_passage)
    }

