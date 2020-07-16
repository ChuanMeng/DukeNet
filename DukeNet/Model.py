from DukeNet.PositionalEmbedding import *
from DukeNet.TransformerSeqEncoderDecoder import *
from DukeNet.Utils import *
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.modules.normalization import LayerNorm


class PriorKnowldgeTracker(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        # in_features – size of each input sample
        # out_features – size of each output sample
        self.project_c = nn.Linear(2*args.hidden_size, args.hidden_size)
        self.project_k = nn.Linear(args.hidden_size, args.hidden_size)

    def forward(self, contexts_encoded, knowledge_tracking_pool_encoded, knowledge_tracking_pool_mask, tracking_ck_mask, knowledge_tracking_label, mode="train"):
        context_x_use = contexts_encoded[1][:,0,:]
        context_y_use = contexts_encoded[1][:,1,:]
        context_x_y_use = torch.cat([context_x_use, context_y_use], dim=1)
        context_x_y_use_pro = self.project_c(context_x_y_use)  # (N,2E)->(N,E)
        knowledge_tracking_pool_use_pro = self.project_k(knowledge_tracking_pool_encoded[1])  # (N,K,E)->(N,K,E)

        # (N,K,E) bmm *(N,E,1) -> (N,K,1)->(N,K)
        knowledge_tracking_score = torch.bmm(knowledge_tracking_pool_use_pro, context_x_y_use_pro.unsqueeze(-1)).squeeze(-1)
        knowledge_tracking_score.masked_fill_(~tracking_ck_mask, neginf(knowledge_tracking_score.dtype))


        # if we're not given the true chosen_sentence (test time), pick our
        # best guess
        if mode == "inference":
            _, knowledge_tracking_ids = knowledge_tracking_score.max(1)  # N
        elif mode == "train":
            knowledge_tracking_ids = knowledge_tracking_label

        N, K, T, H = knowledge_tracking_pool_encoded[0].size()
        #N, K, H = knowledge_tracking_pool_encoded[1].size()
        offsets = torch.arange(N, device=knowledge_tracking_ids.device) * K + knowledge_tracking_ids  # N

        # knowledge_tracking_pool_use (N K E)->(N*K,E)
        flatten_knowledge_tracking_pool_use = knowledge_tracking_pool_encoded[1].view(N*K, -1)
        tracked_knowledge_use = flatten_knowledge_tracking_pool_use[offsets]  # (N H)

        # tracked_knowledge_encoded  (N*K,T,H)
        tracked_knowledge_encoded = knowledge_tracking_pool_encoded[0].reshape(-1, T, H)[offsets]  # (N,T,H)
        # but padding is (N * K, T)
        tracked_knowledge_mask = knowledge_tracking_pool_mask.reshape(-1, T)[offsets]  # (N,T)
        #(N, K)  (N,T,E)  (N,T)  (N E)
        return knowledge_tracking_score, tracked_knowledge_encoded, tracked_knowledge_mask, tracked_knowledge_use


class PosteriorKnowldgeTracker(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        # in_features – size of each input sample
        # out_features – size of each output sample
        self.project_c_k = nn.Linear(3*args.hidden_size, args.hidden_size)
        self.project_k = nn.Linear(args.hidden_size, args.hidden_size)

    def forward(self, contexts_encoded, shifted_knowledge_use, knowledge_tracking_pool_encoded,
                    knowledge_tracking_pool_mask, tracking_ck_mask,
                    knowledge_tracking_label, mode="train"):

        context_x_use = contexts_encoded[1][:,0,:]
        context_y_use = contexts_encoded[1][:,1,:]

        # context_x/y_use (N,E),shifted_knowledge_use (N,E)->(N,3E)
        # tile_inferred_shifted_knowledge_use (NT E)
        con_know_use = torch.cat([context_x_use, context_y_use, shifted_knowledge_use], dim=1)  # (NT,3E)
        con_know_use_pro = self.project_c_k(con_know_use)  # (N,3E)->(N,E);(NT E)
        knowledge_tracking_pool_use_pro = self.project_k(knowledge_tracking_pool_encoded[1])  # (N,K,E)->(N,K,E);(NT,K,E)

        # (N,K,E) bmm *(N,E,1) -> (N,K,1)->(N,K)
        knowledge_tracking_score = torch.bmm(knowledge_tracking_pool_use_pro, con_know_use_pro.unsqueeze(-1)).squeeze(-1)
        knowledge_tracking_score.masked_fill_(~tracking_ck_mask, neginf(knowledge_tracking_score.dtype))

        # (NT, K)
        #knowledge_tracking_prob = F.softmax(knowledge_tracking_attn, 1)

        # if we're not given the true chosen_sentence (test time), pick our
        # best guess
        if mode == "inference":
            _, knowledge_tracking_ids = knowledge_tracking_score.max(1)  # N
        elif mode == "train":
            knowledge_tracking_ids = knowledge_tracking_label

        N, K, T, H = knowledge_tracking_pool_encoded[0].size()
        offsets = torch.arange(N, device=knowledge_tracking_ids.device) * K + knowledge_tracking_ids  # N

        # knowledge_tracking_pool_use (N K E)->(N*K,E)
        flatten_knowledge_tracking_pool_use = knowledge_tracking_pool_encoded[1].view(N*K, -1)
        tracked_knowledge_use = flatten_knowledge_tracking_pool_use[offsets]  # (N E)

        # tracked_knowledge_encoded  (N*K,T,H)
        tracked_knowledge_encoded = knowledge_tracking_pool_encoded[0].reshape(-1, T, H)[offsets]  # (N,T,D)
        # but padding is (N * K, T)
        tracked_knowledge_mask = knowledge_tracking_pool_mask.reshape(-1, T)[offsets]  # (N,T)
        # (N, K)  (N,T,E)  (N,T)  (N E)
        return knowledge_tracking_score, tracked_knowledge_encoded, tracked_knowledge_mask, tracked_knowledge_use


class KnowldgeShifter(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.project_c_q_k = nn.Linear(2*args.hidden_size, args.hidden_size)
        self.project_k = nn.Linear(args.hidden_size, args.hidden_size)

    def forward(self, contexts_encoded, tracked_knowledge_use, knowledge_shifting_pool_encoded,
                    knowledge_shifting_pool_mask, shifting_ck_mask,
                    knowledge_shifting_label, knowledge_shifting_pool, mode="train"):


        context_x_use = contexts_encoded[1][:,0,:]
        context_y_use = contexts_encoded[1][:,1,:]
        query_use = contexts_encoded[1][:,2,:]

        con_query_know_use = torch.cat([query_use, tracked_knowledge_use], dim=1)
        con_query_know_use_pro = self.project_c_q_k(con_query_know_use)  # (N,2E)->(N,E)
        knowledge_shifting_pool_use_pro = self.project_k(knowledge_shifting_pool_encoded[1])  # (N,K,E)->(N,K,E)

        # (N,K,E) bmm *(N,E,1) -> (N,K,1)->(N,K)
        knowledge_shifting_score = torch.bmm(knowledge_shifting_pool_use_pro, con_query_know_use_pro.unsqueeze(-1)).squeeze(-1)
        knowledge_shifting_score.masked_fill_(~shifting_ck_mask, neginf(knowledge_shifting_score.dtype))


        # if we're not given the true chosen_sentence (test time), pick our
        # best guess
        if mode == "inference":
            _, knowledge_shifting_ids = knowledge_shifting_score.max(1)  # N
        elif mode == "train":
            knowledge_shifting_ids = knowledge_shifting_label

        N, K, T, H = knowledge_shifting_pool_encoded[0].size()
        offsets = torch.arange(N, device=knowledge_shifting_ids.device) * K + knowledge_shifting_ids

        # knowledge_tracking_pool_use (N K E)->(N*K,E)
        flatten_knowledge_shifting_pool_use = knowledge_shifting_pool_encoded[1].view(N*K, -1)
        shifted_knowledge_use = flatten_knowledge_shifting_pool_use[offsets]  # (N E)

        # tracked_knowledge_encoded  (N*K,T,H)
        shifted_knowledge_encoded = knowledge_shifting_pool_encoded[0].reshape(-1, T, H)[offsets]  # (N,T,D)
        # but padding is (N * K, T)
        shifted_knowledge_mask = knowledge_shifting_pool_mask.reshape(-1, T)[offsets]  # (N,T)
        shifted_knowledge_shifting_index = knowledge_shifting_pool.reshape(-1, T)[offsets] # (N,T)

        return knowledge_shifting_score, shifted_knowledge_encoded, shifted_knowledge_mask, shifted_knowledge_use, shifted_knowledge_shifting_index


class DukeNet(nn.Module):
    def __init__(self, vocab2id, id2vocab, args):
        super().__init__()

        self.id2vocab = id2vocab
        self.vocab_size = len(id2vocab)
        self.args = args

        self.enc = TransformerSeqEncoder(args, vocab2id, id2vocab, None)  # num_layers, num_heads, src_vocab_size, hidden_size, emb_matrix=None
        self.dec = TransformerSeqDecoder(args, vocab2id, id2vocab, self.enc.enc.embeddings)  # num_memories, num_layers, nhead, tgt_vocab_size, hidden_size, emb_matrix=None

        self.prior_tracker = PriorKnowldgeTracker(args=args)
        self.posterior_tracker = PosteriorKnowldgeTracker(args=args)
        self.shifter = KnowldgeShifter(args=args)

    def encoding_layer(self, data):

        # (batch, 3, context_len)
        contexts_mask = data['contexts'].ne(0).detach()
        knowledge_tracking_pool_mask= data['knowledge_tracking_pool'].ne(0).detach()
        knowledge_shifting_pool_mask= data['knowledge_shifting_pool'].ne(0).detach()

        contexts_encoded = self.enc(data['contexts'], contexts_mask)
        knowledge_tracking_pool_encoded = self.enc(data['knowledge_tracking_pool'], knowledge_tracking_pool_mask)
        knowledge_shifting_pool_encoded = self.enc(data['knowledge_shifting_pool'], knowledge_shifting_pool_mask)


        return {'contexts_encoded': contexts_encoded,
                'knowledge_tracking_pool_encoded': knowledge_tracking_pool_encoded,
                'knowledge_shifting_pool_encoded': knowledge_shifting_pool_encoded,
                "contexts_mask":contexts_mask,
                "knowledge_tracking_pool_mask": knowledge_tracking_pool_mask,
                "knowledge_shifting_pool_mask":knowledge_shifting_pool_mask}


    def dual_knowledge_interaction_layer(self, data, encoded_state):
        # knowledge tracking=================================================================
        tracking_ck_mask = data["tracking_ck_mask"]
        knowledge_tracking_label = data["knowledge_tracking_label"]


        prior_knowledge_tracking_score, prior_tracked_knowledge_encoded, prior_tracked_knowledge_mask, prior_tracked_knowledge_use = self.prior_tracker(
            encoded_state['contexts_encoded'], encoded_state['knowledge_tracking_pool_encoded'], encoded_state['knowledge_tracking_pool_mask'],
            tracking_ck_mask,
            knowledge_tracking_label,
            self.args.mode)

        if self.args.mode =="train":
            N, K, H = encoded_state['knowledge_shifting_pool_encoded'][1].size()
            offsets = torch.arange(N, device=data["knowledge_shifting_label"].device) * K + data["knowledge_shifting_label"]  # N
            # knowledge_shifting_pool_use (N K E)->(N*K,E)
            flatten_knowledge_shifting_pool_use = encoded_state['knowledge_shifting_pool_encoded'][1].view(N * K, -1)
            label_shifted_knowledge_use = flatten_knowledge_shifting_pool_use[offsets]  # (N E)

            # always given shifted knowledge
            posterior_knowledge_tracking_score, posterior_tracked_knowledge_encoded, posterior_tracked_knowledge_mask, posterior_tracked_knowledge_use = self.posterior_tracker(
                encoded_state['contexts_encoded'], label_shifted_knowledge_use, encoded_state['knowledge_tracking_pool_encoded'],
                encoded_state['knowledge_tracking_pool_mask'],
                tracking_ck_mask,
                knowledge_tracking_label,
                self.args.mode)

        # knowledge shifting=================================================================
        knowledge_shifting_label = data["knowledge_shifting_label"]
        knowledge_shifting_pool = data['knowledge_shifting_pool']
        shifting_ck_mask = data["shifting_ck_mask"]


        knowledge_shifting_score, shifted_knowledge_encoded, shifted_knowledge_mask, shifted_knowledge_use, shifted_knowledge_shifting_index= self.shifter(
            encoded_state['contexts_encoded'], prior_tracked_knowledge_use, encoded_state['knowledge_shifting_pool_encoded'],
            encoded_state['knowledge_shifting_pool_mask'], shifting_ck_mask, knowledge_shifting_label, knowledge_shifting_pool,
            self.args.mode)

        # get query and selected knowledge to decoder=======================================
        # finally, concatenate it all
        #query_knowledge_encoded = torch.cat([encoded_state['query_encoded'], shifted_knowledge_encoded], dim=1)  # (N,T,D) cat (N S E)->(N,T+S,D)
        #query_knowledge_mask = torch.cat([encoded_state['query_mask'], shifted_knowledge_mask], dim=1)  # (N,T+S);

        memory1 = encoded_state['contexts_encoded'][0][:,2,:,:] # [batch_size, context_len, embedding_size]
        memory2 = shifted_knowledge_encoded # [batch_size, knowledge_len, embedding_size]
        memory_mask1= encoded_state['contexts_mask'][:,2,:] # [batch_size, context_len]
        memory_mask2= shifted_knowledge_mask # [batch_size, knowledge_len]

        if self.args.mode == "train":
        # (N,S+T,D),(N,S+T),(N,K),(N,K)
            return {'memory1': memory1, 'memory2': memory2, 'memory_mask1': memory_mask1, 'memory_mask2':memory_mask2, "shifted_knowledge_shifting_index":shifted_knowledge_shifting_index,
                'prior_knowledge_tracking_score': prior_knowledge_tracking_score,
                'posterior_knowledge_tracking_score': posterior_knowledge_tracking_score,
                'knowledge_shifting_score': knowledge_shifting_score}
        else:
            return {'memory1': memory1, 'memory2': memory2, 'memory_mask1': memory_mask1, 'memory_mask2':memory_mask2, "shifted_knowledge_shifting_index":shifted_knowledge_shifting_index,
                'prior_knowledge_tracking_score': prior_knowledge_tracking_score,
                'knowledge_shifting_score': knowledge_shifting_score}


    def decoding_layer(self, data, interaction_outputs):
        # [batch, context_len+knowledge_len]-->[batch, context_len+knowledge_len, vocab_size]
        source_map = build_map(torch.cat([data["contexts"][:,2,:], interaction_outputs["shifted_knowledge_shifting_index"]], dim=1), max=self.vocab_size)

        dec_outputs, gen_outputs, extended_gen_outputs, output_indices = self.dec(
            [interaction_outputs['memory1'], interaction_outputs['memory2']], encode_masks=[interaction_outputs['memory_mask1'], interaction_outputs['memory_mask2']],
            groundtruth_index=data['response'], source_map=source_map)
        # [batch_size, max_target_length，tgt_vocab_size]
        # output_indexes [batch, max_target_length]
        return extended_gen_outputs, output_indices


    def to_sentence(self, data, batch_indices):
        return to_sentence(batch_indices, self.id2vocab)


    def do_train(self, data):
        # {'memory': memory, 'memory_mask': memory_mask, 'passage_selection': ck_attn}
        # (N,T+S,D),(N,T+S),(N,K)
        encoded_state = self.encoding_layer(data)
        interaction_outputs = self.dual_knowledge_interaction_layer(data, encoded_state)
        rg=self.decoding_layer(data, interaction_outputs)


        # This criterion combines log_softmax(1.softmax 2.logarithm.) and nll_loss in a single function.
        # data['label']

        _, pri_tracking_pred = interaction_outputs['prior_knowledge_tracking_score'].detach().max(1)
        pri_tracking_acc = (pri_tracking_pred == data['knowledge_tracking_label']).float().mean()

        _, pos_tracking_pred = interaction_outputs['posterior_knowledge_tracking_score'].detach().max(1)
        pos_tracking_acc = (pos_tracking_pred == data['knowledge_tracking_label']).float().mean()

        _, shifting_pred = interaction_outputs['knowledge_shifting_score'].detach().max(1)
        shifting_acc = (shifting_pred == data['knowledge_shifting_label']).float().mean()

        # loss  (N,K) (N)
        loss_pri_tracking = F.nll_loss(F.log_softmax(interaction_outputs['prior_knowledge_tracking_score'], -1), data['knowledge_tracking_label'].view(-1))

        loss_pos_tracking = F.nll_loss(F.log_softmax(interaction_outputs['posterior_knowledge_tracking_score'], -1), data['knowledge_tracking_label'].view(-1))

        loss_shifting = F.nll_loss(F.log_softmax(interaction_outputs['knowledge_shifting_score'], -1), data['knowledge_shifting_label'].view(-1))

        loss_g = F.nll_loss((rg[0] + 1e-8).log().reshape(-1, rg[0].size(-1)), data['response'].reshape(-1), ignore_index=0)

        return loss_pri_tracking, loss_pos_tracking, loss_shifting, loss_g, pri_tracking_acc, pos_tracking_acc, shifting_acc

    def do_inference(self, data):
        encoded_state = self.encoding_layer(data)
        interaction_outputs = self.dual_knowledge_interaction_layer(data, encoded_state)
        rg=self.decoding_layer(data, interaction_outputs)
        batch_size = data['id'].size(0)

        _, tracking_pred = interaction_outputs['prior_knowledge_tracking_score'].max(1)
        _, shifting_pred = interaction_outputs['knowledge_shifting_score'].max(1)

        tracking_acc_one_batch = (tracking_pred == data['knowledge_tracking_label']).float().sum().cpu().item()
        shifting_acc_one_batch = (shifting_pred == data['knowledge_shifting_label']).float().sum().cpu().item()

        # [batch, max_target_length]
        return rg[1], tracking_acc_one_batch, shifting_acc_one_batch, batch_size, shifting_pred

    def forward(self, data, method='train'):
        # [N, 3*context_len+num_sequences*seq_len_p, tgt_vocab_size]]
        #data['source_map'] = build_map(data['source_map'], max=self.vocab_size)
        if method == 'train':
            return self.do_train(data)
        elif method == 'inference':
            return self.do_inference(data)

