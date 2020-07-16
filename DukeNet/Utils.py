from DukeNet.Constants import *
from torch.nn.init import *
import pickle
import bcolz
import torch.nn as nn
import numpy as np
import random
import time
import codecs
from torch.distributions.categorical import *


NEAR_INF = 1e20
NEAR_INF_FP16 = 65504

def neginf(dtype):
    """Return a representable finite number near -inf for a dtype."""
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF

def universal_sentence_embedding(sentences, mask, sqrt=True):
    '''
    :param sentences: [batch_size, seq_len, hidden_size]
    :param mask: [batch_size, seq_len]
    :param sqrt:
    :return: [batch_size, hidden_size]
    '''
    # need to mask out the padded chars
    sentence_sums = torch.bmm(
        sentences.permute(0, 2, 1), mask.float().unsqueeze(-1)
    ).squeeze(-1)
    divisor = (mask.sum(dim=1).view(-1, 1).float())
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums /= divisor
    return sentence_sums


def _get_embtype(emb_type):
    # set up preinitialized embeddings
    try:
        import torchtext.vocab as vocab
    except ImportError as ex:
        print('Please install torch text with `pip install torchtext`')
        raise ex

    if emb_type.startswith('fasttext'):
        init = 'fasttext'
        embs = vocab.Vectors(name='./preinitialized_embeddings/wiki.en.vec', cache='./preinitialized_embeddings/')
    else:
        raise RuntimeError(
            'embedding type {} not implemented. check arg, '
            'submit PR to this function, or override it.'
            ''.format(emb_type)
        )
    return embs, init


def _project_vec(vec, target_dim, method='random'):
    """
    If needed, project vector to target dimensionality.

    Projection methods implemented are the following:

    random - random gaussian matrix multiplication of input vector

    :param vec:
        one-dimensional vector

    :param target_dim:
        dimension of returned vector

    :param method:
        projection method. will be used even if the dim is not changing if
        method ends in "-force".
    """
    pre_dim = vec.size(0)
    if pre_dim != target_dim or method.endswith('force'):
        if method.startswith('random'):
            # random projection

            # Fills self tensor with elements samples from the normal distribution parameterized by mean and std.
            proj_rp = torch.Tensor(pre_dim, target_dim).normal_()
            # rescale so we're not destroying norms too much
            # http://scikit-learn.org/stable/modules/random_projection.html#gaussian-random-projection
            proj_rp /= target_dim

            return torch.mm(vec.unsqueeze(0), proj_rp)
        else:
            # TODO: PCA
            # TODO: PCA + RP
            # TODO: copy
            raise RuntimeError(
                'Projection method not implemented: {}' ''.format(method)
            )
    else:
        return vec


def _copy_embeddings(dict, weight, emb_type, log=True):
    """
    Copy embeddings from the pretrained embeddings to the lookuptable.

    :param weight:
        weights of lookup table (nn.Embedding/nn.EmbeddingBag)

    :param emb_type:
        pretrained embedding type
    """
    #if not is_primary_worker():
        # we're in distributed mode, copying embeddings in the workers
        # slows things down considerably
        #return
    embs, name = _get_embtype(emb_type)
    cnt = 0
    for w, i in dict.items():
        if w in embs.stoi:
            vec = _project_vec(embs.vectors[embs.stoi[w]], weight.size(1))
            weight.data[i] = vec
            cnt += 1

    if log:
        print(
            'Initialized embeddings for {} tokens ({}%) from {}.'
            ''.format(cnt, round(cnt * 100 / len(dict), 1), name)
        )


def get_ms():
    return time.time() * 1000


def init_seed(seed=None):
    if seed is None:
        seed = int(get_ms() // 1000)

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)  # Sets the seed for generating random numbers. Returns a torch.Generator object.
    torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.insufficient to get determinism
    torch.cuda.manual_seed_all(seed)  # Sets the seed for generating random numbers on all GPUs.


def importance_sampling(prob, topk):
    m = Categorical(logits=prob)
    indices = m.sample((topk,)).transpose(0, 1)  # batch, topk

    values = prob.gather(1, indices)
    return values, indices


def decode_to_end(model, data, vocab2id, max_target_length=None, schedule_rate=1, softmax=False, encode_outputs=None, init_decoder_states=None, tgt=None):
    # if tgt is None:
    # tgt = data['output']
    batch_size = len(data['id'])
    if max_target_length is None:
        max_target_length = tgt.size(1)

    if encode_outputs is None:
        encode_outputs = model.encode(data)
    if init_decoder_states is None:
        init_decoder_states = model.init_decoder_states(data, encode_outputs)

    decoder_input = new_tensor([vocab2id[BOS_WORD]] * batch_size, requires_grad=False)

    prob = torch.ones((batch_size,)) * schedule_rate
    if torch.cuda.is_available():
        prob = prob.cuda()

    all_gen_outputs = list()
    all_decode_outputs = [dict({'state': init_decoder_states})]

    for t in range(max_target_length):
        # decoder_outputs, decoder_states,...
        decode_outputs = model.decode(
            data, decoder_input, encode_outputs, all_decode_outputs[-1]
        )

        output = model.generate(data, encode_outputs, decode_outputs, softmax=softmax)

        all_gen_outputs.append(output)
        all_decode_outputs.append(decode_outputs)

        if schedule_rate >=1:
            decoder_input = tgt[:, t]
        elif schedule_rate<=0:
            probs, ids = model.to_word(data, output, 1)
            decoder_input = model.generation_to_decoder_input(data, ids[:, 0])
        else:
            probs, ids = model.to_word(data, output, 1)
            indices = model.generation_to_decoder_input(data, ids[:, 0])

            draws = torch.bernoulli(prob).long()
            decoder_input = tgt[:, t] * draws + indices * (1 - draws)

    # all_gen_outputs = torch.cat(all_gen_outputs, dim=0).transpose(0, 1).contiguous()

    return encode_outputs, init_decoder_states, all_decode_outputs, all_gen_outputs


def randomk(gen_output, k=5, PAD=None, BOS=None, UNK=None):
    if PAD is not None:
        gen_output[:, PAD] = -float('inf')
    if BOS is not None:
        gen_output[:, BOS] = -float('inf')
    if UNK is not None:
        gen_output[:, UNK] = -float('inf')
    values, indices = importance_sampling(gen_output, k)
    # words=[[tgt_id2vocab[id.item()] for id in one] for one in indices]
    return values, indices

# gen_output, k=k
def topk(gen_output, k=5, PAD=None, BOS=None, UNK=None):
    if PAD is not None:
        gen_output[:, PAD] = 0
    if BOS is not None:
        gen_output[:, BOS] = 0
    if UNK is not None:
        gen_output[:, UNK] = 0
    if k > 1:
        values, indices = torch.topk(gen_output, k, dim=1, largest=True,
                                     sorted=True, out=None)
    else:
        values, indices = torch.max(gen_output, dim=1, keepdim=True)  # (batch,1),(batch,1)
    return values, indices


def copy_topk(gen_output, vocab_map, vocab_overlap, k=5, PAD=None, BOS=None, UNK=None):
    vocab = gen_output[:, :vocab_map.size(-1)]
    dy_vocab = gen_output[:, vocab_map.size(-1):]

    vocab = vocab+torch.bmm(dy_vocab.unsqueeze(1), vocab_map).squeeze(1)
    dy_vocab = dy_vocab*vocab_overlap

    gen_output = torch.cat([vocab, dy_vocab], dim=-1)
    return topk(gen_output, k, PAD=PAD, BOS=BOS, UNK=UNK)

# [[tokens],[tokens]...batch]
def remove_duplicate_once(sents, n=3):
    changed = False
    for b in range(len(sents)):  # b[0,batch-1]
        sent = sents[b]  # [token1,token2,token3,...]
        if len(sent) <= n:
            continue

        for i in range(len(sent)-n):
            index = len(sent) - i - n
            if all(elem in sent[:index] for elem in sent[index:]):
                sents[b] = sent[:index]
                changed = True
                break
    return changed

# [[tokens],[tokens]...]
def remove_duplicate(sents, n=3):
    changed = remove_duplicate_once(sents, n)
    while changed:
        changed = remove_duplicate_once(sents, n)

# batch_indices(batch, max_decoder_len)
def to_sentence(batch_indices, id2vocab):
    batch_size = len(batch_indices)
    summ = list()
    for i in range(batch_size):
        indexes = batch_indices[i]
        text_summ2 = []
        for index in indexes:
            # item()
            # Returns the value of this tensor as a standard Python number.
            # This only works for tensors with one element. For other cases, see tolist().
            index = index.item()
            w = id2vocab[index]
            if w == BOS_WORD or w == PAD_WORD:
                continue
            if w == EOS_WORD:
                break
            text_summ2.append(w)
        if len(text_summ2) == 0:
            text_summ2.append(UNK_WORD)
        summ.append(text_summ2)
    return summ  # [[tokens],[tokens]...]


def to_copy_sentence(data, batch_indices, id2vocab, dyn_id2vocab_map):
    ids=data['id']
    batch_size=len(batch_indices)
    summ=list()
    for i in range(batch_size):
        indexes=batch_indices[i]
        text_summ2 = []
        dyn_id2vocab=dyn_id2vocab_map[ids[i].item()]
        for index in indexes:
            index = index.item()
            if index < len(id2vocab):
                w = id2vocab[index]
            elif index - len(id2vocab) in dyn_id2vocab:
                w = dyn_id2vocab[index - len(id2vocab)]
            else:
                w = PAD_WORD

            if w == BOS_WORD or w == PAD_WORD:
                continue

            if w == EOS_WORD:
                break

            text_summ2.append(w)

        if len(text_summ2)==0:
            text_summ2.append(UNK_WORD)

        summ.append(text_summ2)
    return summ


def create_emb_layer(emb_matrix, non_trainable=True):
    num_embeddings, embedding_dim = emb_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
    emb_layer.load_state_dict({'weight': emb_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer


def load_embeddings(emb_text_filepath, id2vocab, emb_dim):
    vectors = bcolz.open(emb_text_filepath+'.dat')[:]
    words = pickle.load(open(emb_text_filepath+'.words.pkl', 'rb'))
    word2idx = pickle.load(open(emb_text_filepath+'.ids.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    matrix_len = len(id2vocab)
    emb_matrix = torch.zeros((matrix_len, emb_dim))
    words_found = 0

    for i in range(len(id2vocab)):
        word=id2vocab[i]
        try:
            emb_matrix[i] = torch.Tensor(glove[word])
            words_found += 1
        except KeyError:
            emb_matrix[i] = torch.Tensor(np.random.normal(scale=0.6, size=(emb_dim,)))
    return emb_matrix


def prepare_embeddings(emb_text_filepath):
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=emb_text_filepath+'.temp', mode='w')

    with open(emb_text_filepath, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((idx, -1)), rootdir=emb_text_filepath+'.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(emb_text_filepath+'.words.pkl', 'wb'))
    pickle.dump(word2idx, open(emb_text_filepath+'.ids.pkl', 'wb'))


def new_tensor(array, requires_grad=False):
    tensor = torch.tensor(array, requires_grad=requires_grad)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor



def gru_forward(gru, input, lengths, state=None, batch_first=True):
    gru.flatten_parameters()
    input_lengths, perm = torch.sort(lengths, descending=True)

    input = input[perm]
    if state is not None:
        state = state[perm].transpose(0, 1).contiguous()

    total_length = input.size(1)
    if not batch_first:
        input = input.transpose(0, 1)  # B x L x N -> L x B x N
    packed = torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths, batch_first)
    # packed = hotfix_pack_padded_sequence(embedded, input_lengths, batch_first)
    # self.gru.flatten_parameters()
    outputs, state = gru(packed, state)  # -> L x B x N * n_directions, 1, B, N
    outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=batch_first, total_length=total_length)  # unpack (back to padded)

    _, perm = torch.sort(perm, descending=False)
    if not batch_first:
        outputs = outputs.transpose(0, 1)
    outputs=outputs[perm]
    state = state.transpose(0, 1)[perm]

    return outputs, state


def build_map(b_map, max=None):
    # b_map [batch, seq_len_q+num_sequences*seq_len_p]
    # max = self.vocab_size
    batch_size, b_len = b_map.size()
    if max is None:
        max=b_map.max() + 1
    b_map_ = torch.zeros(batch_size, b_len, max)
    if torch.cuda.is_available():
        b_map_ = b_map_.cuda()
    b_map_.scatter_(2, b_map.unsqueeze(2), 1.)
    # b_map_[:, :, 0] = 0.
    b_map_.requires_grad=False
    # [batch_size, seq_len_q+num_sequences*seq_len_p, tgt_vocab_size]
    return b_map_


def build_vocab(words, max = 100000):
    dyn_vocab2id = dict({PAD_WORD:0})
    dyn_id2vocab = dict({0:PAD_WORD})
    for w in words:
        if w not in dyn_vocab2id and len(dyn_id2vocab)<max:
            dyn_vocab2id[w] = len(dyn_vocab2id)
            dyn_id2vocab[len(dyn_id2vocab)] = w
    return dyn_vocab2id, dyn_id2vocab


def merge1D(sequences, max_len=None,pad_value=None):
    lengths = [len(seq) for seq in sequences]
    max_len=max(lengths) if max_len is None else max_len
    if pad_value is None:
        padded_seqs = torch.zeros(len(sequences), max_len,requires_grad=False).type_as(sequences[0])
    else:
        padded_seqs = torch.full((len(sequences), max_len),pad_value,requires_grad=False).type_as(sequences[0])

    for i, seq in enumerate(sequences):
        end = min(lengths[i], max_len)
        padded_seqs[i, :end] = seq[:end]
    return padded_seqs


def get_data(i, data):
    ones=dict()
    for key, value in data.items():
        if value is not None:
            if isinstance(value, torch.Tensor):
                ones[key] = value[i].unsqueeze(0)
            elif isinstance(value, dict):
                ones[key] = value[data['id'][i].item()]
            else:
                ones[key] = [value[i]]
        else:
            ones[key] = None
    return ones


def concat_data(datalist):
    data = dict()

    size = len(datalist)

    for key in datalist[0]:
        value=datalist[0][key]
        if value is not None:
            if isinstance(value, torch.Tensor):
                data[key] = torch.cat([datalist[i][key] for i in range(size)], dim=0)
            elif isinstance(value, dict):
                data[key] = dict()
                for i in range(size):
                    data[key][datalist[i]['id'].item()] = datalist[i][key]
            else:
                data[key] = [datalist[i][key] for i in range(size)]
        else:
            data[key]=None
    return data


def load_vocab(vocab_file,t=0):
    thisvocab2id = dict({PAD_WORD:0, BOS_WORD:1, UNK_WORD:2, EOS_WORD:3, SEP_WORD:4, CLS_WORD:5, MASK_WORD:6})
    thisid2vocab = dict({0:PAD_WORD, 1:BOS_WORD, 2:UNK_WORD, 3:EOS_WORD, 4:SEP_WORD, 5:CLS_WORD, 6:MASK_WORD})
    id2freq = {}

    sum_freq = 0
    with codecs.open(vocab_file, encoding='utf-8') as f:
        for line in f:
            try:
                name,freq = line.strip('\n').strip('\r').split('\t')
            except Exception:
                continue
            if int(freq) >= t:
                id = len(thisvocab2id)
                thisvocab2id[name] = id
                thisid2vocab[id] = name
                id2freq[id] = int(freq)
                sum_freq += int(freq)
    id2freq[0] = sum_freq/len(id2freq)
    id2freq[1] = id2freq[0]
    id2freq[2] = id2freq[0]
    id2freq[3] = id2freq[0]

    print('item size: ', len(thisvocab2id))

    return thisvocab2id, thisid2vocab, id2freq


def load_embedding(src_vocab2id, file):
    model=dict()
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            model[word] = torch.tensor([float(val) for val in splitLine[1:]])
    matrix = torch.zeros((len(src_vocab2id), 100))
    xavier_uniform_(matrix)
    for word in model:
        if word in src_vocab2id:
            matrix[src_vocab2id[word]]=model[word]
    return matrix