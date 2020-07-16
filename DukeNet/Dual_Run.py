import sys
sys.path.append('./')
from DukeNet.Dataset import *
from torch import optim
from DukeNet.CumulativeTrainer import *
import torch.backends.cudnn as cudnn
import argparse
from DukeNet.Model import *
from dataset.Utils_DukeNet import *
import os
import time
from torch.distributions import Categorical


def rounder(num, places):
    return round(num, places)


def dual_train(args):
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    print("torch_version:{}".format(torch.__version__))
    print("CUDA_version:{}".format(torch.version.cuda))
    print("cudnn_version:{}".format(cudnn.version()))

    init_seed(123456)

    data_path = args.base_data_path+args.dataset+'/'

    print("Load BERT vocab")
    tokenizer, vocab2id, id2vocab = bert_tokenizer()
    print('--Vocabulary size', len(vocab2id))

    print("Load dataset")
    # load dataset
    query = torch.load(data_path + 'query_DukeNet.pkl')
    train_samples = torch.load(data_path + 'train_DukeNet.pkl')
    passage = torch.load(data_path + 'passage_DukeNet.pkl')
    print("--The number of train_samples:", len(train_samples))


    print("Establish model and load parameters")
    saved_model_path = os.path.join(args.base_output_path + args.name + "/", 'model/')
    with open(saved_model_path + "checkpoints.json", 'r', encoding='utf-8') as r:
        checkpoints = json.load(r)
    last_epoch = checkpoints["time"][-1]
    fuse_dict = torch.load(os.path.join(saved_model_path, '.'.join([str(last_epoch), 'pkl'])))
    model = DukeNet(vocab2id, id2vocab, args)
    model.load_state_dict(fuse_dict["model"])
    # freeze the parameter of encoder，reducing the cost of GPU memory.
    freeze_params(model, "enc")

    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model
    model.train()
    print('--Loading success, last_epoch is {}'.format(last_epoch))

    print("Create optimizer")
    A_optimizer = optim.Adam(model.shifter.parameters(), args.A_lr)
    B_optimizer = optim.Adam(model.posterior_tracker.parameters(), args.B_lr)
    All_optimizer = optim.Adam(model.parameters(), args.ALL_lr)


    A_optimizer.zero_grad()
    B_optimizer.zero_grad()
    All_optimizer.zero_grad()

    print("Define loss")
    loss_nll = torch.nn.NLLLoss(reduction='none')
    KLDLoss = nn.KLDivLoss(reduction='batchmean')

    if isinstance(last_epoch, int):
        last_epoch = -1
    else:
        last_epoch = int(last_epoch.split('_')[1])


    # epoch start ======================================================================================================
    for epoch in range(last_epoch+1, args.epoches):
        print("Epoch:", epoch)
        print("Create dataloader")
        train_dataset = Dataset("train", train_samples, query, passage, vocab2id, args.max_knowledge_pool_when_train, args.max_knowledge_pool_when_inference, args.context_len, args.knowledge_sentence_len, args.max_dec_length)
        train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.dual_train_batch_size, shuffle=True)

        # each example
        for j, data in enumerate(train_loader, 0):
            if torch.cuda.is_available():
                data_cuda = dict()
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data_cuda[key] = value.cuda()
                    else:
                        data_cuda[key] = value
                data = data_cuda


                # A(shifting)============================================================================================
                # (N,K)
                #print("Start from shifting")
                encoded_state = model.encoding_layer(data)

                # get label shifting knowledge
                N, K, H = encoded_state['knowledge_tracking_pool_encoded'][1].size()
                offsets = torch.arange(N, device=data["knowledge_tracking_label"].device) * K + data["knowledge_tracking_label"]  # N
                # knowledge_tracking_pool_use (N K E)->(N*K,E)
                flatten_knowledge_tracking_pool_use = encoded_state['knowledge_tracking_pool_encoded'][1].view(N * K,-1)
                label_tracked_knowledge_use = flatten_knowledge_tracking_pool_use[offsets]  # (N E)


                # N K
                knowledge_shifting_score, _, _, _ ,_ = model.shifter(encoded_state['contexts_encoded'],
                                                                 label_tracked_knowledge_use,  #
                                                                 encoded_state['knowledge_shifting_pool_encoded'],
                                                                 encoded_state['knowledge_shifting_pool_mask'],
                                                                 data["shifting_ck_mask"],
                                                                 data["knowledge_shifting_label"],
                                                                 data['knowledge_shifting_pool'],
                                                                 mode="inference")


                knowledge_shifting_prob = F.softmax(knowledge_shifting_score, -1)

                logist = Categorical(knowledge_shifting_prob)
                inferred_shifted_knowledge_index = logist.sample()  # N


                N, K, H = encoded_state['knowledge_shifting_pool_encoded'][1].size()  # (N K E)
                offsets = torch.arange(N, device=inferred_shifted_knowledge_index.device) * K + inferred_shifted_knowledge_index  # N
                # knowledge_shifting_pool_use (N K E)->(N*K,E)
                flatten_knowledge_shifting_pool_use = encoded_state['knowledge_shifting_pool_encoded'][1].view(N * K, -1)
                inferred_shifted_knowledge_use = flatten_knowledge_shifting_pool_use[offsets]  # (N E)

                # action prob
                # N
                action_prob_loss = loss_nll(torch.log(knowledge_shifting_prob+1e-10), inferred_shifted_knowledge_index)

                # B
                with torch.no_grad():
                    # (N,K)
                    knowledge_tracking_score, _, _, _ = model.posterior_tracker(encoded_state['contexts_encoded'],
                                                                 inferred_shifted_knowledge_use,
                                                                 encoded_state['knowledge_tracking_pool_encoded'],
                                                                 encoded_state['knowledge_tracking_pool_mask'],
                                                                 data['tracking_ck_mask'],
                                                                 data["knowledge_tracking_label"],
                                                                 mode="inference")



                    # N
                    reward = -loss_nll(F.log_softmax(knowledge_tracking_score, 1), data["knowledge_tracking_label"])
                    # N
                    norm_reward = (reward - torch.mean(reward)) / torch.std(reward)


                A_loss = torch.mean(action_prob_loss * norm_reward)
                A_optimizer.zero_grad()
                A_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.shifter.parameters(), 0.4)
                A_optimizer.step()

                print_A_loss = A_loss.cpu().item()



                # B tracking===========================================================================================
                #print("Start from tracking")
                encoded_state = model.encoding_layer(data)
                # get label shifting knowledge
                N, K, H = encoded_state['knowledge_shifting_pool_encoded'][1].size()
                offsets = torch.arange(N, device=data["knowledge_shifting_label"].device) * K + data["knowledge_shifting_label"]  # N
                # knowledge_shifting_pool_use (N K E)->(N*K,E)
                flatten_knowledge_shifting_pool_use = encoded_state['knowledge_shifting_pool_encoded'][1].view(N * K,-1)
                label_shifted_knowledge_use = flatten_knowledge_shifting_pool_use[offsets]  # (N E)

                # N K
                knowledge_tracking_score, _, _, _ = model.posterior_tracker(encoded_state['contexts_encoded'],
                                                                           label_shifted_knowledge_use,
                                                                           encoded_state['knowledge_tracking_pool_encoded'],
                                                                           encoded_state['knowledge_tracking_pool_mask'],
                                                                           data['tracking_ck_mask'],
                                                                           data["knowledge_tracking_label"],
                                                                           mode="inference")

                # N K
                knowledge_tracking_prob = F.softmax(knowledge_tracking_score, -1)

                logist = Categorical(knowledge_tracking_prob)
                # N
                inferred_tracked_knowledge_index = logist.sample()  # batch



                N, K, H = encoded_state['knowledge_tracking_pool_encoded'][1].size()  # (N K E)
                offsets = torch.arange(N, device=inferred_tracked_knowledge_index.device) * K + inferred_tracked_knowledge_index  # N
                flatten_knowledge_tracking_pool_use = encoded_state['knowledge_tracking_pool_encoded'][1].view(N * K, -1)
                inferred_tracked_knowledge_use = flatten_knowledge_tracking_pool_use[offsets]  # (N E)

                # action prob
                # N
                action_prob_loss = loss_nll(torch.log(knowledge_tracking_prob+1e-10), inferred_tracked_knowledge_index)

                with torch.no_grad():
                    knowledge_shifting_score, _, _, _, _ = model.shifter(encoded_state['contexts_encoded'],
                                                                         inferred_tracked_knowledge_use,  # label tracked knowledge
                                                                         encoded_state['knowledge_shifting_pool_encoded'],
                                                                         encoded_state['knowledge_shifting_pool_mask'],
                                                                         data["shifting_ck_mask"],
                                                                         data["knowledge_shifting_label"],
                                                                         data['knowledge_shifting_pool'],
                                                                         mode="inference")


                    reward = -loss_nll(F.log_softmax(knowledge_shifting_score, -1), data["knowledge_shifting_label"])
                    norm_reward = (reward - torch.mean(reward)) / torch.std(reward)


                B_loss = torch.mean(action_prob_loss * norm_reward)
                B_optimizer.zero_grad()
                B_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.posterior_tracker.parameters(), 0.4)
                B_optimizer.step()
                print_B_loss = B_loss.cpu().item()


                # ALL====================================================================================================
                encoded_state = model.encoding_layer(data)
                interaction_outputs = model.dual_knowledge_interaction_layer(data, encoded_state)
                rg = model.decoding_layer(data, interaction_outputs)

                _, pri_tracking_pred = interaction_outputs['prior_knowledge_tracking_score'].detach().max(1)
                pri_tracking_acc = (pri_tracking_pred == data['knowledge_tracking_label']).float().mean()

                _, pos_tracking_pred = interaction_outputs['posterior_knowledge_tracking_score'].detach().max(1)
                pos_tracking_acc = (pos_tracking_pred == data['knowledge_tracking_label']).float().mean()

                _, shifting_pred = interaction_outputs['knowledge_shifting_score'].detach().max(1)
                shifting_acc = (shifting_pred == data['knowledge_shifting_label']).float().mean()

                # As with NLLLoss, the input given is expected to contain log-probabilities
                # The targets are given as probabilities (i.e. without taking the logarithm).
                pri_2_pos = KLDLoss(F.log_softmax(interaction_outputs['prior_knowledge_tracking_score'], 1),
                                    F.softmax(interaction_outputs['posterior_knowledge_tracking_score'], 1).detach())

                loss_pos_tracking = F.nll_loss(F.log_softmax(interaction_outputs['posterior_knowledge_tracking_score'], -1),
                    data['knowledge_tracking_label'].view(-1))

                loss_shifting = F.nll_loss(F.log_softmax(interaction_outputs['knowledge_shifting_score'], -1),
                                           data['knowledge_shifting_label'].view(-1))

                loss_g = F.nll_loss((rg[0] + 1e-8).log().reshape(-1, rg[0].size(-1)), data['response'].reshape(-1),
                                    ignore_index=0)

                ALL_loss = pri_2_pos + (loss_pos_tracking + loss_shifting + loss_g)*0.5

                All_optimizer.zero_grad()
                ALL_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.4)
                All_optimizer.step()

                if j % 10 == 0:
                    print('Training: %s' % "Dual_Game_DukeNet")
                    print(
                        "Epoch:{}, Batch:{}, Loss_A:{}, Loss_B:{}, KLDLoss:{}, Pos_Tra_Loss:{}, Shi_Loss:{}, Gen_Loss:{}, Pri_T_ACC:{}, Pos_T_ACC:{}, Shi_ACC:{}"
                            .format(epoch, j, rounder(print_A_loss, 4), rounder(print_B_loss, 4),
                                    rounder(pri_2_pos.cpu().item(), 4), rounder(loss_pos_tracking.cpu().item(), 4),
                                    rounder(loss_shifting.cpu().item(), 4), rounder(loss_g.cpu().item(), 4),
                                    rounder(pri_tracking_acc.cpu().item(), 2),
                                    rounder(pos_tracking_acc.cpu().item(), 2),
                                    rounder(shifting_acc.cpu().item(), 2)))


        # save model====================================================================================================
        fuse_dict = {"model": model.state_dict()}
        torch.save(fuse_dict,os.path.join(saved_model_path, '.'.join(["d_" + str(epoch), 'pkl'])))
        print("Saved epoch {} model".format("d_" + str(epoch)))
        with open(saved_model_path + "checkpoints.json", 'r', encoding='utf-8') as r:
                checkpoints = json.load(r)
        checkpoints["time"].append("d_" + str(epoch))
        with open(saved_model_path + "checkpoints.json", 'w', encoding='utf-8') as w:
            json.dump(checkpoints, w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default='DukeNet')
    parser.add_argument("--base_output_path", type=str, default='output/')
    parser.add_argument("--base_data_path", type=str, default='datasets/')
    parser.add_argument("--dataset", type=str, default='wizard_of_wikipedia')
    parser.add_argument("--GPU", type=int, default=0)

    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--epoches", type=int, default=5)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--A_lr", type=float, default=1e-5)
    parser.add_argument("--B_lr", type=float, default=1e-5)
    parser.add_argument("--ALL_lr", type=float, default=1e-5)
    parser.add_argument("--dual_train_batch_size", type=int, default=64)

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

    dual_train(args)