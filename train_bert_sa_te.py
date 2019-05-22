from pytorch_pretrained_bert import BertModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse
import math
import os
from pytorch_pretrained_bert.optimization import BertAdam
from models.bert_sa_te import BERT_SA_TE
from torch.utils.data import Dataset
import pickle
from sklearn import metrics

bert_path='/data/bert-pretrained-models/bert-base-uncased'

class MyDataset(Dataset):
    def __init__(self,data_file):
        f=open(data_file,'rb')
        self.data=pickle.load(f)
        f.close()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        bert = BertModel.from_pretrained(bert_path)
        self.model = opt.model(bert, opt).to(opt.device)
        if opt.load_state_dic == 'Yes':
            state_dic_path=opt.load_state_dic_file
            pretrained_dic=torch.load(state_dic_path)
            model_dic=self.model.state_dict()
            #只加载bert中的参数
            new_dic={k:v for k,v in pretrained_dic.items() if 'bert' in k}
            model_dic.update(new_dic)
            print('total param:{},loaded param:{}'.format(len(model_dic),len(new_dic)))
            self.model.load_state_dict(model_dic)
            print('load finish')
        trainset =MyDataset(opt.dataset_file['train'])
        testset = MyDataset(opt.dataset_file['test'])
        self.train_data_loader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(dataset=testset, batch_size=opt.batch_size, shuffle=False)

        if opt.device.type == 'cuda':
            print("cuda memory allocated:", torch.cuda.memory_allocated(device=opt.device.index))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params (with unfreezed bert)
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer):
        writer = SummaryWriter(log_dir=self.opt.logdir)
        senti_max_test_precision,bio_max_test_precision = 0,0
        senti_max_f1,bio_max_f1 = 0,0
        senti_max_epoch=0
        bio_max_epoch=0
        global_step = 0
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_senti_correct, n_senti_total = 0, 0
            n_bio_correct,n_bio_total=0,0
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                # print('train batch num:{}'.format(i_batch))
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                senti_outputs,bio_outputs = self.model(inputs)
                senti_targets = sample_batched[opt.outputs_col[0]].to(self.opt.device)
                bio_targets = sample_batched[opt.outputs_col[1]].to(self.opt.device)

                # print(outputs.shape)
                # print(targets.shape)

                s_batch_size=bio_targets.shape[0]
                loss1=criterion(bio_outputs.view(s_batch_size*opt.max_len,-1), bio_targets.view(s_batch_size*opt.max_len))
                loss2 = criterion(senti_outputs,senti_targets)
                loss=loss1+loss2
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_senti_correct += (torch.argmax(senti_outputs, -1) == senti_targets).sum().item()
                    n_senti_total += senti_targets.view(-1).shape[0]
                    senti_train_precision = n_senti_correct / n_senti_total

                    n_bio_correct += (torch.argmax(bio_outputs, -1) == bio_targets).sum().item()
                    n_bio_total += bio_targets.view(-1).shape[0]
                    bio_train_precision = n_bio_correct / n_bio_total

                    # switch model to evaluation mode
                    self.model.eval()
                    senti_test_precision, senti_f1,bio_test_precision,bio_f1 = self._evaluate_acc_f1()

                    #sentiment evaluation
                    if senti_f1>senti_max_f1:
                        senti_max_test_precision = senti_test_precision
                        senti_max_f1=senti_f1
                        senti_max_epoch=epoch
                        if opt.save_state_dic == "Yes":
                            if not os.path.exists('state_dict'):
                                os.mkdir('state_dict')
                            path = 'state_dict/{0}_senti_{1}_pre{2}_f1{3}'.format(self.opt.model_name, self.opt.dataset,
                                                                     round(senti_test_precision,4),round(senti_f1, 4))
                            torch.save(self.model.state_dict(), path)
                            print('>> saved: ' + path)
                        print('>>>sa_max_precision:{:.4f},sa_max_f1:{:.4f}'.format(senti_max_test_precision,senti_max_f1))
                    #te evaluation
                    if bio_f1>bio_max_f1:
                        bio_max_test_precision = bio_test_precision
                        bio_max_f1=bio_f1
                        bio_max_epoch=epoch
                        if opt.save_state_dic == "Yes":
                            if not os.path.exists('state_dict'):
                                os.mkdir('state_dict')
                            path = 'state_dict/{0}_te_{1}_pre{2}_f1{3}'.format(self.opt.model_name, self.opt.dataset,
                                                                     round(bio_test_precision,4),round(bio_f1, 4))
                            torch.save(self.model.state_dict(), path)
                            print('>> saved: ' + path)
                        print('>>>te_max_precision:{:.4f},te_max_f1:{:.4f}'.format(bio_max_test_precision,bio_max_f1))

                    writer.add_scalar('loss', loss, global_step)
                    writer.add_scalar('senti_train_precision', senti_train_precision, global_step)
                    writer.add_scalar('senti_test_precision', senti_test_precision, global_step)
                    writer.add_scalar('te_train_precision', senti_train_precision, global_step)
                    writer.add_scalar('te_test_precision', senti_test_precision, global_step)
                    print('loss: {:.4f}'.format(loss.item()))
                    print('sa_train_precision: {:.4f},sa_test_precision: {:.4f},sa_f1:{:.4f}'.format(senti_train_precision, senti_test_precision,senti_f1))
                    print('te_train_precision:{:.4f},te_test_precision: {:.4f},te_f1: {:.4f}'.format(bio_train_precision,bio_test_precision, bio_f1))
                    print('-'*100)

        writer.close()
        return senti_max_test_precision, senti_max_f1,senti_max_epoch,bio_max_test_precision,bio_max_f1,bio_max_epoch

    def _get_seq_index(self,seq):
        bio_list=[]
        k=0
        i_label= 2
        while k<opt.max_len:
            pred = seq[k].item()
            if pred!=0:
                index_list=[k]
                k+=1
                while k<opt.max_len and seq[k].item()==i_label:
                    index_list.append(k)
                    k+=1
                bio_list.append(index_list)
            else:
                k+=1
        return bio_list


    def _evaluate_acc_f1(self):
        bio_test_correct, bio_test_total = 0, 0
        bio_ground_truth=0
        senti_targets_all, senti_outputs_all = None, None

        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                senti_targets = t_sample_batched[opt.outputs_col[0]].to(opt.device)
                bio_targets=t_sample_batched[opt.outputs_col[1]].to(opt.device)
                senti_outputs,bio_outputs = self.model(t_inputs)
                s_batch_size=bio_targets.shape[0]

                if senti_targets_all is None:
                    senti_targets_all = senti_targets
                    senti_outputs_all = senti_outputs
                else:
                    senti_targets_all = torch.cat((senti_targets_all, senti_targets), dim=0)
                    senti_outputs_all = torch.cat((senti_outputs_all, senti_outputs), dim=0)


                for i in range(s_batch_size):

                    s_output=bio_outputs[i]
                    s_target=bio_targets[i]
                    # s_token_indices=t_inputs[0][i]
                    s_pred=torch.argmax(s_output,-1)
                    pred_list=self._get_seq_index(s_pred)
                    target_list=self._get_seq_index(s_target)
                    # print('pred_list:{}'.format(pred_list))
                    # print('target_list:{}'.format(target_list))
                    for pred in pred_list:
                        if pred in target_list:
                            bio_test_correct+=1
                    bio_test_total += len(pred_list)
                    bio_ground_truth += len(target_list)


        senti_f1 = metrics.f1_score(senti_targets_all.cpu(), torch.argmax(senti_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        senti_test_precision=metrics.precision_score(senti_targets_all.cpu(), torch.argmax(senti_outputs_all, -1).cpu(), labels=[0, 1, 2],average='macro')

        if bio_test_total == 0:
            bio_test_precision = 0
        else:
            bio_test_precision = bio_test_correct / bio_test_total
        bio_test_recall=bio_test_correct/bio_ground_truth
        # print('test_correct:{},test_total:{},ground_truth:{}'.format(n_test_correct,n_test_total,n_ground_truth))
        if bio_test_precision==0 or bio_test_recall==0:
            bio_f1=0
        else:
            bio_f1=2*bio_test_precision*bio_test_recall/(bio_test_precision+bio_test_recall)
        return senti_test_precision, senti_f1,bio_test_precision,bio_f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.opt.optimizer==BertAdam:
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg,warmup=0.1)
        else:
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        # print(optimizer)

        self._reset_params()
        senti_max_test_precision, senti_max_f1,senti_max_epoch,bio_max_test_precision,bio_max_f1,bio_max_epoch= self._train(criterion, optimizer)
        print('epoch {} get sa_max_test_precision: {:.4f}     sa_max_f1: {:.4f}'.format(senti_max_epoch,senti_max_test_precision, senti_max_f1))
        print('epoch {} get te_max_test_precision: {:.4f}     te_max_f1: {:.4f}'.format(bio_max_epoch, bio_max_test_precision, bio_max_f1))

if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_sa_te', type=str)
    parser.add_argument('--dataset', default='restaurant', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float)  # try 5e-5, 3e-5, 2e-5 for BERT models (sensitive)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=32, type=int)  # try 16, 32, 64 for BERT models
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--logdir', default='log', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_len', default=100, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--load_state_dic',default="No",type=str)
    parser.add_argument('--save_state_dic',default="No",type=str)
    opt = parser.parse_args()

    dataset_path = 'datasets/semeval14'

    models={
        'bert_sa_te':BERT_SA_TE
    }

    state_dic_file={
        'bert_sa_te':'None'
    }

    dataset_files = {
        'restaurant': {
            'train': os.path.join(dataset_path, 'Cutout_Restaurant_Train.pkl'),
            'test': os.path.join(dataset_path, 'Cutout_Restaurant_Test.pkl')
        }
    }

    senti_class_dims = {
        'bert_sa_te':3
    }
    bio_class_dims = {
        'bert_sa_te': 3
    }
    input_columns = {
        'bert_sa_te':['all_index_tensor','all_id_tensor']
    }
    output_columns={
        'bert_sa_te':['polarity','bio_normal_tensor']
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
        'bertAdam':BertAdam
    }
    opt.senti_class_dim=senti_class_dims[opt.model_name]
    opt.bio_class_dim = bio_class_dims[opt.model_name]
    opt.model=models[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_columns[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.load_state_dic_file=state_dic_file[opt.model_name]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    opt.outputs_col=output_columns[opt.model_name]
    ins = Instructor(opt)
    ins.run()
