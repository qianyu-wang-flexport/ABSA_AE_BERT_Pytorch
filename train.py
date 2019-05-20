from pytorch_pretrained_bert import BertModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse
import math
import os
from pytorch_pretrained_bert.optimization import BertAdam
from models.bert_bio import BERT_BIO
from models.bert_sa import BERT_SA
from models.bert_te import BERT_TE
from torch.utils.data import Dataset
import pickle

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
        max_test_acc = 0
        max_f1 = 0
        max_epoch=0
        global_step = 0
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                # print('train batch num:{}'.format(i_batch))
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched[opt.outputs_col].to(self.opt.device)

                # print(outputs.shape)
                # print(targets.shape)

                s_batch_size=targets.shape[0]
                if opt.model_name=='bert_bio' or opt.model_name=='bert_te':
                    loss = criterion(outputs.view(s_batch_size*opt.max_len,-1), targets.view(s_batch_size*opt.max_len))
                else:
                    loss=criterion(outputs,targets)
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += targets.view(-1).shape[0]
                    train_acc = n_correct / n_total

                    # switch model to evaluation mode
                    self.model.eval()
                    test_acc, f1 = self._evaluate_acc_f1()

                    #20190510以f1为评价标准
                    if f1 > max_f1:
                        max_test_acc = test_acc
                        max_f1=f1
                        max_epoch=epoch
                        if not os.path.exists('state_dict'):
                            os.mkdir('state_dict')
                        print('>>>max_acc:{:.4f},max_f1:{:.4f}'.format(max_test_acc,max_f1))
                        # path = 'state_dict/{0}_{1}_f1{2}'.format(self.opt.model_name, self.opt.dataset, round(f1, 4))
                        # torch.save(self.model.state_dict(), path)
                        # print('>> saved: ' + path)

                    writer.add_scalar('loss', loss, global_step)
                    writer.add_scalar('acc', train_acc, global_step)
                    writer.add_scalar('test_acc', test_acc, global_step)
                    print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc, test_acc, f1))

        writer.close()
        return max_test_acc, max_f1,max_epoch

    def _get_seq_index(self,seq):
        bio_list=[]
        k=0
        while k<opt.max_len:
            pred = seq[k].item()
            if pred!=0:
                polarity=pred
                index_list=[k]
                k+=1
                while k<opt.max_len and seq[k].item==4:
                    index_list.append(k)
                    k+=1
                bio_list.append((polarity,index_list))
            else:
                k+=1
        return bio_list


    def _evaluate_acc_f1(self):
        n_test_correct, n_test_total = 0, 0
        n_ground_truth=0
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched[opt.outputs_col].to(opt.device)
                t_outputs = self.model(t_inputs)
                s_batch_size=t_targets.shape[0]

                for i in range(s_batch_size):

                    s_output=t_outputs[i]
                    s_target=t_targets[i]
                    # s_token_indices=t_inputs[0][i]
                    s_pred=torch.argmax(s_output,-1)

                    if opt.model_name == 'bert_bio' or opt.model_name == 'bert_te':
                        pred_list=self._get_seq_index(s_pred)
                        target_list=self._get_seq_index(s_target)
                        for pred in pred_list:
                            if pred in target_list:
                                n_test_correct+=1
                        n_test_total += len(pred_list)
                        n_ground_truth += len(target_list)
                    else:
                        if s_pred==s_target:
                            n_test_correct+=1
                        n_test_total+=1
                        n_ground_truth+=1

        if n_test_total==0:
            test_acc=0
        else:
            test_acc = n_test_correct / n_test_total
        test_recall=n_test_correct/n_ground_truth
        print('test_correct:{},test_total:{},ground_truth:{}'.format(n_test_correct,n_test_total,n_ground_truth))
        if test_acc==0 or test_recall==0:
            f1=0
        else:
            f1=2*test_acc*test_recall/(test_acc+test_recall)
        return test_acc, f1

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
        max_test_acc, max_f1,max_epoch = self._train(criterion, optimizer)
        print('epoch {} get max_test_acc: {:.4f}     max_f1: {:.4f}'.format(max_epoch,max_test_acc, max_f1))


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_bio', type=str)
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
    opt = parser.parse_args()

    dataset_path = 'datasets/semeval14'

    models={
        'bert_bio':BERT_BIO,
        'bert_sa':BERT_SA,
        'bert_te':BERT_TE
    }
    dataset_files = {
        'restaurant':{
            'train':os.path.join(dataset_path, 'Restaurants_Train.pkl'),
            'test':os.path.join(dataset_path, 'Restaurants_Test.pkl')
        }
    }
    class_dims={
        'bert_bio':5,
        'bert_te':3,
        'bert_sa':4
    }
    input_columns = {
        'bert_bio':['all_index_tensor','all_id_tensor'],
        'bert_sa':['all_index_tensor','all_id_tensor'],
        'bert_te':['all_index_tensor','all_id_tensor']
    }
    output_columns={
        'bert_bio':'bio_tensor',
        'bert_sa':'polarity',
        'bert_te':'bio_normal_tensor'
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
    opt.class_dim=class_dims[opt.model_name]
    opt.model=models[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_columns[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    opt.outputs_col=output_columns[opt.model_name]
    ins = Instructor(opt)
    ins.run()
