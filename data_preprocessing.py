from pytorch_pretrained_bert import BertTokenizer
import torch
import os
import pickle
import numpy as np


bert_path='/data/bert-pretrained-models/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(os.path.join(bert_path, 'bert-base-uncased-vocab.txt'))

def data_padding(max_len,seq_tensor):
    x = torch.tensor([0]*max_len)
    trunc = seq_tensor[:max_len]
    x[:len(trunc)] = trunc

    return x
def get_token_tensor(context,aspect,polarity,max_len):

    context_list = context.split('$T$')
    context_left = context_list[0].lower().strip()
    context_left_tokenizer = tokenizer.tokenize(context_left)
    context_right = context_list[1].lower().strip()
    context_right_tokenizer = tokenizer.tokenize(context_right)
    aspect = aspect.lower().strip()
    aspect_tokenizer = tokenizer.tokenize(aspect)
    polarity = polarity.lower().strip()
    aspect_bio = [int(polarity) + 2] + [4] * (len(aspect_tokenizer) - 1)
    # 获得整句话的tokenizer
    all_tokenizer = ['[CLS]'] + context_left_tokenizer + aspect_tokenizer + context_right_tokenizer + ['[SEP]']

    all_index_tensor=data_padding(max_len,torch.tensor(tokenizer.convert_tokens_to_ids(all_tokenizer)))
    all_id_tensor=data_padding(max_len,torch.tensor([0]*len(all_tokenizer)))
    bio = [0] * (1 + len(context_left_tokenizer)) + aspect_bio + [0] * (1 + len(context_right_tokenizer))
    bio_tensor=data_padding(max_len,torch.tensor(bio))
    data={
        'all_tokenizer':all_tokenizer,
        'all_index_tensor':all_index_tensor,
        'all_id_tensor':all_id_tensor,
        'bio':bio,
        'bio_tensor':bio_tensor
    }
    return data

#O:0    B-neg:1    B-neu:2     B-pos:3     I:4
def get_preprocessing_data(file,max_len):

    f=open(file)
    lines=f.readlines()
    f.close()
    all_data=[]
    #获取第一句话的tokenizer
    context = lines[0]
    aspect = lines[1]
    polarity = lines[2]
    last_data=get_token_tensor(context,aspect,polarity,max_len)
    num=-1
    for i in range(3,len(lines),3):
        context=lines[i]
        aspect=lines[i+1]
        polarity=lines[i+2]
        data=get_token_tensor(context,aspect,polarity,max_len)
        if data['all_tokenizer']==last_data['all_tokenizer']:
            data['bio_tensor']=data['bio_tensor']+last_data['bio_tensor']
            num+=1
        else:
            all_data.append(last_data)
        last_data=data
    all_data.append(last_data)
    print('共有{}条重复评论'.format(num))

    return all_data

def get_birt_trainset():
    dataset_path = 'datasets/semeval14'
    train_data_file = os.path.join(dataset_path, 'Restaurants_Train.pkl')
    f=open(train_data_file,'rb')
    all_data=pickle.load(f)
    f.close()
    return all_data

def get_birt_testset():
    dataset_path = 'datasets/semeval14'
    train_data_file = os.path.join(dataset_path, 'Restaurants_Test.pkl')
    f=open(train_data_file,'rb')
    all_data=pickle.load(f)
    f.close()
    return all_data


if __name__=='__main__':
    max_len=100
    dataset_path='datasets/semeval14'
    restaurant_train_file =os.path.join(dataset_path, 'Restaurants_Train.xml.seg')
    restaurant_test_file=os.path.join(dataset_path,'Restaurants_Test_Gold.xml.seg')
    train_data_file =os.path.join(dataset_path, 'Restaurants_Train.pkl')
    test_data_file = os.path.join(dataset_path, 'Restaurants_Test.pkl')
    # train_all_data=get_preprocessing_data(restaurant_train_file,max_len)
    test_all_data=get_preprocessing_data(restaurant_test_file,max_len)
    print("共有{}条测试数据".format(len(test_all_data)))
    # f1=open(train_data_file,'wb')
    # pickle.dump(train_all_data,f1)
    # f1.close()
    f2=open(test_data_file,'wb')
    pickle.dump(test_all_data,f2)
    f2.close()
    # all_data=get_birt_testset()
    # print(all_data)

    # file = open('data_preprocessing.pkl', 'rb')
    # all_data=pickle.load(file)
    # print(len(all_data))
    # print(all_data[4])