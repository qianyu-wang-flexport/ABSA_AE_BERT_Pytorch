import torch
max_len=100
model_name='bert_bio'
def _get_seq_index(seq):
    bio_list = []
    k = 0
    i_label = 4 if model_name == 'bert_bio' else 2
    while k < max_len:
        pred = seq[k].item()
        if pred != 0:
            polarity = pred
            index_list = [k]
            k += 1
            while k < max_len and seq[k].item() == i_label:
                index_list.append(k)
                k += 1
            bio_list.append((polarity, index_list) if model_name == 'bert_bio' else index_list)
        else:
            k += 1
    return bio_list

seq=torch.tensor([0, 0, 2, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0])
print(_get_seq_index(seq))