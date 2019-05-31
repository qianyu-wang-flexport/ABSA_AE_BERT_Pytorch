from pytorch_pretrained_bert import BertModel,BertTokenizer
import torch
import os
from models.bert_multilayer import Layer_Att

bert_path='/data/bert-pretrained-models/bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(os.path.join(bert_path, 'bert-base-uncased-vocab.txt'))
model = BertModel.from_pretrained(bert_path)

text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda:1')
segments_tensors = segments_tensors.to('cuda:1')
model.to('cuda:1')

# Predict hidden states features for each layer
with torch.no_grad():
    encoded_layers, label_layer = model(tokens_tensor, segments_tensors,output_all_encoded_layers=True)

print(label_layer.shape)

att=Layer_Att()
att.to('cuda:1')
att.eval()
with torch.no_grad():
    output2=att(encoded_layers)

print(output2.shape)
