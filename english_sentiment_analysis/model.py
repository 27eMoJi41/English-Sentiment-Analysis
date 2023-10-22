import torch
import torch.nn as nn
from transformers import AutoTokenizer,AutoModel
import numpy as np
from transformers import logging
import warnings

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

tokenizer = AutoTokenizer.from_pretrained("maymuni/bert-base-turkish-cased-emotion-analysis")
bert = AutoModel.from_pretrained("maymuni/bert-base-turkish-cased-emotion-analysis", return_dict=False)

class Arch(nn.Module):
    def __init__(self, bert):
        super(Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc3 = nn.Linear(512, 13)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)

        return x

model = Arch(bert)

path = 'emotion_analysis.pt'
model.load_state_dict(torch.load(path))

def call():
    print("Write a sentence")
    predict_emotion(str(input()))

def predict_emotion(text):
   if text == "exit":
      exit()

   if text.strip() == "":
       print("Write Something")
       call()

   tokenized = tokenizer.encode_plus(
      text,
      pad_to_max_length=True,
      truncation=True,
      return_token_type_ids=False
   )

   input_ids = tokenized['input_ids']
   attention_mask = tokenized['attention_mask']

   seq = torch.tensor(input_ids)
   mask = torch.tensor(attention_mask)
   seq = seq.unsqueeze(0)
   mask = mask.unsqueeze(0)
   preds = model(seq, mask)
   preds = preds.detach().cpu().numpy()
   result = np.argmax(preds, axis=1)
   preds = torch.tensor(preds)
   probabilities = nn.functional.softmax(preds)
   dominant_emotion = ""
   mx = max(float(probabilities[0][0])*100,
       float(probabilities[0][1])*100,
       float(probabilities[0][2])*100,
       float(probabilities[0][3])*100,
       float(probabilities[0][4])*100,
       float(probabilities[0][5])*100,
       float(probabilities[0][6])*100,
       float(probabilities[0][7])*100,
       float(probabilities[0][8])*100,
       float(probabilities[0][9])*100,
       float(probabilities[0][10])*100,
       float(probabilities[0][11])*100,
       float(probabilities[0][12])*100)

   if mx == float(probabilities[0][0])*100:
       dominant_emotion = "empty"

   if mx == float(probabilities[0][1])*100:
       dominant_emotion = 'sadness'

   if mx == float(probabilities[0][2])*100:
       dominant_emotion = 'enthusiasm'

   if mx == float(probabilities[0][3])*100:
       dominant_emotion = 'neutral'

   if mx == float(probabilities[0][4])*100:
       dominant_emotion = 'worry'

   if mx == float(probabilities[0][5])*100:
       dominant_emotion = 'surprise'

   if mx == float(probabilities[0][6])*100:
       dominant_emotion = 'love'

   if mx == float(probabilities[0][7])*100:
       dominant_emotion = 'fun'

   if mx == float(probabilities[0][8])*100:
       dominant_emotion = 'hate'

   if mx == float(probabilities[0][9])*100:
       dominant_emotion = 'happiness'

   if mx == float(probabilities[0][10])*100:
       dominant_emotion = 'boredom'

   if mx == float(probabilities[0][11])*100:
       dominant_emotion = 'relief'

   if mx == float(probabilities[0][12])*100:
       dominant_emotion = 'anger'

   print(dominant_emotion)

   print({'empty': float(probabilities[0][0])*100,
           'sadness': float(probabilities[0][1])*100,
           'enthusiasm': float(probabilities[0][2])*100,
           'neutral': float(probabilities[0][3])*100,
           'worry': float(probabilities[0][4])*100,
           'surprise': float(probabilities[0][5])*100,
           'love': float(probabilities[0][6])*100,
           'fun': float(probabilities[0][7])*100,
           'hate': float(probabilities[0][8])*100,
           'happiness': float(probabilities[0][9])*100,
           'boredom': float(probabilities[0][10])*100,
           'relief': float(probabilities[0][11])*100,
           'anger': float(probabilities[0][12])*100
           })
   call()

call()