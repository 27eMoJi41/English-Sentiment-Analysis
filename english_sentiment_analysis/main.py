import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AdamW,AutoTokenizer,AutoModel
import string
from transformers import logging
import warnings
from tqdm import tqdm

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

batch_size = 40
epochs = 7

sw = set(stopwords.words("english"))
p = set(string.punctuation)

df = pd.read_csv('tweet_emotions.csv')
df = df[["sentiment"]+["content"]]

def preprocess(text):
   tokens = word_tokenize(text.lower())
   filtered_tokens = [token for token in tokens if token not in sw and token not in p]
   lemmatizer = WordNetLemmatizer()
   lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
   processed_text = ' '.join(lemmatized_tokens)
   for word in processed_text.split():
      if word.startswith('pic') or word.startswith('http') or word.startswith('www'):
         continue
      else:
         processed_text += word+' '
   return processed_text

df['content'] = df['content'].apply(preprocess)

for i in df.index:
   if df.sentiment.iloc[i] == 'empty':
      df.at[i, 'sentiment'] = 0
   elif df.sentiment.iloc[i] == 'sadness':
      df.at[i, 'sentiment'] = 1
   elif df.sentiment.iloc[i] == 'enthusiasm':
      df.at[i, 'sentiment'] = 2
   elif df.sentiment.iloc[i] == 'neutral':
      df.at[i, 'sentiment'] = 3
   elif df.sentiment.iloc[i] == 'worry':
      df.at[i, 'sentiment'] = 4
   elif df.sentiment.iloc[i] == 'surprise':
      df.at[i, 'sentiment'] = 5
   elif df.sentiment.iloc[i] == 'love':
      df.at[i, 'sentiment'] = 6
   elif df.sentiment.iloc[i] == 'fun':
      df.at[i, 'sentiment'] = 7
   elif df.sentiment.iloc[i] == 'hate':
      df.at[i, 'sentiment'] = 8
   elif df.sentiment.iloc[i] == 'happiness':
      df.at[i, 'sentiment'] = 9
   elif df.sentiment.iloc[i] == 'boredom':
      df.at[i, 'sentiment'] = 10
   elif df.sentiment.iloc[i] == 'relief':
      df.at[i, 'sentiment'] = 11
   elif df.sentiment.iloc[i] == 'anger':
      df.at[i, 'sentiment'] = 12

df = shuffle(df)

train_text, temp_text, train_labels, temp_labels = train_test_split(df['content'], df['sentiment'])
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels)

df = df.reset_index(drop=True)

tokenizer = AutoTokenizer.from_pretrained("maymuni/bert-base-turkish-cased-emotion-analysis")
bert = AutoModel.from_pretrained("maymuni/bert-base-turkish-cased-emotion-analysis",return_dict=False)

tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

tokens_test = tokenizer.batch_encode_plus(
   test_text.tolist(),
   pad_to_max_length=True,
   truncation=True,
   return_token_type_ids=False
)

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

for param in bert.parameters():
    param.requires_grad = False

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
      _,cls_hs = self.bert(sent_id, attention_mask=mask,return_dict=False)
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc3(x)
      x = self.softmax(x)

      return x

model = Arch(bert)

optimizer = AdamW(model.parameters(), lr = 0.0001)
class_wts = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
weights= torch.tensor(class_wts,dtype=torch.float)
cross_entropy  = nn.CrossEntropyLoss(weight=weights)

def train():
   model.train()

   total_loss, total_accuracy = 0, 0
   total_preds = []

   for step, batch in tqdm(enumerate(train_dataloader)):
      if step % 50 == 0:
         print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

      batch = [r for r in batch]
      sent_id, mask, labels = batch
      preds = model(sent_id, mask)
      loss = cross_entropy(preds, labels)
      total_loss = total_loss + loss.item()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      optimizer.zero_grad()
      preds = preds.detach().cpu().numpy()
      total_preds.append(preds)

   avg_loss = total_loss / len(train_dataloader)
   total_preds = np.concatenate(total_preds, axis=0)
   return avg_loss, total_preds


def evaluate():
   print("\nEvaluating...")

   model.eval()

   total_loss, total_accuracy = 0, 0
   total_preds = []

   for step, batch in tqdm(enumerate(val_dataloader)):
      if step % 50 == 0:
         print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

      batch = [t for t in batch]
      sent_id, mask, labels = batch

      with torch.no_grad():
         preds = model(sent_id, mask)
         loss = cross_entropy(preds, labels)
         total_loss = total_loss + loss.item()
         preds = preds.detach().cpu().numpy()
         total_preds.append(preds)

   avg_loss = total_loss / len(val_dataloader)
   total_preds = np.concatenate(total_preds, axis=0)

   return avg_loss, total_preds

def early_stopping(train_loss, validation_loss, min_delta, tolerance):
   counter = 0
   if (validation_loss - train_loss) > min_delta:
      counter += 1
      if counter >= tolerance:
         return True

best_valid_loss = float('inf')

train_losses=[]
valid_losses=[]

for epoch in range(epochs):
   print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

   train_loss,_ = train()
   valid_loss,_ = evaluate()

   if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
      torch.save(model.state_dict(), 'emotion_analysis.pt')

   train_losses.append(train_loss)
   valid_losses.append(valid_loss)

   print(f'\nTraining Loss: {train_loss:.3f}')
   print(f'Validation Loss: {valid_loss:.3f}')

   if early_stopping(train_loss, valid_loss, min_delta=10, tolerance=20):
      print("We are at epoch:", epoch)
      break

print("---||Training Completed||---")

with torch.no_grad():
  preds = model(test_seq, test_mask)
  preds = preds.detach().cpu().numpy()

predicted_label = []
for pred in preds:
  predicted_label.append(np.argmax(pred))

target_names = ['empty','sadness','enthusiasm','neutral','worry','surprise','love','fun','hate','happiness','boredom','relief','anger']
print(classification_report(test_y, predicted_label, target_names=target_names))

def predict_emotion(text):
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
   preds = torch.tensor(preds)
   probabilities = nn.functional.softmax(preds)

   return {'empty': float(probabilities[0][0]),
           'sadness': float(probabilities[0][1]),
           'enthusiasm': float(probabilities[0][2]),
           'neutral': float(probabilities[0][3]),
           'worry': float(probabilities[0][4]),
           'surprise': float(probabilities[0][5]),
           'love': float(probabilities[0][6]),
           'fun': float(probabilities[0][7]),
           'hate': float(probabilities[0][8]),
           'happiness': float(probabilities[0][9]),
           'boredom': float(probabilities[0][10]),
           'relief': float(probabilities[0][11]),
           'anger': float(probabilities[0][12])
           }

print("Test Predicton:",predict_emotion("I am feeling very alone and sad"))