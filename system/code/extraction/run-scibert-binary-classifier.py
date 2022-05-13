# Hyperparameter search for scibert model
# Code borrowed from:
# https://colab.research.google.com/drive/14Ea4lIzsn5EFvPpYKtWStXEByT9qmbkj?usp=sharing#scrollTo=qAYbKDu4UR6M
# https://github.com/pnageshkar/NLP/blob/master/Medium/Multi_label_Classification_BERT_Lightning.ipynb
# https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
# https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
#----------------------------------------------------------
import pandas as pd
import numpy as np
import random
import torchmetrics
import os.path
from os import path

from tqdm.auto import tqdm
import ipdb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import transformers
from transformers import *

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.model_selection import KFold, GroupShuffleSplit
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
from matplotlib import rc

import json

#----------------------------------------------------------
LABEL = 'is_relational'
# convert raw text into list of tokens using tokenizer
MODEL_NAME = 'scibert'
LOSS_NAME = 'softf1'
EMBEDDING_TYPE = 'cls'
MODEL_PATH = 'allenai/scibert_scivocab_uncased'
NUM_TOKENS = 512
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#----------------------------------------------------------
# wrap tokenization process in a PyTorch Dataset, along with converting the labels to tensors
class RelationalDataset(Dataset):

  def __init__(self, data: pd.DataFrame, tokenizer: tokenizer, max_token_len: int = 128):
    self.tokenizer = tokenizer
    self.data = data
    self.max_token_len = max_token_len
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    sentence = data_row.sentence
    paper_id = data_row.paper_id
    labels = data_row[LABEL]

    encoding = self.tokenizer.encode_plus(
      sentence,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    input_ids = encoding['input_ids'].flatten()
    attention_mask = encoding['attention_mask'].flatten()

    return {
      'paper_id': paper_id,
      'sentence': sentence,
      'input_ids': input_ids,
      'attention_mask': attention_mask,
      'labels': torch.tensor(labels, dtype=torch.long)
    }

#----------------------------------------------------------
# wrap custom dataset into LightningDataModule
class RelationalDataModule(pl.LightningDataModule):

  def __init__(self, train_df, val_df, tokenizer, batch_size, max_token_len):
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train_df
    self.val_df = val_df
    self.tokenizer = tokenizer
    self.max_token_len = max_token_len

  def setup(self, stage=None):
    self.train_dataset = RelationalDataset(self.train_df, self.tokenizer, self.max_token_len)
    self.val_dataset = RelationalDataset(self.val_df, self.tokenizer, self.max_token_len)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2)

  def test_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2)

#----------------------------------------------------------
class RelationalClassifier(pl.LightningModule):

    def __init__(self, n_classes: int, model_seed=None, num_epochs=None, n_training_steps=None, n_warmup_steps=None, learning_rate=None, batch_size=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_PATH, return_dict=False)
        
        self.save_hyperparameters()
        
        self.model_seed = model_seed
        self.num_epochs = num_epochs
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

        self.linear1 = nn.Linear(1536,768)
        self.linear2 = nn.Linear(768,2)
        self.relu = nn.ReLU()
        
        # # Normal binary cross entropy
        # self.loss = nn.NLLLoss()
        # self.softmax = nn.LogSoftmax(dim=1)

        # Custom loss: soft F1 loss
        self.loss = SoftF1Loss()
        self.softmax = nn.Softmax(dim=1)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, input_ids, attention_mask, labels=None):
        # pass the inputs to the model
        last_hs, cls_hs = self.bert(input_ids, attention_mask=attention_mask)

        # # option 1 (maxpool): max pooling over last_hs (doesn't work because there are 0s and negatives)
        # output = last_hs.max(dim=1)[0]

        # option 2 (cls): using cls embedding
        output = cls_hs

        # # option 3 (entities): some pooling of all the tokens in the span + cls embedding
        # span_mask = input_ids==7253
        # span_mask = span_mask.cumsum(dim=1)
        # span_mask = torch.where(span_mask==1, span_mask, 0) # this mask has an extra 1 for the first entity marker

        # marker_mask = ~(input_ids==7253) # so make marker mask to identify entity demarcators
        
        # final_mask = marker_mask*span_mask # multiply to get rid of first entity marker in span mask
        
        # span = last_hs*final_mask.unsqueeze(dim=2) # get weights in last_hs for this span by sending all non-span tokens to 0
        # span = torch.sum(span, dim=1) # [32, 70, 768] --> [32, 768]
        # num_tokens = torch.sum(final_mask, dim=1) # [32, 70] --> [32]
        # mean_pooled_span = torch.div(span, num_tokens.unsqueeze(dim=1)) # get average embedding by dividing sum of token embeddings by num_tokens
        
        # # markers = last_hs*marker_mask.unsqueeze(dim=2) # get entity markers

        # output = torch.cat((cls_hs, mean_pooled_span), dim=1) # concatenate cls embedding and mean pooled ent embeddings to get [32, 1536] embedding
        # if not all(x>0 for x in num_tokens):
        #   ipdb.set_trace()

        # # for cls + pooled entity embedding
        # output = self.linear1(output)
        # output = self.relu(output)
        # output = self.linear2(output)

        # for cls embedding
        output = self.linear2(output)
        output = self.relu(output)

        output = self.softmax(output)
        loss = 0
        if labels is not None:
            loss = self.loss(output, labels)

        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        # log step metric
        self.log('train_acc', self.train_acc(outputs.argmax(dim=1), labels), on_step=True, on_epoch=False)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        # log step metric
        self.log('val_acc', self.val_acc(outputs.argmax(dim=1), labels), on_step=True, on_epoch=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):
      avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
      self.logger.experiment.add_scalar('loss',avg_loss, self.current_epoch)
      # log epoch metric
      # self.logger.experiment.add_scalar('train_acc_epoch', self.train_acc.compute(), self.current_epoch)

    def validation_epoch_end(self, outputs):
      avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
      self.logger.experiment.add_scalar('loss',avg_loss, self.current_epoch)
      # log epoch metric
      # self.logger.experiment.add_scalar('val_acc_epoch', self.val_acc.compute(), self.current_epoch)

    def configure_optimizers(self):
      optimizer = AdamW(self.parameters(), lr=self.learning_rate)

      scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.n_warmup_steps, num_training_steps=self.n_training_steps)

      return dict(
      optimizer=optimizer,
      lr_scheduler=dict(
          scheduler=scheduler,
          interval='step'
          )
      )
#----------------------------------------------------------
# custom loss function for SoftF1Loss
# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class SoftF1Loss(nn.Module):
  def __init__(self, epsilon=1e-7):
    super().__init__()
    self.epsilon = epsilon
 
  def forward(self, y_pred, y_true):
    
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    # ipdb.set_trace()
    if y_pred.ndim == 2:
        y_pred = y_pred[:,1]
    
    tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

    precision = tp / (tp + fp + self.epsilon)
    recall = tp / (tp + fn + self.epsilon)

    f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
    f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
    return 1 - f1.mean()

#----------------------------------------------------------
def runModel(train_df, val_df, test_df, num_epochs, learning_rate, batch_size, setting):
  # create instance of data module
  data_module = RelationalDataModule(train_df, val_df, tokenizer, batch_size=batch_size, max_token_len=NUM_TOKENS)
  # set up data module
  data_module.setup()

  # create an instance of our model
  # to use the scheduler, we need to calculate the number of training and warm-up steps.
  # The number of training steps per epoch is equal to number of training examples / batch size. 
  # The number of total training steps is training steps per epoch * number of epochs:
  steps_per_epoch=len(train_df) // batch_size
  total_training_steps = steps_per_epoch * num_epochs
  warmup_steps = total_training_steps // 5 # use a fifth of the training steps for a warm-up
  print(warmup_steps, total_training_steps)

  model = RelationalClassifier(
    n_classes=len(LABEL),
    # model_seed=model_seed,
    num_epochs=num_epochs,
    n_warmup_steps=warmup_steps,
    n_training_steps=total_training_steps,
    learning_rate=learning_rate,
    batch_size=batch_size
    )

  # set output directories for checkpoints and logger to model name + loss type + embedding type
  output_dir = "%s-%s-%s/%s" % (MODEL_NAME, LOSS_NAME, EMBEDDING_TYPE, setting)
  # set output filename to hyperparam combo
  output_file = "epochs=%d-lr=%f-bs=%d-%s" % (num_epochs, learning_rate, batch_size, setting)
  print(output_dir)
  print(output_file)
  checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/%s" % output_dir,
    filename=output_file,
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
  )

  logger = TensorBoardLogger("lightning_logs/%s" % output_dir, name=output_file)

  trainer = pl.Trainer(
    logger=logger,
    callbacks=[checkpoint_callback], 
    max_epochs=num_epochs,
    gpus=1, 
    progress_bar_refresh_rate=20
    )

  trainer.fit(model, data_module)
  print("fit model")
  trainer.test()

  # load best version of the model according to val loss
  # https://github.com/pykale/pykale/pull/149#discussion_r638841687
  trained_model = RelationalClassifier.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path,
    n_classes=len(LABEL)
    )

  # put model into eval mode
  trained_model.eval()
  trained_model.freeze()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  trained_model = trained_model.to(device)

  # get predictions
  dataset = RelationalDataset(test_df, tokenizer, max_token_len=NUM_TOKENS)

  predictions = []
  labels = []
  sentences = []
  paper_ids = []

  for item in tqdm(dataset):
    _, prediction = trained_model(
      item["input_ids"].unsqueeze(dim=0).to(device), 
      item["attention_mask"].unsqueeze(dim=0).to(device)
    )
    paper_ids.append(item["paper_id"])
    sentences.append(item["sentence"])
    predictions.append(prediction.flatten())
    labels.append(item["labels"].int())

  predictions = torch.stack(predictions).detach().cpu().numpy()
  predictions = predictions[:,1]
  # predictions = np.argmax(predictions, axis = 1)

  labels = torch.stack(labels).detach().cpu().numpy()

  # output preds
  df_preds = pd.DataFrame(list(zip(paper_ids, sentences, predictions)), 
  columns =['paper_id', 'sentence', 'scibert_pred'])
  df_preds.to_csv("./predictions/%s/%s-pred-score.csv" % (output_dir, output_file))

  # print(classification_report(labels, predictions))

  # report = classification_report(labels, predictions, output_dict=True)

  # return report

#----------------------------------------------------------

# # data to split into train and validation sets
# cv_df = pd.read_csv("../annotations-round2/1-sentence-final-§-cv.csv")
# # drop columns with null values
# cv_df = cv_df[cv_df['is_relational'].notna()]

# # define groups for sklearn GroupShuffleSplit
# groups = cv_df['paper_id']

# output_dir = "%s-%s-%s/learning-curve-all-sentences/" % (MODEL_NAME, LOSS_NAME, EMBEDDING_TYPE)

# # for this combo of parameters and this train_size
# for train_size in train_sizes:
#   # define split
#   gss = GroupShuffleSplit(n_splits=2, train_size=train_size, test_size=0.1, random_state=42)
#   # reset results
#   results = {}

#   # go through all model seeds
#   for model_seed in range(1,6):
#     # set model seed
#     pl.seed_everything(model_seed)

#     cv_fold = 0 # count number of folds
#     # split into 5 partitions of train/val with fixed cv seed
#     for train_idx, test_idx in gss.split(X=cv_df, groups=groups):
#       cv_fold += 1
#       train_df = cv_df.iloc[train_idx]
#       val_df = cv_df.iloc[test_idx]
#       this_combo = "seed=%d-fold=%d-epochs=%d-lr=%f-bs=%d-trainsize=%.2f" % (model_seed, cv_fold, num_epochs, learning_rate, batch_size, train_size)
#       print(this_combo)
#       print(len(train_df))
#       print(len(val_df))

#       # if there was already a classification report made for this combo,
#       # don't run model again for it
#       if path.exists("./classification_reports/%s/%s.json" % (output_dir, this_combo)):
#         print("found path")
#       # if this combo does NOT have a classification report, run model, save report
#       else:
#         print("path not found, running model...")
#         report = runModel(train_df, val_df, model_seed, cv_fold, num_epochs, learning_rate, batch_size, train_size)
#         results[this_combo] = report
#         with open('./classification_reports/%s/%s.json' % (output_dir, this_combo), 'w') as outfile:
#           json.dump(results, outfile)

#       # DELETE checkpoint regardless
#       print("deleting checkpoint for combo %s..." % this_combo)
#       if path.exists("./checkpoints/%s/%s.ckpt" % (output_dir, this_combo)):
#         os.remove("./checkpoints/%s/%s.ckpt" % (output_dir, this_combo))




#----------------------------------------------------------

# data to split into train and validation sets
df = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/annotations-round2/union-1sentence-both2sentence/union-binary-label-data-§.csv")
df = df[df['is_relational'].notna()] # drop columns with null values

train_df = df[:3410]
val_df = df[3411:]
#------------------------
test_df = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/inputs/concepts-from-both-sentences/all-2sentence-1concept-rows-demarcated.csv")
# test_df = test_df[:100]
test_df['is_relational'] = [0] * len(test_df)
#------------------------

# get paper_ids of s2orc sentences that were used in data augmentation and do not use those 
df_compare = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/inputs/augmented-data/supplementary-compare-sentences.csv")
df_partof = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/inputs/augmented-data/supplementary-partof-sentences.csv")
df_usedfor = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/inputs/augmented-data/supplementary-usedfor-sentences.csv")

# remove augmentation sentence paper_ids from test set
augmentation_ids_to_exclude = df_compare['paper_ids'].tolist() + df_partof['paper_ids'].tolist() + df_usedfor['paper_ids'].tolist()
test_df = test_df[~test_df['paper_id'].isin(augmentation_ids_to_exclude)]

# remove training/val set ids from test set
train_val_ids_to_exclude = df['paper_id']
test_df = test_df[~test_df['paper_id'].isin(train_val_ids_to_exclude)]

#------------------------

# best params
num_epochs = 10
learning_rate = 1e-5
batch_size = 16
setting = 'all-s2orc'

results = {}

output_dir = "%s-%s-%s/%s" % (MODEL_NAME, LOSS_NAME, EMBEDDING_TYPE, setting)
this_combo = "epochs=%d-lr=%f-bs=%d-%s" % (num_epochs, learning_rate, batch_size, setting)
pl.seed_everything(42)

print("running model...")
runModel(train_df, val_df, test_df, num_epochs, learning_rate, batch_size, setting)

# DELETE checkpoint regardless
print("deleting checkpoint for combo %s..." % this_combo)
if path.exists("./checkpoints/%s/%s.ckpt" % (output_dir, this_combo)):
  os.remove("./checkpoints/%s/%s.ckpt" % (output_dir, this_combo))