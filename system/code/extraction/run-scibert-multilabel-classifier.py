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
import ast

#----------------------------------------------------------
LABEL = 'binarized_relations'
# convert raw text into list of tokens using tokenizer
MODEL_NAME = 'scibert'
LOSS_NAME = 'weightedBCE'
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
    labels = list(ast.literal_eval(data_row[LABEL]))

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
      'sentence': sentence,
      'input_ids': input_ids ,
      'attention_mask': attention_mask,
      'labels': torch.tensor(labels, dtype= torch.long)
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
        self.classifier = nn.Linear(self.bert.config.hidden_size, 4)
        self.relu =  nn.ReLU()
        
        # Normal binary cross entropy (from single label classification)
        # self.loss = nn.NLLLoss()
        # self.softmax = nn.LogSoftmax(dim=1)

        # Binary cross entropy (multilabel classification)
        # Since the output is multi-label (multiple tags associated with a question), we may tend to use a Sigmoid activation function for the final output
        # and a Binary Cross-Entropy loss function. However, the Pytorch documentation recommends using the BCEWithLogitsLoss () function which combines a 
        # Sigmoid layer and the BCELoss in one single class instead of having a plain Sigmoid followed by a BCELoss.
        # self.loss = nn.BCEWithLogitsLoss()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.full((1, 4), 2))

        # # Custom loss: soft F1 loss
        # self.loss = SoftF1Loss()
        # self.softmax = nn.Softmax(dim=1)

        # self.train_acc = torchmetrics.Accuracy()
        # self.val_acc = torchmetrics.Accuracy()

    def forward(self, input_ids, attention_mask, labels=None):
      # pass the inputs to the model
      last_hs, cls_hs = self.bert(input_ids, attention_mask=attention_mask)
      # output = last_hs.max(dim=1)[0] # max pooling
      output = self.classifier(cls_hs)
      loss = 0
      if labels is not None:
          loss = self.loss(output, labels.float())
      return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        # log step metric
        # self.log('train_acc', self.train_acc(outputs.argmax(dim=1), labels), on_step=True, on_epoch=False)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        # log step metric
        # self.log('val_acc', self.val_acc(outputs.argmax(dim=1), labels), on_step=True, on_epoch=True)
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
# Plot precision-recall curves
def plotPrecisionRecallCurve(Y_test, y_score, output_dir, output_file):
  n_classes = Y_test.shape[1]
  # For each class
  precision = dict()
  recall = dict()
  average_precision = dict()
  for i in range(n_classes):
      precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                          y_score[:, i])
      average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

  # A "micro-average": quantifying score on all classes jointly
  precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
      y_score.ravel())
  average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                      average="micro")
  print('Average precision score, micro-averaged over all classes: {0:0.2f}'
        .format(average_precision["micro"]))
  #----------------------------------------------------------
  # # Plot the micro-averaged Precision-Recall curve¶
  # plt.figure()
  # plt.step(recall['micro'], precision['micro'], where='post')

  # plt.xlabel('Recall')
  # plt.ylabel('Precision')
  # plt.ylim([0.0, 1.05])
  # plt.xlim([0.0, 1.0])
  # plt.title(
  #     'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
  #     .format(average_precision["micro"]))
  # plt.savefig("micro-averaged-precision-recall-curve.png")
  #----------------------------------------------------------
  # Plot Precision-Recall curve for each class and iso-f1 curves
  from itertools import cycle
  # setup plot details
  colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

  plt.figure(figsize=(7, 8))
  f_scores = np.linspace(0.2, 0.8, num=4)
  lines = []
  labels = []
  for f_score in f_scores:
      x = np.linspace(0.01, 1)
      y = f_score * x / (2 * x - f_score)
      l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
      plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

  lines.append(l)
  labels.append('iso-f1 curves')
  l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
  lines.append(l)
  labels.append('micro-average Precision-recall (area = {0:0.2f})'
                ''.format(average_precision["micro"]))

  for i, color in zip(range(n_classes), colors):
      l, = plt.plot(recall[i], precision[i], color=color, lw=2)
      lines.append(l)
      labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                    ''.format(i, average_precision[i]))

  fig = plt.gcf()
  fig.subplots_adjust(bottom=0.25)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title(output_file)
  plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

  plt.savefig("./precision-recall-curves/%s/%s.png" % (output_dir,output_file))


#----------------------------------------------------------
def runModel(train_df, val_df, test_df, model_seed, num_epochs, learning_rate, batch_size):
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
    model_seed=model_seed,
    num_epochs=num_epochs,
    n_warmup_steps=warmup_steps,
    n_training_steps=total_training_steps,
    learning_rate=learning_rate,
    batch_size=batch_size
    )

  # set output directories for checkpoints and logger to model name + loss type + embedding type
  output_dir = "%s-%s-%s/%s" % (MODEL_NAME, LOSS_NAME, EMBEDDING_TYPE, experiment)
  # set output filename to hyperparam combo
  output_file = "seed=%d-epochs=%d-lr=%f-bs=%d-%s" % (model_seed, num_epochs, learning_rate, batch_size, experiment)
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
    progress_bar_refresh_rate=20,
    track_grad_norm=2
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

  for item in tqdm(dataset):
    _, prediction = trained_model(
      item["input_ids"].unsqueeze(dim=0).to(device), 
      item["attention_mask"].unsqueeze(dim=0).to(device)
    )
    sentences.append(item["sentence"])
    predictions.append(prediction.flatten())
    # labels.append(item["labels"].int())

  predictions = torch.stack(predictions).detach().cpu().numpy()
  # upper, lower = 1, 0
  # predictions = np.where(predictions > 0.5, upper, lower)

  # labels = torch.stack(labels).detach().cpu().numpy()

  # output preds
  df_preds = pd.DataFrame(list(zip(sentences, predictions)), 
  columns =['sentence', 'scibert_pred'])
  df_preds.to_csv("./predictions/%s/%s.csv" % (output_dir, output_file))

  # print(classification_report(labels, predictions, zero_division=0))

  # report = classification_report(labels, predictions, output_dict=True, zero_division=0)

  # return report

#----------------------------------------------------------
# Train set: use ALL annotations (both cv and held out test sets since we're running on the 170k new sentences)
df = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/annotations-round2/union-1sentence-both2sentence/union-multilabel-data-§-quality-control-with-contrast-class.csv")
train_df = df[:3510]
val_df = df[3511:]
print(len(train_df))
print(len(val_df))
# Test set: all 2sentence-1concepts rows from the 6 batches of s2orc metadata
test_df = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/inputs/concepts-from-both-sentences/all-2sentence-1concept-rows-demarcated.csv")
# test_df = test_df[:100]
test_df['binarized_relations'] = ["[0, 0, 0]"] * len(test_df)
print(len(test_df))

# run best model and test on held-out test set
# best params
num_epochs = 10
learning_rate = 5e-5
batch_size = 32
model_seed = 42
pl.seed_everything(model_seed)

results = {}
experiment = "best-params-all-s2orc"

output_dir = "%s-%s-%s/%s" % (MODEL_NAME, LOSS_NAME, EMBEDDING_TYPE, experiment)
this_combo = "seed=%d-epochs=%d-lr=%f-bs=%d-%s" % (model_seed, num_epochs, learning_rate, batch_size, experiment)

# run model
print("running model...")
runModel(train_df, val_df, test_df, model_seed, num_epochs, learning_rate, batch_size)

# DELETE checkpoint regardless
print("deleting checkpoint for combo %s..." % this_combo)
if path.exists("./checkpoints/%s/%s.ckpt" % (output_dir, this_combo)):
  os.remove("./checkpoints/%s/%s.ckpt" % (output_dir, this_combo))


