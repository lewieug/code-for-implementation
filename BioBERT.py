
# Classes for storing individual sentences:

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 e11_p, e12_p, e21_p, e22_p,
                 e1_mask, e2_mask,
                 segment_ids,
                 label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

        # add enitity position and entity mask for BERT
        self.e11_p = e11_p
        self.e12_p = e12_p
        self.e21_p = e21_p
        self.e22_p = e22_p
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask

    def print_contents(self):
        print(self.input_ids, self.input_mask, self.segment_ids, self.label_id,
              self.e11_p, self.e12_p, self.e21_p,
              self.e22_p, self.e1_mask, self.e2_mask)


# Functions for reading in the data:

import csv
import sys
import logging

logger = logging.getLogger(__name__)


def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(cell for cell in line)
            lines.append(line)
        return lines


def create_examples(lines, set_type):
    """Creates examples for the training and test sets.

    """
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, i)
        logger.info(line)
        text_a = line[1]
        text_b = None
        label = line[2]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


def get_train_examples(data_dir):
    logger.info("LOOKING AT {}".format(
        os.path.join(data_dir, "train.tsv")))
    return create_examples(
        read_tsv(os.path.join(data_dir, "train.tsv")), "train")


def get_test_examples(data_dir):
    return create_examples(
        read_tsv(os.path.join(data_dir, "test.tsv")), "test")

from pytorch_transformers import WEIGHTS_NAME, BertConfig, BertTokenizer

# Configuration parameters:
use_entity_indicator=True
max_seq_len=176

tokenizer = BertTokenizer(vocab_file='E:/THESIS/scibert/BERT-Relation-Extraction/additional_models/biobert_v1.1_pubmed/vocab.txt',
                                           do_lower_case=False).from_pretrained('bert-base-uncased', do_lower_case=True)


n_labels = 18
labels = [str(i) for i in range(n_labels)]


# BERT Class for converting the input to features according to the required input form
def convert_examples_to_features(examples, label_list, max_seq_len,
                                 tokenizer,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    ''' In: sentences with entities marked by $$ and ## around them
      Out: sentence represented as object of the InputFeature class '''

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        # convert the entity information to features as well
        l = len(tokens_a)

        # the start position of entity1:
        e11_p = tokens_a.index("#") + 1
        # the end position of entity1
        e12_p = l - tokens_a[::-1].index("#") + 1
        # the start position of entity2
        e21_p = tokens_a.index("$") + 1
        # the end position of entity2
        e22_p = l - tokens_a[::-1].index("$") + 1

        tokens_b = None

        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3".
            special_tokens_count = 3
            _truncate_seq_pair(tokens_a, tokens_b,
                               max_seq_len - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "
            special_tokens_count = 2
            if len(tokens_a) > max_seq_len - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_len - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + \
                     ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + \
                      ([pad_token_segment_id] * padding_length)

        # add attention mask for entities as well
        e1_mask = [0 for i in range(len(input_mask))]

        e2_mask = [0 for i in range(len(input_mask))]

        for i in range(e11_p, e12_p):
            e1_mask[i] = 1
        for i in range(e21_p, e22_p):
            e2_mask[i] = 1

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        label_id = int(example.label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
            if use_entity_indicator:
                logger.info("e11_p: %s" % e11_p)
                logger.info("e12_p: %s" % e12_p)
                logger.info("e21_p: %s" % e21_p)
                logger.info("e22_p: %s" % e22_p)
                logger.info("e1_mask: %s" %
                            " ".join([str(x) for x in e1_mask]))
                logger.info("e2_mask: %s" %
                            " ".join([str(x) for x in e2_mask]))
            logger.info("segment_ids: %s" %
                        " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, e11_p=e11_p, e12_p=e12_p, e21_p=e21_p,
                                      e22_p=e22_p,
                                      e1_mask=e1_mask, e2_mask=e2_mask, segment_ids=segment_ids, label_id=label_id))
    return features


import os

# Get the training data from the data folder, hosted on google drive:
data_folder = 'E:/THESIS/scibert/anlp_rbert/'
examples = get_train_examples(data_folder)
features = convert_examples_to_features(
    examples, labels, max_seq_len, tokenizer)

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset

all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
all_input_mask = torch.tensor(
    [f.input_mask for f in features], dtype=torch.long)
all_segment_ids = torch.tensor(
    [f.segment_ids for f in features], dtype=torch.long)

#also for entities
all_e1_mask = torch.tensor(
    [f.e1_mask for f in features], dtype=torch.long)
all_e2_mask = torch.tensor(
    [f.e2_mask for f in features], dtype=torch.long)

all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)

dataset = TensorDataset(all_input_ids, all_input_mask,
                            all_segment_ids, all_label_ids, all_e1_mask, all_e2_mask)

# Configuration parameters:

# batch size (low to save memory):
per_gpu_train_batch_size = 4
n_gpu = torch.cuda.device_count()

# the base BERT model (smaller, to save memory)
pretrained_model_name='bert-base-uncased'

# parameters for gradient descent:
max_steps=-1
gradient_accumulation_steps=1

# Number of training epochs:
num_train_epochs=5.0

# Name of task for Bert:
task_name = 'semeval'

# hyperparameter for regularization
l2_reg_lambda=5e-3
local_rank=-1
no_cuda=False

train_batch_size = per_gpu_train_batch_size * \
        max(1, n_gpu)

# For sampling during the training:
train_sampler = RandomSampler(dataset)
train_dataloader = DataLoader(
        dataset, sampler=train_sampler, batch_size=train_batch_size)

# total number of steps for training:
t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

# load
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import (BertModel, BertPreTrainedModel, BertTokenizer)
from torch.nn import MSELoss, CrossEntropyLoss

def l2_loss(parameters):
  '''Calculates L2 loss (euclidian length) of 'parameters' vector.'''
  return torch.sum(   torch.tensor([torch.sum(p ** 2) / 2 for p in parameters if p.requires_grad ]))


# Huggingface Transformers Class for BERT Sequence Classification
class BertForSequenceClassification(BertPreTrainedModel):
    """
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode(
            "Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.l2_reg_lambda = config.l2_reg_lambda
        self.bert = BertModel(config)
        self.latent_entity_typing = config.latent_entity_typing
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        classifier_size = config.hidden_size*3
        self.classifier = nn.Linear(
            classifier_size, self.config.num_labels)
        self.latent_size = config.hidden_size
        self.latent_type = nn.Parameter(torch.FloatTensor(
            3, config.hidden_size), requires_grad=True)

        self.init_weights()

    # Customized forward step, for relation extraction
    # Does the extra steps required, as described in the paper.
    # Enriching Pre-trained Language Model with Entity Information for Relation Classification https://arxiv.org/abs/1905.08284.

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, e1_mask=None, e2_mask=None, labels=None,
                position_ids=None, head_mask=None):

        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]
        sequence_output = outputs[0]

        def extract_entity(sequence_output, e_mask):
            extended_e_mask = e_mask.unsqueeze(1)
            extended_e_mask = torch.bmm(
                extended_e_mask.float(), sequence_output).squeeze(1)
            return extended_e_mask.float()

        e1_h = extract_entity(sequence_output, e1_mask)
        e2_h = extract_entity(sequence_output, e2_mask)
        context = self.dropout(pooled_output)
        pooled_output = torch.cat([context, e1_h, e2_h], dim=-1)

        # Extra logit layer on top of BERT,  in order to do relation extraction:
        logits = self.classifier(pooled_output)

        # add hidden states and attention
        outputs = (logits,) + outputs[2:]

        device = logits.get_device()
        l2 = l2_loss(self.parameters())

        if device >= 0:
            l2 = l2.to(device)
        loss = l2 * self.l2_reg_lambda
        if labels is not None:

            # transform to plausible probabilities,  between 0 and 1:
            probabilities = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

            # Do one hot encoding:
            one_hot_labels = F.one_hot(labels, num_classes=self.num_labels)
            if device >= 0:
                one_hot_labels = one_hot_labels.to(device)

            # Calculate loss:
            dist = one_hot_labels[:, 1:].float() * log_probs[:, 1:]
            example_loss_except_other, _ = dist.min(dim=-1)
            per_example_loss = - example_loss_except_other.mean()

            rc_probabilities = probabilities - probabilities * one_hot_labels.float()
            second_pre,  _ = rc_probabilities[:, 1:].max(dim=-1)
            rc_loss = - (1 - second_pre).log().mean()

            loss += per_example_loss + 5 * rc_loss

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

# Make config variable for the model:
bertconfig = BertConfig.from_pretrained(
        './additional_models/biobert_v1.1_pubmed/bert_config.json')
bertconfig.l2_reg_lambda = l2_reg_lambda
bertconfig.latent_entity_typing = False
bertconfig.num_classes = n_labels

# Load the model:
model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path='./additional_models/biobert_v1.1_pubmed.bin',
                                      config=config,
                                      force_download=False, \
                                      model_size='bert-base-uncased',
                                      task='classification',\
                                      n_classes_=12)

# Prepare optimizer and schedule (linear warmup and decay)

from pytorch_transformers import AdamW, WarmupLinearSchedule

# Hyperparameters for the optimizer:
max_grad_norm = 1.0
learning_rate=2e-5
adam_epsilon=1e-8
warmup_steps=0
weight_decay=0.9


no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# Load optimizer and scheduler:
optimizer = AdamW(optimizer_grouped_parameters,
                  lr=learning_rate, eps=adam_epsilon)
scheduler = WarmupLinearSchedule(
    optimizer, warmup_steps=warmup_steps, t_total=t_total)

# Parallelize in case we have multiple GPUs:
if n_gpu > 1:
    model = torch.nn.DataParallel(model)

# Prepare for trainig:
from tqdm import tqdm, trange
import random
import numpy as np

#  Random seed for reproducability
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

global_step = 0
tr_loss, logging_loss = 0.0, 0.0
model.zero_grad()
train_iterator = trange(int(num_train_epochs),
                        desc="Epoch", disable=local_rank not in [-1, 0])

# put the model to the device
device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
model.to(device)

# Loops through the training set for a few epochs and backpropagate

# Collect the loss values:
loss_values = []

seed = 123456
set_seed(seed)

for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                          disable=local_rank not in [-1, 0])

    # For each epoch,  split into batches and train!

    for step, batch in enumerate(epoch_iterator):
        model.train()
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'labels': batch[3],
                  'e1_mask': batch[4],
                  'e2_mask': batch[5],
                  }

        outputs = model(**inputs)
        # model outputs are always tuple in transformers

        loss = outputs[0]

        # Collect the loss:
        loss_values.append(loss)

        if n_gpu > 1:
            loss = loss.mean()
            # mean() to average on multi-gpu parallel training
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        # Back propagate
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm)

        tr_loss += loss.item()
        if (step + 1) % gradient_accumulation_steps == 0:
            # Take a step!
            optimizer.step()
            scheduler.step()
            # Update learning rate schedule
            model.zero_grad()
            global_step += 1

        if max_steps > 0 and global_step > max_steps:
            # We're done!
            epoch_iterator.close()
            break
    if max_steps > 0 and global_step > max_steps:
        # We're done!
        train_iterator.close()


# Save the trained model:
torch.save(model.state_dict(),'E:/THESIS/scibert/prev trial/model_train1')
# Load the model
state_dict = torch.load('E:/THESIS/scibert/prev trial/model_train1')


# Fix the format on the state_dict:

# create new OrderedDict that does not contain `module.`
##from collections import OrderedDict
#new_state_dict = OrderedDict()
#for k, v in state_dict.items():
 #   name = k[7:] # remove `module.`
  #  new_state_dict[name] = v

device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")

# Load the saved model from the state dict: pretrained_model_name
model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path='./additional_models/biobert_v1.1_pubmed.bin',
                                      config=config,
                                      force_download=False, \
                                      model_size='bert-base-uncased',
                                      task='classification',\
                                      n_classes_=12)
model.load_state_dict(state_dict)
model.to(device)




# Metrics for evaluation (accuracy, f1 score),  from the official script for SemEval task-8

def acc_and_f1(preds, labels, average='macro'):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    return {"acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2}


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

# Evaluation

def evaluate(model, tokenizer, prefix=""):
        '''
        Reads the test set, makes predictions on it, saves the predictions
        returns the predictions / truth and accuracy+f1 score.
        '''
        # Loop to handle MNLI double evaluation (matched, mis-matched)

        # What kind of task it was, for BERT:
        eval_task = task_name

        # Save the evaluation metrics into results:
        results = {}

        # Load the test set and convert to features and to tensors:
        examples = get_test_examples('E:/THESIS/scibert/anlp_rbert/')
        features = convert_examples_to_features(
            examples, labels, max_seq_len, tokenizer, "classification", use_entity_indicator)

        all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long)
        all_e1_mask = torch.tensor(
            [f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
        all_e2_mask = torch.tensor(
            [f.e2_mask for f in features], dtype=torch.long)  # add e2 mask

        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.long)

        eval_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_e1_mask,
                                     all_e2_mask)

        # Size of batch per GPU:
        eval_batch_size = per_gpu_eval_batch_size * \
                          max(1, n_gpu)

        # Sample and load data:
        eval_sampler = SequentialSampler(
            eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        print(preds)
        # Loop through the test set, batch by batch:

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3],
                          'e1_mask': batch[4],
                          'e2_mask': batch[5],
                          }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Extract the predictions from the model's output:
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        # Get the loss, prediction and results:
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        print(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)
        print(results)

        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        # Write results to file:
        output_eval_file = "E:/THESIS/scibert/anlp_rbert/eval/biobertFINAL.txt"
        with open(output_eval_file, "w") as writer:
            for key in range(670):
                writer.write("%d\t%s\n" % (key + 8001, str(RELATION_LABELS[preds[key] if preds[key] < 11 else 2])))

        return result, preds, out_label_ids


import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

RELATION_LABELS = ['causes1-causes2(e1,e2)',
'causes2-causes1(e2,e1)',
'contraindicates1-contraindicates2(e1,e2)',
'contraindicates2-contraindicates1(e2,e1)',
'location1-location2(e1,e2)',
'location2-location1(e2,e1)',
'treats1-treats2(e1,e2)',
'treats2-treats1(e2,e1)',
'diagnosed by1-diagnosed by2(e1,e2)',
'diagnosed by2-diagnosed by1(e2,e1)',
'contraindicates1-contraindicates2(e1,e2)',
'treats1-treats2(e1,e2)']

per_gpu_eval_batch_size=4

result = evaluate(model, tokenizer)
print(result)
