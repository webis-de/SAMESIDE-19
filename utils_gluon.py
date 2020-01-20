# Imports
import datetime
import os
import pickle
import time
from itertools import product

import gluonnlp as nlp
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import seaborn as sns

from bert import *
from gluonnlp.data import BERTSentenceTransform
from mxboard import SummaryWriter
from mxnet import gluon
from mxnet.gluon import Block
from mxnet.gluon import nn
from mxnet.gluon.data import Dataset, SimpleDataset

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn import utils

from utils_data import Timer
from tqdm import tqdm


# ############################################################################
# Utils


# ############################################################################
# Models

# BERT
#   https://gluon-nlp.mxnet.io/examples/sentence_embedding/bert.html
#   
#   
#   
#   
#   

class MyBERTDataset(SimpleDataset):
    def __init__(self, X, y=None):
        self._X = X
        self._y = y
        super(MyBERTDataset, self).__init__(self._convert())

    def _convert(self):
        allsamples = list()

        if self._y is not None:
            df = self._X.merge(self._y, left_index=True, right_index=True)
            for _, row in df.iterrows():
                # allsamples.append([
                #     row['argument1'], row['argument2'],
                #     "1" if str(row['is_same_side']) == "True" else "0"
                # ])
                allsamples.append([
                    row['argument1'], row['argument2'],
                    1 if str(row['is_same_side']) == "True" else 0
                ])

        else:
            for _, row in self._X.iterrows():
                allsamples.append([row['argument1'], row['argument2'], None])

        return allsamples


# ############################################################################


class LastPartBERTSentenceTransform(BERTSentenceTransform):
    def __init__(self, tokenizer, max_seq_length, pad=True, pair=True):
        super(LastPartBERTSentenceTransform, self).__init__(tokenizer, max_seq_length, pad=pad, pair=pair)


    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length.
        Removes from end of token list."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop(0)
            else:
                tokens_b.pop(0)


class FirstAndLastPartBERTSentenceTransform(BERTSentenceTransform):
    def __init__(self, tokenizer, max_seq_length, pad=True, pair=True):
        super(FirstAndLastPartBERTSentenceTransform,
              self).__init__(tokenizer, max_seq_length, pad=pad, pair=pair)

    def __call__(self, line):
        # convert to unicode
        text_a = line[0]
        if self._pair:
            assert len(line) == 2
            text_b = line[1]

        tokens_a = self._tokenizer(text_a)
        tokens_a_epi = tokens_a.copy()
        tokens_b = None
        tokens_b_epi = None

        if self._pair:
            tokens_b = self._tokenizer(text_b)
            tokens_b_epi = tokens_b.copy()

        if tokens_b:
            self._truncate_seq_pair_prolog(tokens_a, tokens_b,
                                           self._max_seq_length - 3)
            self._truncate_seq_pair_epilog(tokens_a_epi, tokens_b_epi,
                                           self._max_seq_length - 3)
        else:
            if len(tokens_a) > self._max_seq_length - 2:
                tokens_a = tokens_a[0:(self._max_seq_length - 2)]
            if len(tokens_a_epi) > self._max_seq_length - 2:
                tokens_a_epi = tokens_a_epi[0:(self._max_seq_length - 2)]

        vocab = self._tokenizer.vocab
        tokens, tokens_epi = [], []
        tokens.append(vocab.cls_token)
        tokens_epi.append(vocab.cls_token)
        tokens.extend(tokens_a)
        tokens_epi.extend(tokens_a_epi)
        tokens.append(vocab.sep_token)
        tokens_epi.append(vocab.sep_token)
        segment_ids = [0] * len(tokens)
        segment_ids_epi = [0] * len(tokens_epi)

        if tokens_b:
            tokens.extend(tokens_b)
            tokens_epi.extend(tokens_b_epi)
            tokens.append(vocab.sep_token)
            tokens_epi.append(vocab.sep_token)
            segment_ids.extend([1] * (len(tokens) - len(segment_ids)))
            segment_ids_epi.extend([1] * (len(tokens) - len(segment_ids_epi)))

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        input_ids_epi = self._tokenizer.convert_tokens_to_ids(tokens_epi)
        valid_length = len(input_ids)
        valid_length_epi = len(input_ids_epi)

        if self._pad:
            padding_length = self._max_seq_length - valid_length
            padding_length_epi = self._max_seq_length - valid_length_epi
            input_ids.extend([vocab[vocab.padding_token]] * padding_length)
            input_ids_epi.extend([vocab[vocab.padding_token]] *
                                 padding_length_epi)
            segment_ids.extend([0] * padding_length)
            segment_ids_epi.extend([0] * padding_length_epi)

        return np.array(input_ids, dtype='int32'), np.array(valid_length, dtype='int32'),\
            np.array(segment_ids, dtype='int32'), np.array(input_ids_epi, dtype='int32'),\
            np.array(valid_length_epi, dtype='int32'), np.array(segment_ids_epi, dtype='int32')

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length, poploc=-1):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop(poploc)
            else:
                tokens_b.pop(poploc)

    def _truncate_seq_pair_prolog(self, tokens_a, tokens_b, max_length):
        self._truncate_seq_pair(tokens_a, tokens_b, max_length, -1)

    def _truncate_seq_pair_epilog(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length.
        Removes from end of token list."""
        self._truncate_seq_pair(tokens_a, tokens_b, max_length, 0)


# ############################################################################


class LastPartBERTDatasetTransform(dataset.BERTDatasetTransform):
    def __init__(self, tokenizer, max_seq_length, labels=None, pad=True, pair=True, label_dtype='float32'):
        super(LastPartBERTDatasetTransform, self).__init__(tokenizer, max_seq_length, labels=labels, pad=pad, pair=pair, label_dtype=label_dtype)
        self._bert_xform = LastPartBERTSentenceTransform(tokenizer, max_seq_length, pad=pad, pair=pair)


class FirstAndLastPartBERTDatasetTransform(dataset.BERTDatasetTransform):
    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 labels=None,
                 pad=True,
                 pair=True,
                 label_dtype='float32'):
        super(FirstAndLastPartBERTDatasetTransform,
              self).__init__(tokenizer,
                             max_seq_length,
                             labels=labels,
                             pad=pad,
                             pair=pair,
                             label_dtype=label_dtype)
        self._bert_xform = FirstAndLastPartBERTSentenceTransform(
            tokenizer, max_seq_length, pad=pad, pair=pair)

    def __call__(self, line):
        input_ids, valid_length, segment_ids, input_ids_epi, valid_length_epi, segment_ids_epi = self._bert_xform(
            line[:-1])

        label = line[-1]

        # if label is None than we are predicting unknown data
        if label is None:
            # early abort
            return input_ids, valid_length, segment_ids, input_ids_epi, valid_length_epi, segment_ids_epi
            
        if self.labels:  # for classification task
            label = self._label_map[label]
        label = np.array([label], dtype=self.label_dtype)

        return input_ids, valid_length, segment_ids, input_ids_epi, valid_length_epi, segment_ids_epi, label


# ############################################################################


class BERTProEpiClassifier(Block):
    """Model for sentence (pair) classification task with BERT.

    The model feeds token ids and token type ids into BERT to get the
    pooled BERT sequence representation, then apply a Dense layer for
    classification. Does this also for an adversarial classifier.

    Parameters
    ----------
    bert: BERTModel
        Bidirectional encoder with transformer.
    num_classes : int, default is 2
        The number of target classes.
    dropout : float or None, default 0.0.
        Dropout probability for the bert output.
    prefix : str or None
        See document of `mx.gluon.Block`.
    params : ParameterDict or None
        See document of `mx.gluon.Block`.
    """

    def __init__(self,
                 bert,
                 num_classes=2,
                 dropout=0.0,
                 prefix=None,
                 params=None):
        super(BERTProEpiClassifier, self).__init__(prefix=prefix, params=params)
        self.bert = bert
        with self.name_scope():
            self.classifier = nn.HybridSequential(prefix=prefix)
            if dropout:
                self.classifier.add(nn.Dropout(rate=dropout))
            self.classifier.add(nn.Dense(units=num_classes))

    def forward(self,
                inputs,
                token_types,
                valid_length=None,
                inputs_epi=None,
                token_types_epi=None,
                valid_length_epi=None):  # pylint: disable=arguments-differ
        """Generate the unnormalized scores for the given the input sequences.
        From both classifiers (classifier + adversarial_classifier).

        Parameters
        ----------
        inputs : NDArray, shape (batch_size, seq_length)
            Input words for the sequences.
        token_types : NDArray, shape (batch_size, seq_length)
            Token types for the sequences, used to indicate whether the word belongs to the
            first sentence or the second one.
        valid_length : NDArray or None, shape (batch_size)
            Valid length of the sequence. This is used to mask the padded tokens.
        inputs_epi : NDArray or None, shape (batch_size, seq_length)
            Input words for the sequences. If None then same as inputs.
        token_types_epi : NDArray or None, shape (batch_size, seq_length)
            Token types for the sequences, used to indicate whether the word belongs to the
            first sentence or the second one. If None then same as token_types.
        valid_length_epi : NDArray or None, shape (batch_size)
            Valid length of the sequence. This is used to mask the padded tokens.

        Returns
        -------
        outputs : NDArray
            Shape (batch_size, num_classes), outputs of classifier.
        """
        # if inputs_epi is None and token_types_epi is None:
        #     inputs_epi = inputs
        #     token_types_epi = token_types
        #     valid_length_epi = valid_length

        _, pooler_out = self.bert(inputs, token_types, valid_length)
        _, pooler_out_epi = self.bert(inputs_epi, token_types_epi, valid_length_epi)
        pooler_concat = mx.nd.concat(pooler_out, pooler_out_epi, dim=1)
        return self.classifier(pooler_concat)


# ############################################################################




# ############################################################################


def setup_bert(gpu=0, classifier=None, is_binary=True, max_seq_len=128, labels=None):
    if gpu is not None:
        ctx = mx.gpu(gpu)
    else:
        # change `ctx` to `mx.cpu()` if no GPU is available.
        ctx = mx.cpu()
    # ctx = [mx.gpu(i) for i in range(2)]
    # ctx =  mx.gpu() if mx.context.num_gpus() else mx.cpu()

    bert_base, vocabulary = nlp.model.get_model(
        'bert_12_768_12',
        dataset_name='book_corpus_wiki_en_uncased',
        pretrained=True,
        ctx=ctx,
        use_pooler=True,
        use_decoder=False,
        use_classifier=False)
    # print(bert_base)
    
    num_classes = 1 if is_binary else 2

    if classifier == "proepi":
        model = BERTProEpiClassifier(bert_base, num_classes=num_classes, dropout=0.1)
    else:
        model = bert.BERTClassifier(bert_base, num_classes=num_classes, dropout=0.1)

    # only need to initialize the classifier layer.
    model.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    model.hybridize(static_alloc=True)

    if is_binary:
        # sigmoid binary cross entropy loss for classification
        loss_function = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    else:
        # softmax cross entropy loss for classification
        loss_function = gluon.loss.SoftmaxCELoss()

    loss_function.hybridize(static_alloc=True)

    metric = mx.metric.Accuracy()

    # use the vocabulary from pre-trained model for tokenization
    bert_tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=True)

    # maximum sequence length
    # max_len = 128  # + batch_size: 32
    # 384 - 12
    # max_len = 512  # + batch_size: 6 ?

    # the labels for the two classes
    #all_labels = ["0", "1"]
    if labels is None:
        labels = [0, 1]
    
    # whether to transform the data as sentence pairs.
    # for single sentence classification, set pair=False
    if classifier == "proepi":
        transform = FirstAndLastPartBERTDatasetTransform(bert_tokenizer,
                                                          max_seq_len,
                                                          labels=labels,
                                                          label_dtype='int32',
                                                          pad=True,
                                                          pair=True)
    elif classifier == "epi":
        transform = LastPartBERTDatasetTransform(bert_tokenizer, max_seq_len,
                                                 labels=labels,
                                                 label_dtype='int32',
                                                 pad=True,
                                                 pair=True)
    else:
        transform = dataset.BERTDatasetTransform(bert_tokenizer, max_seq_len,
                                                 labels=labels,
                                                 label_dtype='int32',
                                                 pad=True,
                                                 pair=True)

    return model, vocabulary, ctx, bert_tokenizer, transform, loss_function, metric, labels


def setup_bert_pro128bce(gpu=0):
    return setup_bert(gpu=gpu, classifier="pro", is_binary=True, max_seq_len=128, labels=[0, 1])


def setup_bert_epi128bce(gpu=0):
    return setup_bert(gpu=gpu, classifier="epi", is_binary=True, max_seq_len=128, labels=[0, 1])


def setup_bert_pro512bce(gpu=0):
    return setup_bert(gpu=gpu, classifier="pro", is_binary=True, max_seq_len=512, labels=[0, 1])


def setup_bert_epi512bce(gpu=0):
    return setup_bert(gpu=gpu, classifier="epi", is_binary=True, max_seq_len=512, labels=[0, 1])


def setup_bert_proepi512bce(gpu=0):
    return setup_bert(gpu=gpu, classifier="proepi", is_binary=True, max_seq_len=512, labels=[0, 1])


# ############################################################################


def transform_dataset(X, y, transform):
    data_train_raw = MyBERTDataset(X, y)
    data_train = data_train_raw.transform(transform)
    return data_train_raw, data_train


# ############################################################################


def predict_out_to_ys(all_predictions, all_labels):
    y_true, y_pred = list(), list()

    for _, y_true_many, y_pred_many in all_predictions:
        y_true_many = y_true_many.T[0].asnumpy()
        # https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.SoftmaxCrossEntropyLoss
        # pred: the prediction tensor, where the batch_axis dimension ranges over batch size and axis dimension ranges over the number of classes.
        #y_pred_many = np.argmax(y_pred_many, axis=1).asnumpy()
        y_pred_many = y_pred_many.asnumpy()

        y_true.extend(list(y_true_many))
        y_pred.extend(list(y_pred_many))
        # TODO: convert label_id to label?
        # y_pred.extend(all_labels[c] for c in list(y_pred_many))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return y_true, y_pred


# ############################################################################


def train(model,
          data_train,
          ctx,
          metric,
          loss_function,
          sw=None,
          **kwargs):
    
    batch_size = kwargs.get("batch_size", 32)
    num_epochs = kwargs.get("epochs", 3)
    use_checkpoints = kwargs.get("use_checkpoints", True)
    checkpoint_dir = kwargs.get("checkpoint_dir", None)
    if use_checkpoints and checkpoint_dir is None:
        raise Exception("Need to specify a checkpoint_dir!")
    
    with Timer("setup training"):
        train_sampler = nlp.data.FixedBucketSampler(
            lengths=[int(item[1]) for item in tqdm(data_train)],
            batch_size=batch_size,
            shuffle=True)
        bert_dataloader = mx.gluon.data.DataLoader(data_train,
                                                   batch_sampler=train_sampler)

        trainer = gluon.Trainer(model.collect_params(), kwargs.get("optimizer", "adam"), {
            'learning_rate': kwargs.get("learning_rate", 5e-6),
            'epsilon': kwargs.get("epsilon", 1e-9)
        })

        # collect all differentiable parameters
        # grad_req == 'null' indicates no gradients are calculated (e.g. constant parameters)
        # the gradients for these params are clipped later
        params = [
            p for p in model.collect_params().values() if p.grad_req != 'null'
        ]

    log_interval = kwargs.get("log_interval", 500)
    global_step = kwargs.get("global_step", 0)
    with Timer("training"):
        stats = list()
        for epoch_id in range(num_epochs):
            if use_checkpoints:
                epoch_checkpoint_savefile = "bert.model.checkpoint{}.params".format(
                    epoch_id)
                if checkpoint_dir is not None:
                    epoch_checkpoint_savefile = os.path.join(
                        checkpoint_dir, epoch_checkpoint_savefile)
                if os.path.exists(epoch_checkpoint_savefile):
                    model.load_parameters(epoch_checkpoint_savefile, ctx=ctx)
                    print("loaded checkpoint for epoch {}".format(epoch_id))
                    continue

            with Timer("epoch {}".format(epoch_id)):
                metric.reset()
                step_loss = 0
                global_step = epoch_id * len(bert_dataloader)
                t_p = time.time()  # time keeping
                for batch_id, (token_ids, valid_length, segment_ids,
                               label) in enumerate(tqdm(bert_dataloader)):
                    global_step += 1
                    with mx.autograd.record():
                        # load data to GPU
                        token_ids = token_ids.as_in_context(ctx)
                        valid_length = valid_length.as_in_context(ctx)
                        segment_ids = segment_ids.as_in_context(ctx)
                        label = label.as_in_context(ctx)

                        # forward computation
                        out = model(token_ids, segment_ids,
                                    valid_length.astype('float32'))
                        label = label.astype('float32')
                        ls = loss_function(out, label).mean()

                    # backward computation
                    ls.backward()

                    # gradient clipping
                    trainer.allreduce_grads()
                    nlp.utils.clip_grad_global_norm(params, 1)
                    trainer.update(1)

                    step_loss += ls.asscalar()
                    out = out.sigmoid().round().astype('int32')
                    label = label.astype('int32')
                    metric.update([label], [out])
                    stats.append((metric.get()[1], ls.asscalar()))

                    if sw:
                        sw.add_scalar(tag='T-ls', value=ls.asscalar(), global_step=global_step)
                        sw.add_scalar(tag='T-acc', value=metric.get()[1], global_step=global_step)

                    if (batch_id + 1) % (log_interval) == 0:
                        print(
                            '[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f} - time {}'
                            .format(
                                epoch_id, batch_id + 1, len(bert_dataloader),
                                step_loss / log_interval,
                                trainer.learning_rate,
                                metric.get()[1],
                                datetime.timedelta(seconds=(time.time() -
                                                            t_p))))
                        t_p = time.time()
                        step_loss = 0

            if use_checkpoints:
                model.save_parameters(epoch_checkpoint_savefile)

    return stats


# ############################################################################


def train_proepi(model,
          data_train,
          ctx,
          metric,
          loss_function,
          sw=None,
          **kwargs):
    
    batch_size = kwargs.get("batch_size", 32)
    num_epochs = kwargs.get("epochs", 3)
    use_checkpoints = kwargs.get("use_checkpoints", True)
    checkpoint_dir = kwargs.get("checkpoint_dir", "data")

    with Timer("setup training"):
        train_sampler = nlp.data.FixedBucketSampler(
            lengths=[int(item[1]) for item in tqdm(data_train)],
            batch_size=batch_size,
            shuffle=True)
        bert_dataloader = mx.gluon.data.DataLoader(data_train,
                                                   batch_sampler=train_sampler)

        trainer = gluon.Trainer(model.collect_params(), kwargs.get("optimizer", "adam"), {
            'learning_rate': kwargs.get("learning_rate", 5e-6),
            'epsilon': kwargs.get("epsilon", 1e-9)
        })

        # collect all differentiable parameters
        # grad_req == 'null' indicates no gradients are calculated (e.g. constant parameters)
        # the gradients for these params are clipped later
        params = [
            p for p in model.collect_params().values() if p.grad_req != 'null'
        ]

    log_interval = kwargs.get("log_interval", 500)
    global_step = kwargs.get("global_step", 0)
    with Timer("training"):
        stats = list()
        for epoch_id in range(num_epochs):
            if use_checkpoints:
                epoch_checkpoint_savefile = "bert.model.checkpoint{}.params".format(
                    epoch_id)
                if checkpoint_dir is not None:
                    epoch_checkpoint_savefile = os.path.join(
                        checkpoint_dir, epoch_checkpoint_savefile)
                if os.path.exists(epoch_checkpoint_savefile):
                    model.load_parameters(epoch_checkpoint_savefile, ctx=ctx)
                    print("loaded checkpoint for epoch {}".format(epoch_id))
                    continue

            with Timer("epoch {}".format(epoch_id)):
                metric.reset()
                step_loss = 0
                global_step = epoch_id * len(bert_dataloader)
                t_p = time.time()  # time keeping
                for batch_id, (token_ids, valid_length, segment_ids,
                               token_ids_epi, valid_length_epi,
                               segment_ids_epi,
                               label) in enumerate(tqdm(bert_dataloader)):
                    global_step += 1
                    with mx.autograd.record():
                        # load data to GPU
                        token_ids = token_ids.as_in_context(ctx)
                        valid_length = valid_length.as_in_context(ctx)
                        segment_ids = segment_ids.as_in_context(ctx)
                        token_ids_epi = token_ids_epi.as_in_context(ctx)
                        valid_length_epi = valid_length_epi.as_in_context(ctx)
                        segment_ids_epi = segment_ids_epi.as_in_context(ctx)
                        label = label.as_in_context(ctx)

                        # forward computation
                        out = model(token_ids, segment_ids,
                                    valid_length.astype('float32'),
                                    token_ids_epi, segment_ids_epi,
                                    valid_length_epi.astype('float32'))
                        label = label.astype('float32')
                        ls = loss_function(out, label).mean()

                    # backward computation
                    ls.backward()

                    # gradient clipping
                    trainer.allreduce_grads()
                    nlp.utils.clip_grad_global_norm(params, 1)
                    trainer.update(1)

                    step_loss += ls.asscalar()
                    out = out.sigmoid().round().astype('int32')
                    label = label.astype('int32')
                    metric.update([label], [out])
                    stats.append((metric.get()[1], ls.asscalar()))

                    if sw:
                        sw.add_scalar(tag='T-ls', value=ls.asscalar(), global_step=global_step)
                        sw.add_scalar(tag='T-acc', value=metric.get()[1], global_step=global_step)

                    if (batch_id + 1) % (log_interval) == 0:
                        print(
                            '[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f} - time {}'
                            .format(
                                epoch_id, batch_id + 1, len(bert_dataloader),
                                step_loss / log_interval,
                                trainer.learning_rate,
                                metric.get()[1],
                                datetime.timedelta(seconds=(time.time() -
                                                            t_p))))
                        t_p = time.time()
                        step_loss = 0

            if use_checkpoints:
                model.save_parameters(epoch_checkpoint_savefile)

    return stats


# ############################################################################
# TODO: re-write this
# hasn't worked yet

def train_multi(model,
                data_train,
                ctx,
                metric,
                loss_function,
                batch_size=32,
                lr=5e-6,
                num_epochs=3,
                checkpoint_dir="data",
                use_checkpoints=True):
    with Timer("setup training"):
        train_sampler = nlp.data.FixedBucketSampler(
            lengths=[int(item[1]) for item in tqdm(data_train)],
            batch_size=batch_size,
            shuffle=True)
        bert_dataloader = mx.gluon.data.DataLoader(data_train,
                                                   batch_sampler=train_sampler)

        trainer = gluon.Trainer(model.collect_params(),
                                'adam', {
                                    'learning_rate': lr,
                                    'epsilon': 1e-9
                                },
                                update_on_kvstore=False)

        # collect all differentiable parameters
        # grad_req == 'null' indicates no gradients are calculated (e.g. constant parameters)
        # the gradients for these params are clipped later
        params = [
            p for p in model.collect_params().values() if p.grad_req != 'null'
        ]

    log_interval = 500
    with Timer("training"):
        stats = list()
        for epoch_id in range(num_epochs):
            if use_checkpoints:
                epoch_checkpoint_savefile = "bert.model.checkpoint{}.params".format(
                    epoch_id)
                if checkpoint_dir is not None:
                    epoch_checkpoint_savefile = os.path.join(
                        checkpoint_dir, epoch_checkpoint_savefile)
                if os.path.exists(epoch_checkpoint_savefile):
                    model.load_parameters(epoch_checkpoint_savefile, ctx=ctx)
                    print("loaded checkpoint for epoch {}".format(epoch_id))
                    continue

            with Timer("epoch {}".format(epoch_id)):
                metric.reset()
                step_loss = 0
                t_p = time.time()  # time keeping
                for batch_id, (token_ids, valid_length, segment_ids,
                               label) in enumerate(bert_dataloader):
                    with mx.autograd.record():
                        # load data to GPU
                        token_ids = gluon.utils.split_and_load(
                            token_ids, ctx, even_split=False)
                        valid_length = gluon.utils.split_and_load(
                            valid_length, ctx, even_split=False)
                        segment_ids = gluon.utils.split_and_load(
                            segment_ids, ctx, even_split=False)
                        label = gluon.utils.split_and_load(label,
                                                           ctx,
                                                           even_split=False)

                        # forward computation
                        out = [
                            model(t1, s1, v1.astype('float32'), t2, s2,
                                  v2.astype('float32'))
                            for t1, s1, v1, t2, s2, v2 in zip(
                                token_ids, segment_ids, valid_length)
                        ]
                        ls = [
                            loss_function(o, l.astype('float32')).mean()
                            for o, l in zip(out, label)
                        ]

                    # backward computation
                    for l in ls:
                        l.backward()

                    # gradient clipping
                    trainer.allreduce_grads()
                    nlp.utils.clip_grad_global_norm(params, 1)
                    trainer.update(1)

                    for l in ls:
                        step_loss += l.asscalar()
                    for o, l in zip(out, label):
                        metric.update([l.astype('int32')],
                                      [o.sigmoid().round().astype('int32')])
                    stats.append((metric.get()[1], [l.asscalar() for l in ls]))
                    if (batch_id + 1) % (log_interval) == 0:
                        print(
                            '[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f} - time {}'
                            .format(
                                epoch_id, batch_id + 1, len(bert_dataloader),
                                step_loss / log_interval,
                                trainer.learning_rate,
                                metric.get()[1],
                                datetime.timedelta(seconds=(time.time() -
                                                            t_p))))
                        t_p = time.time()
                        step_loss = 0

            if use_checkpoints:
                model.save_parameters(epoch_checkpoint_savefile)

    return stats


def train_multi_proepi(model,
                data_train,
                ctx,
                metric,
                loss_function,
                batch_size=32,
                lr=5e-6,
                num_epochs=3,
                checkpoint_dir="data",
                use_checkpoints=True):
    with Timer("setup training"):
        train_sampler = nlp.data.FixedBucketSampler(
            lengths=[int(item[1]) for item in tqdm(data_train)],
            batch_size=batch_size,
            shuffle=True)
        bert_dataloader = mx.gluon.data.DataLoader(data_train,
                                                   batch_sampler=train_sampler)

        trainer = gluon.Trainer(model.collect_params(),
                                'adam', {
                                    'learning_rate': lr,
                                    'epsilon': 1e-9
                                },
                                update_on_kvstore=False)

        # collect all differentiable parameters
        # grad_req == 'null' indicates no gradients are calculated (e.g. constant parameters)
        # the gradients for these params are clipped later
        params = [
            p for p in model.collect_params().values() if p.grad_req != 'null'
        ]

    log_interval = 500
    with Timer("training"):
        stats = list()
        for epoch_id in range(num_epochs):
            if use_checkpoints:
                epoch_checkpoint_savefile = "bert.model.checkpoint{}.params".format(
                    epoch_id)
                if checkpoint_dir is not None:
                    epoch_checkpoint_savefile = os.path.join(
                        checkpoint_dir, epoch_checkpoint_savefile)
                if os.path.exists(epoch_checkpoint_savefile):
                    model.load_parameters(epoch_checkpoint_savefile, ctx=ctx)
                    print("loaded checkpoint for epoch {}".format(epoch_id))
                    continue

            with Timer("epoch {}".format(epoch_id)):
                metric.reset()
                step_loss = 0
                t_p = time.time()  # time keeping
                for batch_id, (token_ids, valid_length, segment_ids,
                               token_ids_epi, valid_length_epi,
                               segment_ids_epi,
                               label) in enumerate(bert_dataloader):
                    with mx.autograd.record():
                        # load data to GPU
                        token_ids = gluon.utils.split_and_load(
                            token_ids, ctx, even_split=False)
                        valid_length = gluon.utils.split_and_load(
                            valid_length, ctx, even_split=False)
                        segment_ids = gluon.utils.split_and_load(
                            segment_ids, ctx, even_split=False)
                        token_ids_epi = gluon.utils.split_and_load(
                            token_ids_epi, ctx, even_split=False)
                        valid_length_epi = gluon.utils.split_and_load(
                            valid_length_epi, ctx, even_split=False)
                        segment_ids_epi = gluon.utils.split_and_load(
                            segment_ids_epi, ctx, even_split=False)
                        label = gluon.utils.split_and_load(label,
                                                           ctx,
                                                           even_split=False)

                        # forward computation
                        out = [
                            model(t1, s1, v1.astype('float32'), t2, s2,
                                  v2.astype('float32'))
                            for t1, s1, v1, t2, s2, v2 in zip(
                                token_ids, segment_ids, valid_length,
                                token_ids_epi, segment_ids_epi,
                                valid_length_epi)
                        ]
                        ls = [
                            loss_function(o, l.astype('float32')).mean()
                            for o, l in zip(out, label)
                        ]

                    # backward computation
                    for l in ls:
                        l.backward()

                    # gradient clipping
                    trainer.allreduce_grads()
                    nlp.utils.clip_grad_global_norm(params, 1)
                    trainer.update(1)

                    for l in ls:
                        step_loss += l.asscalar()
                    for o, l in zip(out, label):
                        metric.update([l.astype('int32')],
                                      [o.sigmoid().round().astype('int32')])
                    stats.append((metric.get()[1], [l.asscalar() for l in ls]))
                    if (batch_id + 1) % (log_interval) == 0:
                        print(
                            '[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f} - time {}'
                            .format(
                                epoch_id, batch_id + 1, len(bert_dataloader),
                                step_loss / log_interval,
                                trainer.learning_rate,
                                metric.get()[1],
                                datetime.timedelta(seconds=(time.time() -
                                                            t_p))))
                        t_p = time.time()
                        step_loss = 0

            if use_checkpoints:
                model.save_parameters(epoch_checkpoint_savefile)

    return stats


# ############################################################################
# predicting


def predict(model, data_predict, ctx, metric, loss_function, batch_size=32, sw=None):
    bert_dataloader = mx.gluon.data.DataLoader(data_predict,
                                               batch_size=batch_size)

    all_predictions = list()

    with Timer("prediction"):
        metric.reset()
        cum_loss = 0
        for batch_id, (token_ids, valid_length, segment_ids,
                       label) in enumerate(tqdm(bert_dataloader)):
            global_step = batch_id
            # load data to GPU
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)
            segment_ids = segment_ids.as_in_context(ctx)
            label = label.as_in_context(ctx)

            # forward computation
            out = model(token_ids, segment_ids, valid_length.astype('float32'))
            label = label.astype('float32')
            ls = loss_function(out, label).mean()

            out = out.sigmoid().round().astype('int32')
            label = label.astype('int32')
            metric.update([label], [out])
            cum_loss += ls.asscalar()  # .sum() ?

            if sw:
                sw.add_scalar(tag='P-ls', value=ls.asscalar(), global_step=global_step)
                sw.add_scalar(tag='P-acc', value=metric.get()[1], global_step=global_step)

            all_predictions.append((batch_id, label, out))

    return all_predictions, cum_loss


# ############################################################################


def predict_proepi(model, data_predict, ctx, metric, loss_function, batch_size=32, sw=None):
    bert_dataloader = mx.gluon.data.DataLoader(data_predict,
                                               batch_size=batch_size)

    all_predictions = list()

    with Timer("prediction"):
        metric.reset()
        cum_loss = 0
        for batch_id, (token_ids, valid_length, segment_ids, token_ids_epi,
                       valid_length_epi, segment_ids_epi,
                       label) in enumerate(tqdm(bert_dataloader)):
            global_step = batch_id
            # load data to GPU
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)
            segment_ids = segment_ids.as_in_context(ctx)
            token_ids_epi = token_ids_epi.as_in_context(ctx)
            valid_length_epi = valid_length_epi.as_in_context(ctx)
            segment_ids_epi = segment_ids_epi.as_in_context(ctx)
            label = label.as_in_context(ctx)

            # forward computation
            out = model(token_ids, segment_ids, valid_length.astype('float32'),
                        token_ids_epi, segment_ids_epi,
                        valid_length_epi.astype('float32'))
            label = label.astype('float32')
            ls = loss_function(out, label).mean()

            out = out.sigmoid().round().astype('int32')
            label = label.astype('int32')
            metric.update([label], [out])
            cum_loss += ls.asscalar()  # .sum() ?

            if sw:
                sw.add_scalar(tag='P-ls', value=ls.asscalar(), global_step=global_step)
                sw.add_scalar(tag='P-acc', value=metric.get()[1], global_step=global_step)

            all_predictions.append((batch_id, label, out))

    return all_predictions, cum_loss


# ############################################################################


def predict_unknown(model, data_predict, ctx, label_map=None, batch_size=32):
    bert_dataloader = mx.gluon.data.DataLoader(data_predict,
                                               batch_size=batch_size)

    predictions = list()

    with Timer("prediction"):
        for batch_id, (token_ids, valid_length, segment_ids) in enumerate(tqdm(bert_dataloader)):
            global_step = batch_id
            # load data to GPU
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)
            segment_ids = segment_ids.as_in_context(ctx)

            # forward computation
            out = model(token_ids, segment_ids, valid_length.astype('float32'))

            # to binary: 0/1
            out = out.sigmoid().round().astype('int32')
            # to numpy (not mxnet)
            out = out.asnumpy()
            # get mapping type
            if label_map:
                out = [label_map[c] for c in list(out)]

            predictions.extend(out)

    # list to numpy array
    predictions = np.array(predictions)

    return predictions


# ############################################################################


def predict_unknown_proepi(model, data_predict, ctx, label_map=None, batch_size=32):
    bert_dataloader = mx.gluon.data.DataLoader(data_predict,
                                               batch_size=batch_size)

    predictions = list()

    with Timer("prediction"):
        for batch_id, (token_ids, valid_length, segment_ids, token_ids_epi,
                       valid_length_epi,
                       segment_ids_epi) in enumerate(tqdm(bert_dataloader)):
            global_step = batch_id
            # load data to GPU
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)
            segment_ids = segment_ids.as_in_context(ctx)
            token_ids_epi = token_ids_epi.as_in_context(ctx)
            valid_length_epi = valid_length_epi.as_in_context(ctx)
            segment_ids_epi = segment_ids_epi.as_in_context(ctx)

            # forward computation
            out = model(token_ids, segment_ids, valid_length.astype('float32'),
                        token_ids_epi, segment_ids_epi,
                        valid_length_epi.astype('float32'))

            # to binary: 0/1
            out = out.sigmoid().round().astype('int32')
            # to numpy (not mxnet)
            out = out.asnumpy()
            # get mapping type
            if label_map:
                out = [label_map[c] for c in list(out)]

            predictions.extend(out)

    # list to numpy array
    predictions = np.array(predictions)

    return predictions


# ############################################################################
# Dataset infos


def print_infos(vocabulary, data_train_raw, data_train):
    sample_id = 0

    # sentence a
    print(data_train_raw[sample_id][0])
    # sentence b
    print(data_train_raw[sample_id][1])
    # 1 means equivalent, 0 means not equivalent
    print(data_train_raw[sample_id][2])

    print('vocabulary used for tokenization = \n%s' % vocabulary)
    print('[PAD] token id = %s' % (vocabulary['[PAD]']))
    print('[CLS] token id = %s' % (vocabulary['[CLS]']))
    print('[SEP] token id = %s' % (vocabulary['[SEP]']))

    print('token ids = \n%s' % data_train[sample_id][0])
    print('valid length = \n%s' % data_train[sample_id][1])
    print('segment ids = \n%s' % data_train[sample_id][2])
    print('label = \n%s' % data_train[sample_id][3])


def print_infos_proepi(vocabulary, data_train_raw, data_train):
    sample_id = 0

    # sentence a
    print(data_train_raw[sample_id][0])
    # sentence b
    print(data_train_raw[sample_id][1])
    # 1 means equivalent, 0 means not equivalent
    print(data_train_raw[sample_id][2])

    print('vocabulary used for tokenization = \n%s' % vocabulary)
    print('[PAD] token id = %s' % (vocabulary['[PAD]']))
    print('[CLS] token id = %s' % (vocabulary['[CLS]']))
    print('[SEP] token id = %s' % (vocabulary['[SEP]']))

    print('token ids = \n%s' % data_train[sample_id][0])
    print('valid length = \n%s' % data_train[sample_id][1])
    print('segment ids = \n%s' % data_train[sample_id][2])
    print('epi token ids = \n%s' % data_train[sample_id][3])
    print('epi valid length = \n%s' % data_train[sample_id][4])
    print('epi segment ids = \n%s' % data_train[sample_id][5])
    print('label = \n%s' % data_train[sample_id][6])


# ############################################################################


def plot_train_stats(stats):
    if not stats:
        print("no stats to plot")
        return

    x = np.arange(len(stats))  # arange/linspace

    acc_dots, loss_dots = zip(*stats)
    # if isinstance(loss_dots, tuple):
    #     loss_dots, loss_dots2 = zip(*loss_dots)

    plt.subplot(2, 1, 1)
    plt.plot(x, acc_dots)  # Linie: '-', 'o-', '.-'
    plt.title('Training BERTClassifier')
    plt.ylabel('Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(x, loss_dots)
    plt.xlabel('Batches')
    plt.ylabel('Loss')

    plt.show()


# ############################################################################
# Metrics + Reporting


def compute_metrics_old(conf_mat, precision=3, dump=True):
    conf_mat = np.array(conf_mat)
    tn, fp, fn, tp = conf_mat.ravel()

    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp)
    rec  = tp / (tp + fn)
    f1 = 2 * (prec * rec) / (prec + rec)

    if dump:
        print("{:>10}: {:.{prec}f}".format("accuracy", acc, prec=precision))
        print("{:>10}: {:.{prec}f}".format("precision", prec, prec=precision))
        print("{:>10}: {:.{prec}f}".format("recall", rec, prec=precision))
        print("{:>10}: {:.{prec}f}".format("f1-score", f1, prec=precision))

    return prec, rec, f1, acc


def compute_metrics(labels, preds, precision=3, averaging="macro", dump=True):
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, labels=[0, 1], average=averaging)
    rec  = recall_score(labels, preds, labels=[0, 1], average=averaging)
    f1 = f1_score(labels, preds, labels=[0, 1], average=averaging)
    cm = confusion_matrix(labels, preds)

    if dump:
        print("CM:", cm.ravel(), "\n[tn, fp, fn, tp]")
        print("{:>10}: {:.{prec}f}".format("accuracy", acc, prec=precision))
        print("{:>10}: {:.{prec}f}".format("precision", prec, prec=precision))
        print("{:>10}: {:.{prec}f}".format("recall", rec, prec=precision))
        print("{:>10}: {:.{prec}f}".format("f1-score", f1, prec=precision))

    return prec, rec, f1, acc, cm


def heatconmat(y_test, y_pred):
    sns.set_context('talk')
    plt.figure(figsize=(9, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True,
                fmt='d',
                cbar=False,
                cmap='gist_earth_r',
                yticklabels=sorted(np.unique(y_test)))
    plt.show()


# see: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html#sklearn.metrics.plot_confusion_matrix
class ConfusionMatrixDisplay:
    """Confusion Matrix visualization.
    It is recommend to use :func:`~sklearn.metrics.plot_confusion_matrix` to
    create a :class:`ConfusionMatrixDisplay`. All parameters are stored as
    attributes.
    Read more in the :ref:`User Guide <visualizations>`.
    Parameters
    ----------
    confusion_matrix : ndarray of shape (n_classes, n_classes)
        Confusion matrix.
    display_labels : ndarray of shape (n_classes,)
        Display labels for plot.
    Attributes
    ----------
    im_ : matplotlib AxesImage
        Image representing the confusion matrix.
    text_ : ndarray of shape (n_classes, n_classes), dtype=matplotlib Text, \
            or None
        Array of matplotlib axes. `None` if `include_values` is false.
    ax_ : matplotlib Axes
        Axes with confusion matrix.
    figure_ : matplotlib Figure
        Figure containing the confusion matrix.
    """
    def __init__(self, confusion_matrix, display_labels, title=None):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels
        self.title = title

    def plot(self, include_values=True, cmap='viridis', show_colorbar=True,
             xticks_rotation='horizontal', values_format=None, ax=None):
        """Plot visualization.
        Parameters
        ----------
        include_values : bool, default=True
            Includes values in confusion matrix.
        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.
        xticks_rotation : {'vertical', 'horizontal'} or float, \
                         default='vertical'
            Rotation of xtick labels.
        values_format : str, default=None
            Format specification for values in confusion matrix. If `None`,
            the format specification is '.2f' for a normalized matrix, and
            'd' for a unnormalized matrix.
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        Returns
        -------
        display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
        """
        utils.check_matplotlib_support("ConfusionMatrixDisplay.plot")
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if self.title:
            fig.suptitle(self.title)

        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        self.text_ = None

        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)
            if values_format is None:
                values_format = '.2g'

            # print text with appropriate color depending on background
            thresh = (cm.max() - cm.min()) / 2.
            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min
                self.text_[i, j] = ax.text(j, i,
                                           format(cm[i, j], values_format),
                                           ha="center", va="center",
                                           color=color)

        if show_colorbar:
            fig.colorbar(self.im_, ax=ax)
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=self.display_labels,
               yticklabels=self.display_labels,
               ylabel="True label",
               xlabel="Predicted label")

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self


def plot_confusion_matrix(y_true, y_pred, labels=None,
                          sample_weight=None, normalize=None,
                          display_labels=None, include_values=True,
                          xticks_rotation='horizontal',
                          values_format=None, title=None,
                          include_colorbar=True,
                          cmap='viridis', ax=None):
    """Plot Confusion Matrix.
    Read more in the :ref:`User Guide <confusion_matrix>`.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Target values.
    y_pred : array-like of shape (n_samples,)
        Prediction values.
    labels : array-like of shape (n_classes,), default=None
        List of labels to index the matrix. This may be used to reorder or
        select a subset of labels. If `None` is given, those that appear at
        least once in `y_true` or `y_pred` are used in sorted order.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.
    display_labels : array-like of shape (n_classes,), default=None
        Target names used for plotting. By default, `labels` will be used if
        it is defined, otherwise the unique labels of `y_true` and `y_pred`
        will be used.
    include_values : bool, default=True
        Includes values in confusion matrix.
    xticks_rotation : {'vertical', 'horizontal'} or float, \
                        default='vertical'
        Rotation of xtick labels.
    values_format : str, default=None
        Format specification for values in confusion matrix. If `None`,
        the format specification is '.2f' for a normalized matrix, and
        'd' for a unnormalized matrix.
    cmap : str or matplotlib Colormap, default='viridis'
        Colormap recognized by matplotlib.
    ax : matplotlib Axes, default=None
        Axes object to plot on. If `None`, a new figure and axes is
        created.
    Returns
    -------
    display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
    """
    utils.check_matplotlib_support("plot_confusion_matrix")

    if normalize not in {'true', 'pred', 'all', None}:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight, labels=labels)

    if display_labels is None:
        if labels is None:
            raise ValueError("Missing labels!")
        else:
            display_labels = labels

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels,
                                  title=title)
    return disp.plot(include_values=include_values,
                     values_format=values_format, show_colorbar=include_colorbar,
                     cmap=cmap, ax=ax, xticks_rotation=xticks_rotation)


def report_training_results(y_test, y_pred, name=None, heatmap=True, metrics=True):
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print()
    if metrics:
        # compute_metrics(confusion_matrix(y_test, y_pred))
        compute_metrics(y_test, y_pred)
    if heatmap:
        heatconmat(y_test, y_pred)
    print()
    print('Accuracy: ', round(accuracy_score(y_test, y_pred), 3), '\n')  #

    print('Report{}:'.format("" if not name else " for [{}]".format(name)))
    print(classification_report(y_test, y_pred))

    f1_dic = {}
    f1_dic['macro'] = round(
        f1_score(y_pred=y_pred, y_true=y_test, average='macro'), 3)
    f1_dic['micro'] = round(
        f1_score(y_pred=y_pred, y_true=y_test, average='micro'), 3)
    return f1_dic


# ############################################################################



# ############################################################################



# ############################################################################
# Functions