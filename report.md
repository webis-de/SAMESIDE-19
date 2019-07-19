Argmining19 - Same Side Classification
======================================

Repo: https://github.com/Querela/argmining19-same-side-classification/  
&rarr; main dev (test) branch: "bert"

# training and evaluation runs

## general parameters (for comparisons)
- train/dev split: 0.7
- epochs: 5
- main dataset: `within` (two tags --> more generalization)

## setups

### Doc2Vec:
[notebook](https://github.com/Querela/argmining19-same-side-classification/blob/bert/same-side-classification-doc2vec.ipynb)

- train a doc2vec model and just replace the n-gram vectorization part of the baseline notebook
- tests on 'cross' dataset
- results: at most 59%, around baseline model provided by author
	- dbow: 53%
	- concat dbow-dmm: 57% ?
	- SGD (stochastic gradient descent) better than SVM / LogisticRegression: 59%

### XLNet:
[xlnet repo](https://github.com/zihangdai/xlnet)

- Standard XLNet implementation, only modified for dataset loading
- randomness may play a role, pre-trained dataset from other text type?
- results are not promising: between 47-53% (static, no major changes after first hundred steps)
- only a single run got over 82% (repeat with same presets did not!)
- tried different max_sequence_lengths (128 (default), 256, 512), different number of training steps (2k - 40k)

### BERT:
- pretrained model from GluonNLP, model='bert_12_768_12', dataset_name='book_corpus_wiki_en_uncased'
- with provided classifier
	- simply layers with Dropout and Dense (num_classes)  
	- (a) multi-label (default, with `softmax_crossentrophy`) - 2 classes  
	- (b) binary (with `sigmoid_binary_crossentrophy`) - single class
	- **binary classification resulted in 0.5-1% better accuracy**
- inputs are both arguments (sentence pair)
- arguments are truncated to `max_seq_len` (or half of it, since we provide sentence pairs)
- repeatable since random-state is always set
- setups:
	- (a) max_seq_len
		- (1) 128 &rarr; (ii) 256 &rarr; (iii) 512
		- **longer sequence lengths subsequently achieved better results**
	- (b) location of argument (argument is truncated to `max_seq_len` but from which direction)
		- (i) trim from end (standard)
		- (ii) trim from front
		- (iii) trim from both sides
		- (iv) trim from random side til correct length
	- (c) concatenation of two views on argument
		- similar to (b), but run _(i) start_ and _(ii) end of argument_ through BERT and concatenate vectors before classification
			- seems to improve because of longer argument (since two parts of argument are run through BERT)
	- (d) num_classes (is_same_side=True/False)
		- (i) 2 classes `softmax_crossentrophy`
		- (ii) 1 class `sigmoid_binary_crossentrophy`
	- best result:
		- [notebook](https://github.com/Querela/argmining19-same-side-classification/blob/bert/same-side-classification-bert-BCE-prolog%2Bepilog.ipynb)
		- (a)(iii) `max_seq_len` of 512
		- (c) with front & end of argument
		- (d)(ii) `sigmoid_binary_crossentrophy`
		- **currently best result with around 91%**
			- cross-model on cross devset:   92%
			- cross-model on within devset:  85%
			- within-model on within devset: 91%
			- within-model on cross devset:  94%

#### Runs:

| nr | setup | results | notes | ? | notebook |
|--- |---	|---	|---	|---	|---	|
| 1  | (a)(i), (b)(i), (d)(i)			| within 82% - cross 85%					| (out-of-the-box) 							|   	| [notebook](https://github.com/Querela/argmining19-same-side-classification/blob/bert/same-side-classification-bert.ipynb) |
| 2  | (a)(i), (b)(i), (d)(ii)			| within 86%								| binary classification						|   	| [notebook](https://github.com/Querela/argmining19-same-side-classification/blob/bert/same-side-classification-bert-BCE.ipynb) |
| 3  | (a)(iii), (b)(i), (d)(i)			| within 87%								| longer sequence of argument				|   	| [notebook](https://github.com/Querela/argmining19-same-side-classification/blob/bert/same-side-classification-bert-experiment.ipynb) |
| 4  | (a)(iii), (b)(ii), (d)(i)		| within 90%, within->cross 93%				| only last part of argument				|   	| [notebook](https://github.com/Querela/argmining19-same-side-classification/blob/bert/same-side-classification-bert-epilog.ipynb) |
| 5  | (a)(iii), (c), (d)(i)			| within 86%								| (3 epochs?) prolog + epilog				| | - |
| 6  | (a)(iii), (c), d(ii)				| within 91%, cross 92%, within->cross 94%, cross->within 85%  	| pro+epi, 512, binary	| best	| [notebook](https://github.com/Querela/argmining19-same-side-classification/blob/bert/same-side-classification-bert-BCE-prolog%2Bepilog.ipynb) |
| 7  | (a)(iii), (b)(iv), (c), d(ii)	| within 90% 								| pro+epi, trim random  					| 		| [notebook](https://github.com/Querela/argmining19-same-side-classification/tree/bert/same-side-classification-bert-BCE-rand2.ipynb) |
| 8  | (a)(iii), (b)(iii), (d)(i)		| within 88% 								| middle part 								|   	| [notebook](https://github.com/Querela/argmining19-same-side-classification/tree/bert/same-side-classification-bert-middle.ipynb) |


### Adversarial:
- idea: trying to suppress tag-related information ('abortion', 'gay marriage') from argument 'similarity' ('is_same_side')
- use pre-trained BERT model, just adjust standard loss computation
- two classifications trained at same time: (a) 'is_same_side', (b) 'tag'
- only promising strategy is: trying to direct the gradient (for training) for tag to around 50% (1 / number of tags)
	- if accuaracy of 'tag' at current training step is above 50% then reverse the gradient of 'tag'
	- then only sum both losses of 'is_same_side' and 'tag'
- result:
	- only works if gradient of tag is not too small (e. g. tag is already trained (0% or 100% accuracy))
	- if tag is around 50%, training works, but result are  
	83% for within model on within dev set,  
	83% within model on cross dev set

