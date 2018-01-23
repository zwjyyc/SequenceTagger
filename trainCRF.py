import os
import argparse
import sys
import numpy as np

from processor import processor
from graph_conv import GraphConvCRF
from graph_conv_2 import GraphConvCRFV2
from graph_conv_4 import GraphConvCRFV4
from self_matching import SelfMatchingNN
from graph_kernel_nn import GraphKernelCRF
from utils import words_load, chars_load, rels_load, say, load_embedding_iterator, create_input, evaluate
from nn import EmbeddingLayer, MatrixLayer

np.set_printoptions(precision=3)

argparser = argparse.ArgumentParser(sys.argv[0])
argparser.add_argument("--train",
    type = str,
    default = "",
    help = "Train"
)
argparser.add_argument("--test",
    type = str,
    default = "",
    help = "Test set location"
)
argparser.add_argument("--char_dim",
    type = int,
    default = 25,
    help = "Char embedding dimension"
)
argparser.add_argument("--char_lstm_dim",
    type = int,
    default = 50,
    help = "Char lstm dimension"
)
argparser.add_argument("--char_bidirect",
    type = int,
    default = 1,
    help = "Use a bidirectional LSTM for chars"
)
argparser.add_argument("--word_dim",
    type = int,
    default = 100,
    help = "Token embedding dimension"
)
argparser.add_argument("--word_lstm_dim",
    type = int,
    default = 100,
    help = "Token LSTM hidden layer size"
)
argparser.add_argument("--word_bidirect",
    type = int,
    default = 1,
    help = "Use a bidirectional LSTM for words"
)
argparser.add_argument("--pre_emb",
    type = str,
    default = '',
    help = 'Location of pre-trained embeddings'
)
argparser.add_argument("--dropout",
    type = float,
    default = 0.5,
    help = 'Droupout on the input (0 = no dropout)'
)
argparser.add_argument("--load",
    type = str,
    default = "",
    help = "load model from this file"
)
argparser.add_argument("--log",
    type = str,
    default = "result.log",
    help = "log file"
)
argparser.add_argument("--learning",
    type = str,
    default = "sgd",
    help = "learning method (sgd, adagrad, adam, ...)"
)
argparser.add_argument("--save",
    type = str,
    default = "",
    help = "save model to this file"
)
argparser.add_argument("--pooling",
    type = int,
    default = 0,
    help = "pooling strategy"
)
argparser.add_argument("--l2_reg",
    type = float,
    default = 0,
    help = "regularization"
)
argparser.add_argument("--learning_rate",
    type = float,
    default = 0.005,
    help = "regularization"
)
argparser.add_argument("--classes",
    type = int,
    default = 3,
    help = 'class number'
)
argparser.add_argument("--epoch",
    type = int,
    default = 100,
    help = 'epoch'
)
argparser.add_argument("--start_epoch",
    type = int,
    default = 80,
    help = 'epoch'
)
argparser.add_argument("--conv",
    type = int,
    default = 1,
    help = ''
)
argparser.add_argument("--model",
    type = int,
    default = 1,
    help = ''
)
argparser.add_argument("--clayer",
    type = int,
    default = 1,
    help = ''
)
argparser.add_argument("--rank",
    type = int,
    default = 10,
    help = ''
)
argparser.add_argument("--head",
    type = int,
    default = 4,
    help = ''
)

args = argparser.parse_args()

print args

assert os.path.isfile(args.train)
assert os.path.isfile(args.test)
assert args.save and args.pre_emb

train_processor = processor(args.train)
test_processor = processor(args.test)

print "Load data..."
train_cors = train_processor.loadSrc()
test_cors = test_processor.loadSrc()

print 'Constructing word and character list...'
word_lis = words_load(train_cors, test_cors)
char_lis = chars_load(word_lis)
char_lis.append('<unk>')
rel_lis = rels_load(train_cors, test_cors)
rel_lis.append('<unk>')
print 'Find ' + str(len(word_lis)) + ' unique words!'
print 'Find ' + str(len(char_lis)) + ' unique chars!'
print 'Find ' + str(len(rel_lis)) + ' unique dep relations!'

word_embedding_layer = EmbeddingLayer(
	n_d = args.word_dim,
	vocab = ['<unk>'],
	embs = load_embedding_iterator(args.pre_emb),
	fix_init_embs = False
)

char_embedding_layer = EmbeddingLayer(
        n_d = args.char_dim,
	vocab = char_lis,
	fix_init_embs = False
)

rel_embedding_layers = []
rel_matrix_layers = []
for i in range(args.clayer):
    rel_embedding_layers.append(EmbeddingLayer(
	n_d = args.word_dim,
	vocab = rel_lis,
	fix_init_embs = False
	))

if args.model == 4:
    for i in range(args.clayer):
	rel_matrix_layers.append(MatrixLayer(
	n_d = args.word_dim,
	vocab = rel_lis,
	rank = args.rank,
	fix_init_embs = False
	))

word_embedding_layer.word_matching(word_lis)

print 'Construting input...'

train_input = []
dev_input = []
test_input = []

train_set = train_cors
test_set = test_cors

ratio = 0.80
train_size = len(train_set)
print 'Training set : ' + str(train_size)
test_size = len(test_set)

for i, index in enumerate(np.random.permutation(train_size)):
    ins = create_input(train_set[index], word_embedding_layer, char_embedding_layer, rel_embedding_layers[0])
    if i * 1.0 / train_size >= ratio:
	dev_input.append(ins)
    else:
	train_input.append(ins)

print len(train_input)
print len(dev_input)

for index in range(test_size):
    ins = create_input(test_set[index], word_embedding_layer, char_embedding_layer, rel_embedding_layers[0])
    test_input.append(ins)
print len(test_input)

if args.model == 1:
    model = GraphConvCRF(args, word_embedding_layer, char_embedding_layer, rel_embedding_layers)
elif args.model == 2:
    model = GraphConvCRFV2(args, word_embedding_layer, char_embedding_layer, rel_embedding_layers)
elif args.model == 3:
    model = SelfMatchingNN(args, word_embedding_layer, char_embedding_layer, rel_embedding_layers)
elif args.model == 4:
    model = GraphConvCRFV4(args, word_embedding_layer, char_embedding_layer, rel_embedding_layers, rel_matrix_layers)
elif args.model == 5:
    model = GraphKernelCRF(args, word_embedding_layer, char_embedding_layer, rel_embedding_layers, rel_matrix_layers)

print 'Building model....'
f_train, f_eval = model.ready()

if args.load:
    print 'Loading model......'
    model.load(args.load) 
    model.dropout.set_value(0.0)
    test_preds = [ f_eval(*input[:-1]) for input in test_input]
    test_score, pre, recall = evaluate(test_preds, test_input)
    print "Score on test: %.5f" % test_score    
    sys.exit(0) 

n_epochs = args.epoch
freq_eval = 1000
best_dev = -np.inf
best_test = -np.inf
count = 0

log_file = args.log
bufsize = 1
f_log = open(log_file, 'w', bufsize)

for epoch in xrange(n_epochs):
    epoch_costs = []
    epoch_gnorm = []
    print "Starting epoch %i..." % epoch
    for i, index in enumerate(np.random.permutation(len(train_input))):
	count += 1
	input = train_input[index]
	model.s_num = len(input[0])
	all_cost, new_cost = f_train(*input)
	epoch_costs.append(new_cost)	
	if i % 100 == 0 and i > 0 == 0:
            print "%i, cost average: %f" % (i, np.mean(epoch_costs[-50:]))
	if count % freq_eval == 0:
	    model.dropout.set_value(0.0)
	    dev_preds = [ f_eval(*input[:-1]) for input in dev_input]
	    test_preds = [ f_eval(*input[:-1]) for input in test_input]
	    dev_score, pre, recall = evaluate(dev_preds, dev_input)
	    f_log.write('dev\t%.5f\t%.5f\t%.5f\n' % (pre, recall, dev_score))
	    test_score, pre, reall = evaluate(test_preds, test_input)
	    f_log.write('test\t%.5f\t%.5f\t%.5f\n' % (pre, recall, test_score))
	    
	    print "Score on dev: %.5f" % dev_score
            print "Best score on dev: %.5f" % best_dev
	    if dev_score > best_dev and epoch >= args.start_epoch:
                best_dev = dev_score
                print "New best score on dev."
                print "Saving model to disk..."
                model.save(args.save)
                print "Score on test: %.5f" % test_score
	    model.dropout.set_value(args.dropout)
    print "Epoch %i done. Average cost: %f" % (epoch, np.mean(epoch_costs))

f_log.close()
