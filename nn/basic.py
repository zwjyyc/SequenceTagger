'''
    This file contains implementations of various NN components, including
      -- Dropout
      -- Feedforward layer (with custom activations)
      -- RNN (with customizable activations)
      -- LSTM
      -- GRU
      -- CNN

    Each instance has a forward() method which takes x as input and return the
    post-activation representation y;

    Recurrent layers has two forward methods implemented:
        -- forward(x_t, h_tm1):  one step of forward given input x and previous
                                 hidden state h_tm1; return next hidden state

        -- forward_all(x, h_0):  apply successively steps given all inputs and
                                 initial hidden state, and return all hidden
                                 states h1, ..., h_n

    @author: Tao Lei
'''

import numpy as np
import math
import theano
import theano.tensor as T

from utils import say
from .initialization import default_srng, default_rng, USE_XAVIER_INIT
from .initialization import set_default_rng_seed, random_init, create_shared
from .initialization import ReLU, sigmoid, tanh, softmax, linear, get_activation_by_name

class Dropout(object):
    '''
        Dropout layer. forward(x) returns the dropout version of x

        Inputs
        ------

        dropout_prob : theano shared variable that stores the dropout probability
        srng         : theano random stream or None (default rng will be used)
        v2           : which dropout version to use

    '''
    def __init__(self, dropout_prob, srng=None, v2=False):
        self.dropout_prob = dropout_prob
        self.srng = srng if srng is not None else default_srng
        self.v2 = v2

    def forward(self, x):
        d = (1-self.dropout_prob) if not self.v2 else (1-self.dropout_prob)**0.5
        mask = self.srng.binomial(
                n = 1,
                p = 1-self.dropout_prob,
                size = x.shape,
                dtype = theano.config.floatX
            )
        return x*mask/d # why divide d ?


def apply_dropout(x, dropout_prob, v2=False):
    '''
        Apply dropout on x with the specified probability
    '''
    return Dropout(dropout_prob, v2=v2).forward(x)


class Layer(object):
    def __init__(self, n_in, n_out, activation,
                            clip_gradients=False,
                            has_bias=True,
			    scale = 1):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.clip_gradients = clip_gradients
        self.has_bias = has_bias
	self.scale = scale * np.sqrt(6.0/(n_in+n_out), dtype=theano.config.floatX)
        self.create_parameters()

        # not implemented yet
        if clip_gradients is True:
            raise Exception("gradient clip not implemented")

    def create_parameters(self):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation
        self.initialize_params(n_in, n_out, activation)

    def initialize_params(self, n_in, n_out, activation):
        scale = self.scale
	if USE_XAVIER_INIT:
            if activation == ReLU:
                #scale = np.sqrt(4.0/(n_in+n_out), dtype=theano.config.floatX)
                b_vals = np.ones(n_out, dtype=theano.config.floatX) * 0.0
            elif activation == softmax:
                #scale = np.float64(0.001 * scale).astype(theano.config.floatX)
                b_vals = np.zeros(n_out, dtype=theano.config.floatX)
            else:
                #scale = np.sqrt(2.0/(n_in+n_out), dtype=theano.config.floatX)
                b_vals = np.zeros(n_out, dtype=theano.config.floatX)
            #W_vals = random_init((n_in,n_out), rng_type="normal") * scale
	    W_vals = random_init((n_in, n_out)) * scale
	    #W_vals = scale * np.random.uniform(low=-1.0, high=1.0, size=(n_in,n_out))
        else:
	    scale = np.sqrt(6. / (n_in + n_out))
            W_vals = random_init((n_in,n_out)) * scale
	    #W_vals = np.random.uniform(low=-1.0, high=1.0, size=(n_in,n_out)) * scale
            if activation == softmax:
                W_vals *= (0.001 * self.scale)
            if activation == ReLU:
                b_vals = np.ones(n_out, dtype=theano.config.floatX) * 0.0
            else:
                b_vals = random_init((n_out,)) * 0.0
        self.W = create_shared(W_vals, name="W")
        if self.has_bias: self.b = create_shared(b_vals, name="b")

    def forward(self, x):
        if self.has_bias:
            return self.activation(
                    T.dot(x, self.W) + self.b
                )
        else:
            return self.activation(
                    T.dot(x, self.W)
                )

    @property
    def params(self):
        if self.has_bias:
            return [ self.W, self.b ]
        else:
            return [ self.W ]

    @params.setter
    def params(self, param_list):
        self.W.set_value(param_list[0].get_value())
        if self.has_bias: self.b.set_value(param_list[1].get_value())


class DynamicMemory(Layer):
    def __init__(self, n_in, n_out, activation,
            clip_gradients=False):
        super(DynamicMemory, self).__init__(
                n_in, n_out, activation,
                clip_gradients = clip_gradients
            )

    def create_parameters(self):
	n_in, n_out, activation = self.n_in, self.n_out, self.activation
	self.initialize_params(10 * n_in, n_out, activation)

    def forward(self, x, h, key):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation
        M1 = tanh(T.dot(x, self.W[:n_in]) + T.dot(key, self.W[n_in:2 * n_in]))
        M2 = tanh(T.dot(x, self.W[2*n_in:3*n_in]) + T.dot(h, self.W[3*n_in:4*n_in]))
        gate1 = sigmoid(T.dot(x ,self.W[4*n_in:5*n_in]) + T.dot(h, self.W[5*n_in:6*n_in]) + T.dot( key, self.W[6*n_in:7*n_in]))
        gate2 = sigmoid(T.dot(x ,self.W[7*n_in:8*n_in]) + T.dot(h, self.W[8*n_in:9*n_in]) + T.dot( key, self.W[9*n_in:10*n_in]))

        return activation( gate2 * M2 + gate1 * M1 + x)

    def forward_all(self, x, h0=None):
        if x.ndim > 1:
            h0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
        else:
            h0 = T.zeros((self.n_out,), dtype=theano.config.floatX)

        key = x[0]
        h, _ = theano.scan(
                    fn = self.forward,
                    sequences = x,
                    outputs_info = [ h0 ],
                    non_sequences = key
                )
        return h

class RecurrentLayer(Layer):
    def __init__(self, n_in, n_out, activation,
            clip_gradients=False):
        super(RecurrentLayer, self).__init__(
                n_in, n_out, activation,
                clip_gradients = clip_gradients
            )

    def create_parameters(self):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation

        # re-use the code in super-class Layer
        self.initialize_params(n_in + n_out, n_out, activation)

    def forward(self, x, h):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation
        return activation(
                T.dot(x, self.W[:n_in]) + T.dot(h, self.W[n_in:]) + self.b
            )

    def forward_all(self, x, h0=None):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out,), dtype=theano.config.floatX)
        h, _ = theano.scan(
                    fn = self.forward,
                    sequences = x,
                    outputs_info = [ h0 ]
                )
        return h

class MatrixLayer(object):
    def __init__(self, n_d, vocab, rank, oov = "<unk>", fix_init_embs = True, scale = 1):
        lst_words = []
        vocab_map = {}
        for word in vocab:
            if word not in vocab_map:
                vocab_map[word] = len(vocab_map)
                lst_words.append(word)

        self.lst_words = lst_words
        self.vocab_map = vocab_map
        emb_vals_v1 = random_init((len(self.vocab_map), rank, rank)) * scale
        emb_vals_v2 = random_init((len(self.vocab_map), rank, rank)) * scale
	emb_vals_v3 = random_init((len(self.vocab_map), rank, rank)) * scale
        emb_vals_v4 = random_init((len(self.vocab_map), rank, rank)) * scale	

        self.init_end = -1

        if oov is not None and oov is not False:
            assert oov in self.vocab_map, "oov {} not in vocab".format(oov)
            self.oov_tok = oov
            self.oov_id = self.vocab_map[oov]
        else:
            self.oov_tok = None
            self.oov_id = -1


	scale = np.sqrt(6.0 /  (n_d + rank))
        A1 = random_init((n_d, rank)) * scale
	A2 =  random_init((n_d, rank)) * scale
	A3 = random_init((n_d, rank)) * scale
        A4 =  random_init((n_d, rank)) * scale
	b = np.ones((n_d, n_d)) * 0.01
	b2 = np.ones((n_d, n_d)) * 0.01
        self.emb_values_v1 = emb_vals_v1
        self.emb_values_v2 = emb_vals_v2
	self.emb_values_v3 = emb_vals_v3
        self.emb_values_v4 = emb_vals_v4
        self.embeddings_v1 = create_shared(emb_vals_v1)
        self.embeddings_v2 = create_shared(emb_vals_v2)
	self.embeddings_v3 = create_shared(emb_vals_v3)
        self.embeddings_v4 = create_shared(emb_vals_v4)
	self.A1 = create_shared(A1)
	self.A2 = create_shared(A2)
	self.A3 = create_shared(A3)
        self.A4 = create_shared(A4)
	self.b = create_shared(b.astype(theano.config.floatX))
	self.b2 = create_shared(b2.astype(theano.config.floatX))
        if self.init_end > -1:
            self.embeddings_trainable = [self.embeddings_v1[self.init_end:], self.embeddings_v2[self.init_end:], self.embeddings_v3[self.init_end:], self.embeddings_v4[self.init_end:], self.A1, self.A2, self.A3, self.A4, self.b, self.b2 ]
        else:
            self.embeddings_trainable = [self.embeddings_v1, self.embeddings_v2, self.embeddings_v3, self.embeddings_v4, self.A1, self.A2, self.A3, self.A4, self.b, self.b2]

        self.n_V = len(self.vocab_map)
        self.n_d = n_d

    def map_to_word(self, id):
        n_V, lst_words = self.n_V, self.lst_words
        return lst_words[id] if id < n_V else '<err>'

    def map_to_words(self, ids):
        n_V, lst_words = self.n_V, self.lst_words
        return [ lst_words[i] if i < n_V else "<err>" for i in ids ]

    def map_to_ids(self, words, filter_oov=False):
        vocab_map = self.vocab_map
	oov_id = self.oov_id
        if filter_oov:
            not_oov = lambda x: x!=oov_id
            return np.array(
                    filter(not_oov, [ vocab_map.get(x, oov_id) for x in words ]),
                    dtype="int32"
                )
        else:
            return np.array(
                    [ vocab_map.get(x, oov_id) for x in words ],
                    dtype="int32"
                )

    def forward4(self, x):
        tmp = self.embeddings_v4[x]
        copy_factor = T.ones((x.shape[0], 1, 1))

        tmp = T.batched_dot(self.A4.dimshuffle('x', 0, 1) * copy_factor, tmp)
        tmp = T.batched_dot(tmp, self.A4.dimshuffle('x', 1, 0) * copy_factor)
        #return tmp
        #return ReLU(tmp)
        return T.tanh(tmp)

    def forward3(self, x):
        tmp = self.embeddings_v3[x]
        copy_factor = T.ones((x.shape[0], 1, 1))

        tmp = T.batched_dot(self.A3.dimshuffle('x', 0, 1) * copy_factor, tmp)
        tmp = T.batched_dot(tmp, self.A3.dimshuffle('x', 1, 0) * copy_factor)
        #return tmp
        #return ReLU(tmp)
        return T.tanh(tmp)

    def forward2(self, x):
	tmp = self.embeddings_v2[x]
	copy_factor = T.ones((x.shape[0], 1, 1))
	
	tmp = T.batched_dot(self.A2.dimshuffle('x', 0, 1) * copy_factor, tmp)
	tmp = T.batched_dot(tmp, self.A2.dimshuffle('x', 1, 0) * copy_factor)
        #return tmp
	#return ReLU(tmp)
	return T.tanh(tmp)

    def forward1(self, x):
	tmp = self.embeddings_v1[x]
	copy_factor = T.ones((x.shape[0], 1, 1))
	
        tmp = T.batched_dot(self.A1.dimshuffle('x', 0, 1) * copy_factor, tmp)
        tmp = T.batched_dot(tmp, self.A1.dimshuffle('x', 1, 0) * copy_factor)
        #return tmp
	return T.tanh(tmp)
	return ReLU(tmp)

    @property
    def params(self):
        return  self.embeddings_trainable

    @params.setter
    def params(self, param_list):
        for p, q in zip(self.embeddings_trainable, param_list):
            p.set_value(q.get_value())


class EmbeddingLayer(object):
    def __init__(self, n_d, vocab, oov="<unk>", embs=None, fix_init_embs=True, scale = 1, norm=-1):

        if embs is not None:
            lst_words = [ ]
            vocab_map = {}
            emb_vals = [ ]
	    emb_vals_ = []
            for word, vector in embs:
                #assert word not in vocab_map, "Duplicate words in initial embeddings"
                if word in vocab_map:
		    continue
		vocab_map[word] = len(vocab_map)
		if norm > 0:
		    vector = vector / np.linalg.norm(vector) * math.sqrt(norm)

                emb_vals.append(vector)
		emb_vals_.append(random_init((n_d,))*(0.001 if word != oov else 0.0))
                lst_words.append(word)

            self.init_end = len(emb_vals) if fix_init_embs else -1
            if n_d != len(emb_vals[0]):
                say("WARNING: n_d ({}) != init word vector size ({}). Use {} instead.\n".format(
                        n_d, len(emb_vals[0]), len(emb_vals[0])
                    ))
                n_d = len(emb_vals[0])

            say("{} pre-trained embeddings loaded.\n".format(len(emb_vals)))

            for word in vocab:
                if word not in vocab_map:
		    scale = np.sqrt(6.0 / (len(vocab_map) + n_d))
                    vocab_map[word] = len(vocab_map)
                    emb_vals.append(random_init((n_d,))*( scale if word != oov else 0.0))
		    emb_vals_.append(random_init((n_d,))*(scale if word != oov else 0.0))
                    lst_words.append(word)

	    emb_vals_ = np.vstack(emb_vals_).astype(theano.config.floatX)
            emb_vals = np.vstack(emb_vals).astype(theano.config.floatX)
            self.vocab_map = vocab_map
            self.lst_words = lst_words
        else:
            lst_words = [ ]
            vocab_map = {}
            for word in vocab:
                if word not in vocab_map:
                    vocab_map[word] = len(vocab_map)
                    lst_words.append(word)

            self.lst_words = lst_words
            self.vocab_map = vocab_map
            emb_vals = random_init((len(self.vocab_map), n_d)) * scale
	    emb_vals_ = random_init((len(self.vocab_map), n_d)) * scale
            self.init_end = -1

        if oov is not None and oov is not False:
            assert oov in self.vocab_map, "oov {} not in vocab".format(oov)
            self.oov_tok = oov
            self.oov_id = self.vocab_map[oov]
        else:
            self.oov_tok = None
            self.oov_id = -1
	
	self.emb_values = emb_vals
	self.emb_values_ = emb_vals_
        self.embeddings = create_shared(emb_vals)
	self.bembeddings = create_shared(emb_vals_)

        if self.init_end > -1:
            self.embeddings_trainable = [self.embeddings[self.init_end:], self.bembeddings[self.init_end:] ]
        else:
            self.embeddings_trainable = [self.embeddings, self.bembeddings ]

        self.n_V = len(self.vocab_map)
        self.n_d = n_d

    def word_matching(self, vocab):
	vocab_map = self.vocab_map
	numV = len(vocab)
	num = 0
	for word in vocab:
	    if word in vocab_map:
		num += 1
	print 'matched ' + str(num) + '(' + str(num * 1.0 / numV) + '%) words'

    def map_to_word(self, id):
	n_V, lst_words = self.n_V, self.lst_words
	return lst_words[id] if id < n_V else '<err>' 

    def map_to_words(self, ids):
        n_V, lst_words = self.n_V, self.lst_words
        return [ lst_words[i] if i < n_V else "<err>" for i in ids ]
    
    def map_to_ids(self, words, filter_oov=False):
        vocab_map = self.vocab_map
        oov_id = self.oov_id
        if filter_oov:
            not_oov = lambda x: x!=oov_id
            return np.array(
                    filter(not_oov, [ vocab_map.get(x, oov_id) for x in words ]),
                    dtype="int32"
                )
        else:
            return np.array(
                    [ vocab_map.get(x, oov_id) for x in words ],
                    dtype="int32"
                )
    def forward2(self, x):
	return self.bembeddings[x]

    def forward(self, x, isNode = True):
	if isNode:
            return self.embeddings[x]
	else:
	    return self.emb_values[x]

    @property
    def params(self):
        return  self.embeddings_trainable 

    @params.setter
    def params(self, param_list):
        for p, q in zip(self.embeddings_trainable, param_list):
            p.set_value(q.get_value())
	#self.embeddings.set_value(param_list[0].get_value())
	#print param_list[0].get_value()
	#print type(param_list[0].get_value())



class TreeDynamicMemory(Layer):
    def __init__(self, n_in, n_out, activation = tanh,
            clip_gradients=False):
        super(TreeDynamicMemory, self).__init__(
                n_in, n_out, activation,
                clip_gradients = clip_gradients
            )

    def create_parameters(self):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation
        self.initialize_params(10 * n_in, n_out, activation)

    def forward(self, x, id, pid, hc, key):
	h = hc[pid]
	
	n_in, n_out, activation = self.n_in, self.n_out, self.activation
        M1 = tanh(T.dot(x, self.W[:n_in]) + T.dot(key, self.W[n_in:2 * n_in]))
        M2 = tanh(T.dot(x, self.W[2*n_in:3*n_in]) + T.dot(h, self.W[3*n_in:4*n_in]))
        gate1 = sigmoid(T.dot(x ,self.W[4*n_in:5*n_in]) + T.dot(h, self.W[5*n_in:6*n_in]) + T.dot( key, self.W[6*n_in:7*n_in]))
        gate2 = sigmoid(T.dot(x ,self.W[7*n_in:8*n_in]) + T.dot(h, self.W[8*n_in:9*n_in]) + T.dot( key, self.W[9*n_in:10*n_in]))

        tmp = activation( gate2 * M2 + gate1 * M1 + x)
	previous = T.concatenate( [hc[:id, :], tmp, hc[(id + 1):, :]])
	return previous

    def forward_all(self, x,  ids, pids, h0=None):
        if x.ndim > 1:
            h0 = T.zeros((x.shape[1]  * x.shape[0], self.n_out), dtype=theano.config.floatX)
        else:
            h0 = T.zeros((self.n_out,), dtype=theano.config.floatX)
	
        key = x[0]
        h, _ = theano.scan(
                    fn = self.forward,
                    sequences = [x, ids, pids ],
                    outputs_info = [ h0 ],
                    non_sequences = key
                )
        return h[-1]


class TreeLSTM(Layer):
    def __init__(self, n_in, n_out, activation=tanh,
            clip_gradients=False, direction = "si"):

        self.n_in = n_in
        self.n_out = n_out
        n_out_t = self.n_out_t = n_out
        self.activation = activation
        self.clip_gradients = clip_gradients
        self.direction =  direction
        
	self.in_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.forget_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.out_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.input_layer = RecurrentLayer(n_in, n_out, activation, clip_gradients)

        self.internal_layers = [ self.input_layer, self.in_gate,
                                 self.forget_gate , self.out_gate ]

    def forward(self, x, id, pid, mask, hc):
        n_in, n_out, activation = self.n_in, self.n_out_t, self.activation

        if hc.ndim > 1:
            c_tm1 = hc[pid, :n_out]
            h_tm1 = hc[pid, n_out:]
        else:
            c_tm1 = hc[:n_out]
            h_tm1 = hc[n_out:]

        in_t = self.in_gate.forward(x,h_tm1)
        forget_t = self.forget_gate.forward(x,h_tm1)
        out_t = self.out_gate.forward(x, h_tm1)

        c_t = forget_t * c_tm1 + in_t * self.input_layer.forward(x,h_tm1)
        c_t = c_t * mask.dimshuffle(0, 'x')
        c_t = T.cast(c_t, 'float32')
        h_t = out_t * T.tanh(c_t)
        h_t = h_t * mask.dimshuffle(0, 'x')
        h_t = T.cast(h_t, 'float32')

        if hc.ndim > 1:
            tmp = T.concatenate( [ c_t, h_t ], axis=1)
	    hc = T.concatenate( [hc[:id, :], tmp, hc[(id + 1):, :]])
        else:
            tmp = T.concatenate([ c_t, h_t ])
	    hc = T.concatenate( [hc[:id, :], tmp, hc[(id + 1):, :]])
	return hc
	
    def forward_all(self, x, ids, pids, masks = None, h0=None, return_c=False):
        n_out_t = self.n_out_t
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1] * x.shape[0], n_out_t * 2), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((n_out_t * 2 * x.shape[0],), dtype=theano.config.floatX)
        if masks is None:
            masks = T.ones((x.shape[0], x.shape[1]), dtype = theano.config.floatX)

        h, _ = theano.scan(
                    fn = self.forward,
                    sequences = [x, ids, pids, masks],
                    outputs_info = [ h0 ]
                )
        
	if return_c:
            return h[-1]
        else:
            return h[-1][:,n_out_t:]

    @property
    def params(self):
        return [ x for layer in self.internal_layers for x in layer.params ]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.internal_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end

class SelfMatching(Layer):
    def __init__(self, n_in, n_out, activation=tanh,
            clip_gradients=False, direction = "si"):

        self.n_in = n_in
        self.n_out = n_out
        n_out_t = self.n_out_t = n_out
        if direction == 'bi':
            n_out_t = self.n_out_t = n_out / 2
        self.activation = activation
        self.clip_gradients = clip_gradients
        self.direction =  direction
        self.in_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.forget_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.out_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.input_layer = RecurrentLayer(n_in, n_out, activation, clip_gradients)

        if direction == "bi":
            self.in_gate = RecurrentLayer(n_in, n_out_t , sigmoid, clip_gradients)
            self.forget_gate = RecurrentLayer(n_in, n_out_t, sigmoid, clip_gradients)
            self.out_gate = RecurrentLayer(n_in, n_out_t, sigmoid, clip_gradients)
            self.input_layer = RecurrentLayer(n_in, n_out_t, activation, clip_gradients)
            self.in_gate_b = RecurrentLayer(n_in, n_out_t, sigmoid, clip_gradients)
            self.forget_gate_b = RecurrentLayer(n_in, n_out_t, sigmoid, clip_gradients)
            self.out_gate_b = RecurrentLayer(n_in, n_out_t, sigmoid, clip_gradients)
            self.input_layer_b = RecurrentLayer(n_in, n_out_t, activation, clip_gradients)
        
	self.internal_layers = [ self.input_layer, self.in_gate,
                                 self.forget_gate , self.out_gate ]
        if direction == "bi":
            self.internal_layers = [self.input_layer_b, self.input_layer, self.in_gate_b, self.in_gate,
                                    self.forget_gate_b, self.forget_gate, self.out_gate_b, self.out_gate]
	
	scale = np.sqrt(3.0 / n_in, dtype=theano.config.floatX)
        self.W_g = create_shared(random_init((n_in, n_in)) * scale , name = 'W_g')
	self.W_g_b = create_shared(random_init((n_in, n_in)) * scale , name = 'W_g_b')
	self.W_Pv = create_shared(random_init((n_in / 2, n_in / 2)) * scale, name = 'W_Pv')
	self.W_Pv_b = create_shared(random_init((n_in / 2, n_in / 2)) * scale, name = 'W_Pv_b')
	self.W_Pv2 = create_shared(random_init((n_in / 2, n_in / 2)) * scale, name = 'W_Pv2')
	self.W_Pv2_b = create_shared(random_init((n_in / 2, n_in / 2)) * scale, name = 'W_Pv2_b')
	self.v = create_shared(random_init((n_in / 2,)) * scale, name = 'v')
	self.v_b = create_shared(random_init((n_in / 2,)) * scale, name = 'v_b')
	self.lst_params = [self.W_g, self.W_Pv, self.W_Pv2, self.v, self.W_g_b, self.W_Pv_b, self.W_Pv2_b, self.v_b]  

    def forward(self, x, mask, hc, xs):
        n_in, n_out, activation = self.n_in, self.n_out_t, self.activation

        c_tm1 = hc[:, :n_out]
        h_tm1 = hc[:, n_out:]
	
	st = T.dot(T.tanh(T.dot(xs, self.W_Pv) + T.dot(x, self.W_Pv2)), self.v)
	st = T.exp(st) * mask.dimshuffle(0, 'x')
	sst = T.sum(st)
	st = st / sst	
	
	tmp = st.dimshuffle(0)
	tmp = tmp.dimshuffle(0, 'x') * xs.dimshuffle(0, 2)
	c = T.sum(tmp, axis = 0)
	
	x_ = T.concatenate([x, c.dimshuffle('x', 0)], axis = 1)
	gt = T.nnet.sigmoid(T.dot(x_, self.W_g))
	x_ = gt * x_	

        in_t = self.in_gate.forward(x_, h_tm1)
        forget_t = self.forget_gate.forward(x_, h_tm1)
        out_t = self.out_gate.forward(x_, h_tm1)

        c_t = forget_t * c_tm1 + in_t * self.input_layer.forward(x_, h_tm1)
        c_t = c_t * mask.dimshuffle(0, 'x')
        c_t = T.cast(c_t, 'float32')
        h_t = out_t * T.tanh(c_t)
        h_t = h_t * mask.dimshuffle(0, 'x')
        h_t = T.cast(h_t, 'float32')

        return T.concatenate([ c_t, h_t ], axis=1)

    def backward(self, x, mask, hc, xs):
        n_in, n_out, activation = self.n_in, self.n_out_t, self.activation

        c_tm1 = hc[:, :n_out]
        h_tm1 = hc[:, n_out:]
	
	st = T.dot(T.tanh(T.dot(xs, self.W_Pv_b) + T.dot(x, self.W_Pv2_b)), self.v_b)
        st = T.exp(st) * mask.dimshuffle(0, 'x')
        sst = T.sum(st)
        st = st / sst

	tmp = st.dimshuffle(0)
        tmp = tmp.dimshuffle(0, 'x') * xs.dimshuffle(0, 2)
	c = T.sum(tmp, axis = 0)	

        x_ = T.concatenate([x, c.dimshuffle('x', 0)], axis = 1)
        gt = T.nnet.sigmoid(T.dot(x_, self.W_g_b))
        x_ = gt * x_

        in_t = self.in_gate_b.forward(x_, h_tm1)
        forget_t = self.forget_gate_b.forward(x_, h_tm1)
        out_t = self.out_gate_b.forward(x_, h_tm1)

        c_t = forget_t * c_tm1 + in_t * self.input_layer_b.forward(x_, h_tm1)
        c_t = c_t * mask.dimshuffle(0, 'x')
        c_t = T.cast(c_t, 'float32')
        h_t = out_t * T.tanh(c_t)
        h_t = h_t * mask.dimshuffle(0, 'x')
        h_t = T.cast(h_t, 'float32')

        return T.concatenate([ c_t, h_t ], axis=1)

    def forward_all(self, x, masks = None, h0 = None, return_c = False):
        n_out_t = self.n_out_t
        if h0 is None:
            h0 = T.zeros((x.shape[1], n_out_t * 2), dtype=theano.config.floatX)

        if masks is None:
            masks = T.ones((x.shape[0], x.shape[1]), dtype = theano.config.floatX)

        h, _ = theano.scan(
                    fn = self.forward,
                    sequences = [x, masks],
		    non_sequences = x,
                    outputs_info = [ h0 ]
                )

        if self.direction == "bi":
            h1, _ = theano.scan(
                        fn = self.backward,
                        sequences = [x[::-1, ::, ::], masks[::-1, ::]],
			non_sequences = x[::-1, ::, ::],
                        outputs_info = [h0]
                    )
            h = T.concatenate((h, h1[::-1, ::, ::][:, :, n_out_t:]), axis = 2)

	return h[:,:,n_out_t:]

    @property
    def params(self):
	tmp = [ x for layer in self.internal_layers for x in layer.params ]
	tmp.extend([x for x in self.lst_params])
	return tmp

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.internal_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
	    start = end
	
	assert len(param_list[start:]) == len(self.lst_params)
        for p, q in zip(self.lst_params, param_list[start:]):
            p.set_value(q.get_value())

class LSTM(Layer):
    def __init__(self, n_in, n_out, activation=tanh,
            clip_gradients=False, direction = "si"):

        self.n_in = n_in
        self.n_out = n_out
        n_out_t = self.n_out_t = n_out
	if direction == 'bi':
	    n_out_t = self.n_out_t = n_out / 2
	self.activation = activation
        self.clip_gradients = clip_gradients
	self.direction =  direction
        self.in_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.forget_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.out_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.input_layer = RecurrentLayer(n_in, n_out, activation, clip_gradients)
	
	if direction == "bi":
	    self.in_gate = RecurrentLayer(n_in, n_out_t , sigmoid, clip_gradients)
            self.forget_gate = RecurrentLayer(n_in, n_out_t, sigmoid, clip_gradients)
            self.out_gate = RecurrentLayer(n_in, n_out_t, sigmoid, clip_gradients)
            self.input_layer = RecurrentLayer(n_in, n_out_t, activation, clip_gradients)
	    self.in_gate_b = RecurrentLayer(n_in, n_out_t, sigmoid, clip_gradients)
	    self.forget_gate_b = RecurrentLayer(n_in, n_out_t, sigmoid, clip_gradients)
	    self.out_gate_b = RecurrentLayer(n_in, n_out_t, sigmoid, clip_gradients)
	    self.input_layer_b = RecurrentLayer(n_in, n_out_t, activation, clip_gradients)


	
        self.internal_layers = [ self.input_layer, self.in_gate,
                                 self.forget_gate , self.out_gate ]
	
	if direction == "bi":
	    self.internal_layers = [self.input_layer_b, self.input_layer, self.in_gate_b, self.in_gate, 
				    self.forget_gate_b, self.forget_gate, self.out_gate_b, self.out_gate]

    def forward(self, x, mask, hc):
        n_in, n_out, activation = self.n_in, self.n_out_t, self.activation

        if hc.ndim > 1:
            c_tm1 = hc[:, :n_out]
            h_tm1 = hc[:, n_out:]
        else:
            c_tm1 = hc[:n_out]
            h_tm1 = hc[n_out:]

        in_t = self.in_gate.forward(x,h_tm1)
        forget_t = self.forget_gate.forward(x,h_tm1)
        out_t = self.out_gate.forward(x, h_tm1)

        c_t = forget_t * c_tm1 + in_t * self.input_layer.forward(x,h_tm1)
	c_t = c_t * mask.dimshuffle(0, 'x')
	c_t = T.cast(c_t, 'float32')
        h_t = out_t * T.tanh(c_t)
	h_t = h_t * mask.dimshuffle(0, 'x')
	h_t = T.cast(h_t, 'float32')

        if hc.ndim > 1:
            return T.concatenate([ c_t, h_t ], axis=1)
        else:
            return T.concatenate([ c_t, h_t ])

    def backward(self, x, mask, hc):
        n_in, n_out, activation = self.n_in, self.n_out_t, self.activation

        if hc.ndim > 1:
            c_tm1 = hc[:, :n_out]
            h_tm1 = hc[:, n_out:]
        else:
            c_tm1 = hc[:n_out]
            h_tm1 = hc[n_out:]

        in_t = self.in_gate_b.forward(x,h_tm1)
        forget_t = self.forget_gate_b.forward(x,h_tm1)
        out_t = self.out_gate_b.forward(x, h_tm1)

        c_t = forget_t * c_tm1 + in_t * self.input_layer_b.forward(x,h_tm1)
        c_t = c_t * mask.dimshuffle(0, 'x')
	c_t = T.cast(c_t, 'float32')
        h_t = out_t * T.tanh(c_t)
        h_t = h_t * mask.dimshuffle(0, 'x')
	h_t = T.cast(h_t, 'float32')

        if hc.ndim > 1:
            return T.concatenate([ c_t, h_t ], axis=1)
        else:
            return T.concatenate([ c_t, h_t ])

    def forward_all(self, x, masks = None, h0=None, return_c=False):
	n_out_t = self.n_out_t
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], n_out_t * 2), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((n_out_t * 2,), dtype=theano.config.floatX)
        if masks is None:
	    masks = T.ones((x.shape[0], x.shape[1]), dtype = theano.config.floatX)
	
	h, _ = theano.scan(
                    fn = self.forward,
                    sequences = [x, masks],
                    outputs_info = [ h0 ]
                )
	if self.direction == "bi":
	    if x.ndim > 1:
	    	h1, _ = theano.scan(
			fn = self.backward,
			sequences = [x[::-1, ::, ::], masks[::-1, ::]],
			outputs_info = [h0]	
		    )
	    	h = T.concatenate((h, h1[::-1, ::, ::][:, :, n_out_t:]), axis = 2)
	    else:
                h1, _ = theano.scan(
                        fn = self.backward,
                        sequences = [x[::-1, ::], masks[::-1]],
                        outputs_info = [h0]
                    )
                h = T.concatenate((h, h1[::-1, ::][:, n_out_t:]), axis = 1)

        if return_c:
            return h
        elif x.ndim > 1:
            return h[:,:,n_out_t:]
        else:
            return h[:,n_out_t:]

    @property
    def params(self):
        return [ x for layer in self.internal_layers for x in layer.params ]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.internal_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end

class GRU(Layer):
    '''
        GRU implementation
    '''
    def __init__(self, n_in, n_out, activation=tanh,
            clip_gradients=False):

        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.clip_gradients = clip_gradients

        self.reset_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.update_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        self.input_layer = RecurrentLayer(n_in, n_out, activation, clip_gradients)

        self.internal_layers = [ self.reset_gate, self.update_gate, self.input_layer ]

    def forward(self, x, h):
        n_in, n_out, activation = self.n_in, self.n_out, self.activation

        reset_t = self.reset_gate.forward(x, h)
        update_t = self.update_gate.forward(x, h)
        h_reset = reset_t * h

        h_new = self.input_layer.forward(x, h_reset)
        h_out = update_t*h_new + (1.0-update_t)*h
        return h_out

    def forward_all(self, x, h0=None, return_c=True):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out,), dtype=theano.config.floatX)
        h, _ = theano.scan(
                    fn = self.forward,
                    sequences = x,
                    outputs_info = [ h0 ]
                )
        return h

    @property
    def params(self):
        return [ x for layer in self.internal_layers for x in layer.params ]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.internal_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end


class CNN(Layer):
    '''
        CNN implementation. Return feature maps over time. No pooling is used.

        Inputs
        ------

            order       : feature filter width
    '''
    def __init__(self, n_in, n_out, activation=tanh,
            order=1, clip_gradients=False):

        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.order = order
        self.clip_gradients = clip_gradients

        internal_layers = self.internal_layers = [ ]
        for i in range(order):
            input_layer = Layer(n_in, n_out, linear, has_bias=False, \
                    clip_gradients=clip_gradients)
            internal_layers.append(input_layer)

        self.bias = create_shared(random_init((n_out,)), name="bias")

    def forward(self, x, hc):
        order, n_in, n_out, activation = self.order, self.n_in, self.n_out, self.activation
        layers = self.internal_layers
        if hc.ndim > 1:
            h_tm1 = hc[:, n_out*order:]
        else:
            h_tm1 = hc[n_out*order:]

        lst = [ ]
        for i in range(order):
            if hc.ndim > 1:
                c_i_tm1 = hc[:, n_out*i:n_out*i+n_out]
            else:
                c_i_tm1 = hc[n_out*i:n_out*i+n_out]
            if i == 0:
                c_i_t = layers[i].forward(x)
            else:
                c_i_t = c_im1_tm1 + layers[i].forward(x)
            lst.append(c_i_t)
            c_im1_tm1 = c_i_tm1

        h_t = activation(c_i_t + self.bias)
        lst.append(h_t)

        if hc.ndim > 1:
            return T.concatenate(lst, axis=1)
        else:
            return T.concatenate(lst)

    def forward_all(self, x, h0=None, return_c=False):
        '''
            Apply filters to every local chunk of the sequence x. Return the feature
            maps as a matrix, or a tensor instead if x is a batch of sequences
        '''
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out*(self.order+1)), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out*(self.order+1),), dtype=theano.config.floatX)
        h, _ = theano.scan(
                    fn = self.forward,
                    sequences = x,
                    outputs_info = [ h0 ]
                )
        if return_c:
            return h
        elif x.ndim > 1:
            return h[:,:,self.n_out*self.order:]
        else:
            return h[:,self.n_out*self.order:]

    @property
    def params(self):
        return [ x for layer in self.internal_layers for x in layer.params ] + [ self.bias ]

    @params.setter
    def params(self, param_list):
        start = 0
        for layer in self.internal_layers:
            end = start + len(layer.params)
            layer.params = param_list[start:end]
            start = end
        self.bias.set_value(param_list[-1].get_value())


