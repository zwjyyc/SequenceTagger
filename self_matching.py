import numpy as np
import gzip, os, pickle
import theano
import theano.tensor as T

from nn import get_activation_by_name, create_optimization_updates, softmax, ReLU, tanh, linear
from nn import Layer, LSTM, apply_dropout, CRFForward, GraphCNN, SelfMatching
from utils import say, shared
from optimization import Optimization

class SelfMatchingNN:
    def __init__(self, args, w_emb_layer, c_emb_layer, r_emb_layers):
	self.args = args
	self.w_emb_layer = w_emb_layer
	self.c_emb_layer = c_emb_layer
	self.r_emb_layers = r_emb_layers

    def save(self, path):
	args = self.args

	if not path.endswith(".pkl.gz"):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        with gzip.open(path, "wb") as fout:
            pickle.dump(
                ([ x.get_value() for x in self.params ], args),
                fout,
                protocol = pickle.HIGHEST_PROTOCOL)

    def load(self, path):
	if not os.path.exists(path):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        with gzip.open(path, "rb") as fin:
            param_values, args = pickle.load(fin)

	self.args = args
	for x,v in zip(self.params, param_values):
            x.set_value(v)

    def ready(self):
	args = self.args
	w_emb_layer = self.w_emb_layer
	c_emb_layer = self.c_emb_layer
	r_emb_layer = self.r_emb_layers[0]

	char_dim = self.char_dim = args.char_dim
	char_lstm_dim = self.char_lstm_dim = args.char_lstm_dim
	word_dim = self.word_dim = args.word_dim
	word_lstm_dim = self.word_lstm_dim = args.word_lstm_dim
	
	dropout = self.dropout = theano.shared(
                np.float64(args.dropout).astype(theano.config.floatX)
            )

	word_ids = self.word_ids = T.ivector('word_ids')
	char_ids = self.char_ids = T.imatrix('char_ids')
	char_lens = self.char_lens = T.fvector('char_lens')
	char_masks = self.char_masks = T.imatrix('char_masks')
	up_ids = self.up_ids = T.imatrix('up_ids')
	up_rels = self.up_rels = T.imatrix('up_rels')
	up_id_masks = self.up_id_masks = T.imatrix('up_id_masks')
	down_ids = self.down_ids = T.imatrix('down_ids')
	down_rels = self.down_rels = T.imatrix('down_rels')
	down_id_masks = self.down_id_masks = T.imatrix('down_id_masks')
	tag_ids = self.tag_ids = T.ivector('tag_ids')
	
	layers = self.layers = [w_emb_layer, c_emb_layer, r_emb_layer]
	
	inputs = self.inputs = []

	inputs.append(self.word_ids)
	inputs.append(self.char_ids)
	inputs.append(self.char_lens)
	inputs.append(self.char_masks)
	inputs.append(self.up_ids)
	inputs.append(self.up_rels)
	inputs.append(self.up_id_masks)
	inputs.append(self.down_ids)
	inputs.append(self.down_rels)
	inputs.append(self.down_id_masks)
	inputs.append(self.tag_ids)
	wslices = w_emb_layer.forward(word_ids)
	cslices = c_emb_layer.forward(char_ids.ravel())
	cslices = cslices.reshape((char_ids.shape[0], char_ids.shape[1], char_dim))
	cslices = cslices.dimshuffle(1, 0, 2)
	
	bv_ur_slices = r_emb_layer.forward(up_rels.ravel())
	bv_dr_slices = r_emb_layer.forward(down_rels.ravel())
	b_ur_slices = r_emb_layer.forward2(up_rels.ravel())
	b_dr_slices = r_emb_layer.forward2(down_rels.ravel())
	bv_ur_slices = bv_ur_slices.reshape((up_rels.shape[0], up_rels.shape[1], word_dim))
	bv_dr_slices = bv_dr_slices.reshape((down_rels.shape[0], down_rels.shape[1], word_dim))
	b_ur_slices = b_ur_slices.reshape((up_rels.shape[0], up_rels.shape[1], word_dim))
	b_dr_slices = b_dr_slices.reshape((down_rels.shape[0], down_rels.shape[1], word_dim))
	
	char_masks = char_masks.dimshuffle(1, 0)

	prev_output = wslices
	prev_size = word_dim

	if char_dim:
	    layers.append(LSTM(
		n_in = char_dim,
		n_out = char_lstm_dim,
		direction = 'bi' if args.char_bidirect else 'si'	
	    ))
	    prev_output_2 = cslices
	    prev_output_2 = apply_dropout(prev_output_2, dropout, v2 = True)
	    prev_output_2 = layers[-1].forward_all(cslices, char_masks)
	    prev_output_2 = T.sum(prev_output_2, axis = 0)
	    prev_output_2 = prev_output_2 / (1e-6 * T.ones_like(char_lens) + char_lens).dimshuffle(0, 'x')

	    prev_size += char_lstm_dim
	    prev_output = T.concatenate([prev_output, prev_output_2], axis = 1)
	
	prev_output = apply_dropout(prev_output, dropout)
	if args.conv != 0:
            for ind in range(args.clayer):
                layers.append(GraphCNN(
                        n_in = prev_size,
                        n_out = prev_size,
                        ))
                prev_output = layers[-1].forward_all(prev_output, up_ids, up_id_masks, bv_ur_slices, b_ur_slices, down_ids, down_id_masks, bv_dr_slices, b_dr_slices)
		prev_output = apply_dropout(prev_output, dropout)
	
	
	
	layers.append(LSTM(
	    n_in = prev_size,
	    n_out = word_lstm_dim,
	    direction = 'bi' if args.word_bidirect else 'si'
	))
	
	prev_output = prev_output.dimshuffle(0, 'x', 1)
	prev_output = layers[-1].forward_all(prev_output)
	prev_output = prev_output.reshape((prev_output.shape[0], prev_output.shape[-1]))
	
	prev_size = word_lstm_dim
	layers.append(SelfMatching(
            n_in = prev_size * 2,
            n_out = word_lstm_dim,
            direction = 'bi' if args.word_bidirect else 'si'
        ))

	prev_output = prev_output.dimshuffle(0, 'x', 1)
        prev_output = layers[-1].forward_all(prev_output)
        prev_output = prev_output.reshape((prev_output.shape[0], prev_output.shape[-1]))

	#prev_output = apply_dropout(prev_output, dropout)
	
	prev_size = word_lstm_dim
	
	layers.append(Layer(
	    n_in = prev_size,
	    n_out = args.classes,
	    activation = linear, #ReLU,
	    has_bias = False
	))

	n_tags = args.classes
	s_len = char_ids.shape[0]
	tags_scores = layers[-1].forward(prev_output)
	transitions = shared((n_tags + 2, n_tags + 2), 'transitions')
	small = -1000
        b_s = np.array([[small] * n_tags + [0, small]]).astype(np.float32)
        e_s = np.array([[small] * n_tags + [small, 0]]).astype(np.float32)
        observations = T.concatenate(
            [tags_scores, small * T.ones((s_len, 2))],
            axis=1
        )
	
        observations = T.concatenate(
            [b_s, observations, e_s],
            axis=0
        )

        real_path_score = tags_scores[T.arange(s_len), tag_ids].sum()
	b_id = theano.shared(value=np.array([n_tags], dtype=np.int32))
        e_id = theano.shared(value=np.array([n_tags + 1], dtype=np.int32))
        padded_tags_ids = T.concatenate([b_id, tag_ids, e_id], axis=0)
	
	pre_ids = T.arange(s_len + 1)
	
	s_ids = T.arange(s_len + 1) + 1
	
        real_path_score += transitions[
           padded_tags_ids[pre_ids],
           padded_tags_ids[s_ids]
        ].sum()
	
	all_paths_scores = CRFForward(observations, transitions)
        self.nll_loss = nll_loss = - (real_path_score - all_paths_scores)
        preds = CRFForward(observations, transitions, viterbi = True,
                        return_alpha = False, return_best_sequence=True)
        
	self.pred = preds[1:-1]
	
	self.l2_sqr = None
        params = self.params = [transitions]
        for layer in layers:
            self.params += layer.params
        for p in self.params:
            if self.l2_sqr is None:
                self.l2_sqr = args.l2_reg * T.sum(p**2)
            else:
                self.l2_sqr += args.l2_reg * T.sum(p**2)

	for l, i in zip(layers[3:], range(len(layers[3:]))):
                say("layer {}: n_in={}\tn_out={}\n".format(
                    i, l.n_in, l.n_out
                ))

        nparams = sum(len(x.get_value(borrow=True).ravel()) \
                        for x in self.params)
        say("total # parameters: {}\n".format(nparams))
	
	cost = self.nll_loss + self.l2_sqr

	lr_method_name = args.learning
	lr_method_parameters = {}
	lr_method_parameters['lr'] = args.learning_rate
	updates = Optimization(clip=5.0).get_updates(lr_method_name, cost, params, **lr_method_parameters)
	
	f_train = theano.function(
	    	inputs = self.inputs,
		outputs = [cost, nll_loss],
		updates = updates,
		allow_input_downcast = True
	)

	f_eval = theano.function(
		inputs = self.inputs[:-1],
		outputs = self.pred,
		allow_input_downcast = True
	)
	
	return f_train, f_eval
