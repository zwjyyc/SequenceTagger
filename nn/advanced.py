'''
    This file contains implementations of advanced NN components, including
      -- Attention layer (two versions)
      -- StrCNN: non-consecutive & non-linear CNN
      -- RCNN: recurrent convolutional network

    Sequential layers (recurrent/convolutional) has two forward methods implemented:
        -- forward(x_t, h_tm1):  one step of forward given input x and previous
                                 hidden state h_tm1; return next hidden state

        -- forward_all(x, h_0):  apply successively steps given all inputs and
                                 initial hidden state, and return all hidden
                                 states h1, ..., h_n

    @author: Tao Lei (taolei@csail.mit.edu)
'''

import numpy as np
import theano
import theano.tensor as T

from .initialization import random_init, create_shared
from .initialization import ReLU, tanh, linear, sigmoid
from .basic import Layer, RecurrentLayer

'''
    This class implements the non-consecutive, non-linear CNN model described in
        Molding CNNs for text (http://arxiv.org/abs/1508.04112)
'''
class StrCNN(Layer):

    def __init__(self, n_in, n_out, activation=None, decay=0.0, order=2, use_all_grams=True, direction = None):
        self.n_in = n_in
        self.n_out = n_out
        self.order = order
        self.use_all_grams = use_all_grams
        self.decay = theano.shared(np.float64(decay).astype(theano.config.floatX))
        if activation is None:
            self.activation = lambda x: x
        else:
            self.activation = activation

        self.create_parameters()

    def create_parameters(self):
        n_in, n_out = self.n_in, self.n_out
        rng_type = "uniform"
        scale = 1.0/self.n_out**0.5
        #rng_type = None
        #scale = 1.0
        self.P = create_shared(random_init((n_in, n_out), rng_type=rng_type)*scale, name="P")
        self.Q = create_shared(random_init((n_in, n_out), rng_type=rng_type)*scale, name="Q")
        self.R = create_shared(random_init((n_in, n_out), rng_type=rng_type)*scale, name="R")
        self.O = create_shared(random_init((n_out, n_out), rng_type=rng_type)*scale, name="O")
        if self.activation == ReLU:
            self.b = create_shared(np.ones(n_out, dtype=theano.config.floatX)*0.01, name="b")
        else:
            self.b = create_shared(random_init((n_out,)), name="b")

    def forward(self, x_t, mask, f1_tm1, s1_tm1, f2_tm1, s2_tm1, f3_tm1):
        P, Q, R, decay = self.P, self.Q, self.R, self.decay
        f1_t = T.dot(x_t, P)
        s1_t = s1_tm1 * decay + f1_t
        f2_t = T.dot(x_t, Q) * s1_tm1
        s2_t = s2_tm1 * decay + f2_t
        f3_t = T.dot(x_t, R) * s2_tm1
        return f1_t * T.cast(mask.dimshuffle(0, 'x'), 'float32'), s1_t * T.cast(mask.dimshuffle(0, 'x'), 'float32'), f2_t * T.cast(mask.dimshuffle(0, 'x'), 'float32'), s2_t * T.cast( mask.dimshuffle(0, 'x'), 'float32'), f3_t * T.cast( mask.dimshuffle(0, 'x'), 'float32')

    def forward_all(self, x, masks = None, v0=None, direction = None):
        if v0 is None:
            if x.ndim > 1:
                v0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
            else:
                v0 = T.zeros((self.n_out,), dtype=theano.config.floatX)
        
	if masks == None:
	    masks = T.ones((x.shape[0], x.shape[1]), dtype=theano.config.floatX)
	([f1, s1, f2, s2, f3], updates) = theano.scan(
                        fn = self.forward,
                        sequences =[ x, masks ],
                        outputs_info = [ v0, v0, v0, v0, v0 ]
                )
        if self.order == 3:
            h = f1+f2+f3 if self.use_all_grams else f3
        elif self.order == 2:
            h = f1+f2 if self.use_all_grams else f2
        elif self.order == 1:
            h = f1
        else:
            raise ValueError(
                    "Unsupported order: {}".format(self.order)
                )
        return self.activation(T.dot(h, self.O) + self.b)

    @property
    def params(self):
        if self.order == 3:
            return [ self.b, self.O, self.P, self.Q, self.R ]
        elif self.order == 2:
            return [ self.b, self.O, self.P, self.Q ]
        elif self.order == 1:
            return [ self.b, self.O, self.P ]
        else:
            raise ValueError(
                    "Unsupported order: {}".format(self.order)
                )

    @params.setter
    def params(self, param_list):
        for p, q in zip(self.params, param_list):
            p.set_value(q.get_value())



'''
    This class implements the *basic* attention layer described in
        Reasoning about Entailment with Neural Attention (http://arxiv.org/abs/1509.06664)

    This layer is uni-directional and non-recurrent.
'''
class AttentionLayer(Layer):
    def __init__(self, n_d, activation):
        self.n_d = n_d
        self.activation = activation
        self.create_parameters()

    def create_parameters(self):
        n_d = self.n_d
        self.W1_c = create_shared(random_init((n_d, n_d)), name="W1_c")
        self.W1_h = create_shared(random_init((n_d, n_d)), name="W1_h")
        self.w = create_shared(random_init((n_d,)), name="w")
        self.W2_r = create_shared(random_init((n_d, n_d)), name="W1_r")
        self.W2_h = create_shared(random_init((n_d, n_d)), name="W1_h")
        self.lst_params = [ self.W1_h, self.W1_c, self.W2_h, self.W2_r, self.w ]

    '''
        One step of attention activation.

        Inputs
        ------

        h_before        : the state before attention at time/position t
        h_after_tm1     : the state after attention at time/position t-1; not used
                          because the current attention implementation is not
                          recurrent
        C               : the context to pay attention to
        mask            : which positions are valid for attention; specify this when
                          some tokens in the context are non-meaningful tokens such
                          as paddings

        Outputs
        -------

        eturn the state after attention at time/position t
    '''
    def forward(self, h_before, h_after_tm1, C, mask=None):
        # C is batch*len*d
        # h is batch*d

        M = self.activation(
                T.dot(C, self.W1_c) + T.dot(h_before, self.W1_h).dimshuffle((0,'x',1))
            )

        # batch*len*1
        alpha = T.nnet.softmax(
                    T.dot(M, self.w)
                )
        alpha = alpha.reshape((alpha.shape[0], alpha.shape[1], 1))
        if mask is not None:
            eps = 1e-8
            if mask.dtype != theano.config.floatX:
                mask = T.cast(mask, theano.config.floatX)
            alpha = alpha*mask.dimshuffle((0,1,'x'))
            alpha = alpha / (T.sum(alpha, axis=1).dimshuffle((0,1,'x'))+eps)

        # batch * d
        r = T.sum(C*alpha, axis=1)

        # batch * d
        h_after = self.activation(
                T.dot(r, self.W2_r) + T.dot(h_before, self.W2_h)
            )
        return h_after

    '''
        Can change this when recurrent attention is needed.
    '''
    def one_step(self, h_before, h_after_tm1, r):
        h_after = self.activation(
                T.dot(r, self.W2_r) + T.dot(h_before, self.W2_h)
            )
        return h_after

    '''
        Apply the attention-based activation to all input tokens x_1, ..., x_n

        Return the post-activation representations
    '''
    def forward_all(self, x, C, mask=None):
        # batch*len2*d
        C2 = T.dot(C, self.W1_c).dimshuffle(('x',0,1,2))
        # len1*batch*d
        x2 = T.dot(x, self.W1_h).dimshuffle((0,1,'x',2))
        # len1*batch*len2*d
        M = self.activation(C2 + x2)

        # len1*batch*len2*1
        alpha = T.nnet.softmax(
                    T.dot(M, self.w).reshape((-1, C.shape[1]))
                )
        alpha = alpha.reshape((x.shape[0],x.shape[1],C.shape[1],1))
        if mask is not None:
            # mask is batch*len2
            if mask.dtype != theano.config.floatX:
                mask = T.cast(mask, theano.config.floatX)
            mask = mask.dimshuffle(('x',0,1,'x'))
            alpha = alpha*mask
            alpha = alpha / (T.sum(alpha, axis=2).dimshuffle((0,1,2,'x')) + 1e-8)

        # len1*batch*d
        r = T.sum(C.dimshuffle(('x',0,1,2)) * alpha, axis=2)

        # len1*batch*d
        h = self.activation(
                T.dot(r, self.W2_r) + T.dot(x, self.W2_h)
            )

        '''
            The current version is non-recurrent, so theano scan is not needed.
            Use scan when recurrent attention is implemented
        '''
        #func = lambda h, r: self.one_step(h, None, r)
        #h, _ = theano.scan(
        #            fn = func,
        #            sequences = [ x, r ]
        #        )
        return h

    @property
    def params(self):
        return self.lst_params

    @params.setter
    def params(self, param_list):
        assert len(param_list) == len(self.lst_params)
        for p, q in zip(self.lst_params, param_list):
            p.set_value(q.get_value())


class Attention(Layer):
    def __init__(self, n_in, activation = tanh):
	self.n_in = n_in
	self.n_out = n_in
	self.activation = activation
	self.create_parameters()

    def create_parameters(self):
        n_d = self.n_in
        self.W_r = create_shared(random_init((n_d, n_d)), name="W_r")
        self.W_m = create_shared(random_init((n_d, n_d)), name="w_m")
	self.W_h = create_shared(random_init((n_d, n_d)), name="W_h")
        self.V = create_shared(random_init((n_d,)), name="V")
	self.H = create_shared(random_init((n_d,)), name="H")
        self.lst_params = [ self.W_r, self.W_h, self.W_m, self.V, self.H ]

    def forward(self, hiddens, svec):
	W_r = self.W_r
	W_m = self.W_m
	W_h = self.W_h
	V = self.V
	H = self.H
	activation = self.activation
	vecs = T.concatenate([H.dimshuffle('x', 'x', 0), hiddens[:-1]], axis = 0)
	pvec = hiddens[-1]
	alpha = T.sum(T.dot(activation(T.dot(svec, W_r) + T.dot(pvec, W_m) + T.dot(hiddens, W_h)), V), axis = 1)
	alpha = T.exp(alpha)
	alphaS = T.sum(alpha)
	alpha = alpha / alphaS
	return T.sum(alpha.dimshuffle(0, 'x', 'x') * vecs, axis = 0)

    @property
    def params(self):
        return self.lst_params

    @params.setter
    def params(self, param_list):
        assert len(param_list) == len(self.lst_params)
        for p, q in zip(self.lst_params, param_list):
            p.set_value(q.get_value())

class GKNNMultiHeadGate(Layer):
    def __init__(self, n_in, n_out, n_head = 4, activation = ReLU):
	self.n_in = n_in
	self.n_out = n_out
	self.n_head = n_head
	self.create_parameters()
	self.activation = activation

    def create_parameters(self):
        n_d = self.n_in
	n_head = self.n_head
        scale = np.sqrt(3.0 / n_d, dtype=theano.config.floatX)
        
	self.W1 = create_shared(random_init((n_d, n_d)) * scale, name = 'W1')
	self.b1 = create_shared(random_init((n_d,)) * 0.001, name = 'b1')
	self.W2 = create_shared(random_init((n_d, n_d)) * scale, name = 'W2')
	self.b2 = create_shared(random_init((n_d,)) * 0.001, name = 'b2')
	
	self.s = create_shared(random_init((n_d,)) * scale, name = 's')
	self.b = create_shared(random_init((n_d,)) * 0.001, name = 'b')
	self.s_ = create_shared(random_init((n_d,)) * scale, name = 's_')
        self.b_ = create_shared(random_init((n_d,)) * 0.001, name = 'b_')
	self.W2s = []
	for i in range(n_head):
	    self.W2s.append(create_shared(random_init((n_d, n_d / n_head)) * scale, name = 'W2_%d_0' % (i)))
	    self.W2s.append(create_shared(random_init((n_d, n_d / n_head)) * scale, name = 'W2_%d_1' % (i)))	
	    self.W2s.append(create_shared(random_init((n_d, n_d / n_head)) * scale, name = 'W2_%d_2' % (i)))
	    self.W2s.append(create_shared(random_init((n_d, n_d / n_head)) * scale, name = 'W2_%d_3' % (i)))

	self.lst_params = [self.W1, self.W2, self.b1, self.b2, self.b, self.s, self.b_, self.s_]
	self.lst_params.extend(self.W2s)

    def forward_order(self, hidden, up_id, up_id_mask, bv_ur_slice, down_id, down_id_mask, bv_dr_slice, pre, t_hs_order_1):
	if up_id_mask.dtype != theano.config.floatX:
            up_id_mask = T.cast(up_id_mask, theano.config.floatX)
	if down_id_mask.dtype != theano.config.floatX:
            down_id_mask = T.cast(down_id_mask, theano.config.floatX)
	
	n_head = self.n_head
	activation = self.activation

	results = []
        for i in range(n_head):
	    tmp = T.dot(hidden, self.W2s[3 * i])
            gate_up = T.nnet.sigmoid(T.dot(bv_ur_slice, self.W2s[3 * i + 1]))
            tmp_S = T.sum(gate_up * up_id_mask.dimshuffle(0, 'x') * T.dot(t_hs_order_1[up_id, :], self.W2s[3 * i + 2]), axis = 0)

            gate_down = T.nnet.sigmoid(T.dot(bv_dr_slice, self.W2s[3 * i + 3]))
            tmp_S += T.sum(gate_down * down_id_mask.dimshuffle(0, 'x') * T.dot(t_hs_order_1[down_id, :], self.W2s[3 * i + 2]), axis = 0 )
	    results.append(tmp_S + tmp)
        #####
        result = T.concatenate(results, axis = 0)
        return result

    def forward_all(self, hiddens, up_ids, up_id_masks, bv_ur_slices, down_ids, down_id_masks, bv_dr_slices):
        h0 = T.zeros_like(hiddens[0])

        t_hiddens = T.ones_like(hiddens) * hiddens

        hs_order, _ = theano.scan(
                fn = self.forward_order,
                sequences = [t_hiddens, up_ids, up_id_masks, bv_ur_slices, down_ids, down_id_masks, bv_dr_slices],
                outputs_info = h0,
                non_sequences = hiddens
            )

	_eps = 1e-5
	# Add & Norm
	hs_order = hs_order + hiddens
	hs_order = (hs_order - hs_order.mean(1)[:, None]) / T.sqrt(hs_order.var(1)[:, None] + _eps)
	hs_order = self.s[None, :] * hs_order + self.b[None, :]
	
	FFNx = self.activation(T.dot(hs_order, self.W1) + self.b1)
	FFNx = T.dot(FFNx, self.W2) + self.b2
	
	# Add & Norm
	FFNx = FFNx + hs_order

	FFNx = (FFNx - FFNx.mean(1)[:, None]) / T.sqrt(FFNx.var(1)[:, None] + _eps)
        FFNx = self.s_[None, :] * FFNx + self.b_[None, :]
	return FFNx
    @property
    def params(self):
        return self.lst_params

    @params.setter
    def params(self, param_list):
        assert len(param_list) == len(self.lst_params)
        for p, q in zip(self.lst_params, param_list):
            p.set_value(q.get_value())


class GraphKernelNN(Layer):
    def __init__(self, n_in, n_out, activation=ReLU):
        self.n_in = n_in
        self.n_out = n_out
        self.create_parameters()
        self.activation = activation

    def create_parameters(self):
        n_d = self.n_in
        scale = np.sqrt(3.0 / n_d, dtype=theano.config.floatX)
        self.W1 = create_shared(random_init((n_d, n_d)) * scale, name = 'W1')
	self.W2 = create_shared(random_init((n_d, n_d)) * scale, name = 'W2')
	self.W21 = create_shared(random_init((n_d, n_d)) * scale, name = 'W21')
	self.W22 = create_shared(random_init((n_d, n_d)) * scale, name = 'W22')
	self.W3 = create_shared(random_init((n_d, n_d)) * scale, name = 'W3')
	self.W31 = create_shared(random_init((n_d, n_d)) * scale, name = 'W31')
	self.W32 = create_shared(random_init((n_d, n_d)) * scale, name = 'W32')
        self.lst_params = [self.W1, self.W2, self.W21, self.W22, self.W3, self.W31, self.W32]

    def forward_order_1(self, hidden, pre):
	return T.dot(hidden, self.W1)

    def forward_order_2(self, hidden, up_id, up_id_mask, bv_ur_slice, down_id, down_id_mask, bv_dr_slice, pre, t_hs_order_1):
	tmp = T.dot(hidden, self.W2)


	#gate_up = T.nnet.sigmoid(bv_ur_slice)
	gate_up = T.nnet.sigmoid(T.dot(bv_ur_slice, self.W21))
	if up_id_mask.dtype != theano.config.floatX:
            up_id_mask = T.cast(up_id_mask, theano.config.floatX)
	
	tmp_S = gate_up * up_id_mask.dimshuffle(0, 'x') * t_hs_order_1[up_id, :]
	tmp_S = T.sum(tmp_S, axis = 0)	

	gate_down = T.nnet.sigmoid(T.dot(bv_dr_slice, self.W22))
	if down_id_mask.dtype != theano.config.floatX:
	    down_id_mask = T.cast(down_id_mask, theano.config.floatX)
	tmp_S += T.sum(gate_down * down_id_mask.dimshuffle(0, 'x') * t_hs_order_1[down_id, :], axis = 0)
	
	
	#####
	result = tmp + tmp_S
	return result

    def forward_order_3(self, hidden, up_id, up_id_mask, bv_ur_slice, down_id, down_id_mask, bv_dr_slice, pre, t_hs_order_2):
	tmp = T.dot(hidden, self.W3)

        gate_up = T.nnet.sigmoid(T.dot(bv_ur_slice, self.W31))
        if up_id_mask.dtype != theano.config.floatX:
            up_id_mask = T.cast(up_id_mask, theano.config.floatX)

        tmp_S = gate_up * up_id_mask.dimshuffle(0, 'x') * t_hs_order_2[up_id, :]
	tmp_S = T.sum(tmp_S, axis = 0)

        gate_down = T.nnet.sigmoid(T.dot(bv_dr_slice, self.W32))
        if down_id_mask.dtype != theano.config.floatX:
            down_id_mask = T.cast(down_id_mask, theano.config.floatX)
        tmp_S += T.sum(gate_down * down_id_mask.dimshuffle(0, 'x') * t_hs_order_2[down_id, :], axis = 0)

        #####
        result = tmp + tmp_S
        return result

    def forward_all(self, hiddens, up_ids, up_id_masks, bv_ur_slices, down_ids, down_id_masks, bv_dr_slices):
        h0 = T.zeros_like(hiddens[0])
        
	t_hiddens = T.ones_like(hiddens) * hiddens
        #ids = T.arange(hiddens.shape[0])
        
	hs_order_1, _ = theano.scan(
               fn = self.forward_order_1,
               sequences = t_hiddens,
               outputs_info =  h0
            )

	
	t_hs_order_1 = T.ones_like(hs_order_1) * hs_order_1
	hs_order_2, _ = theano.scan(
		fn = self.forward_order_2,
		sequences = [t_hiddens, up_ids, up_id_masks, bv_ur_slices, down_ids, down_id_masks, bv_dr_slices],
		outputs_info = h0,
		non_sequences = t_hs_order_1
	    )
	
	#t_hs_order_2 = T.ones_like(hs_order_2) * hs_order_2
	#hs_order_3, _ = theano.scan(
	#	fn = self.forward_order_3,
	#	sequences = [t_hiddens, up_ids, up_id_masks, bv_ur_slices, down_ids, down_id_masks, bv_dr_slices],
	#	outputs_info = h0,
	#	non_sequences = t_hs_order_2
	#    )
	return self.activation(T.concatenate([hs_order_1, hs_order_2], axis = 1))
        return self.activation(T.concatenate([hs_order_1, hs_order_2, hs_order_3], axis = 1))

    @property
    def params(self):
        return self.lst_params

    @params.setter
    def params(self, param_list):
        assert len(param_list) == len(self.lst_params)
        for p, q in zip(self.lst_params, param_list):
            p.set_value(q.get_value())

class GraphCNNTensor(Layer):
    def __init__(self, n_in, n_out, activation=ReLU):
        self.n_in = n_in
        self.n_out = n_out
        self.create_parameters()
        self.activation = activation

    def create_parameters(self):
        n_d = self.n_in
        scale = np.sqrt(3.0 / n_d, dtype=theano.config.floatX)
        self.W_loop = create_shared(random_init((n_d, n_d)) * scale , name='W_LOOP')
        self.b_loop = create_shared(random_init((n_d, )) * scale, name='b_LOOP')
        self.b = create_shared(np.ones((n_d, ), dtype=theano.config.floatX) * 0.01, name='b')
        self.lst_params = [self.W_loop, self.b_loop, self.b]

    def forward(self, id, up_id, up_id_mask, bv_ur_slice, bv_ur_matrix, b_ur_slice, b_ur_matrix, down_id, down_id_mask, bv_dr_slice, bv_dr_matrix, b_dr_slice, b_dr_matrix, previous, hiddens):
        hidden = hiddens[id]
        #### Loop
        h_loop_vec = T.dot(hidden, self.W_loop) + self.b_loop
	
	#### Up
        tmp = hiddens[up_id, :]
	
        h_up_vec = T.batched_dot(bv_ur_matrix, tmp.dimshuffle(0, 1, 'x'))
	h_up_vec = h_up_vec.reshape((h_up_vec.shape[0], h_up_vec.shape[1])) + bv_ur_slice
        
	if up_id_mask.dtype != theano.config.floatX:
            up_id_mask = T.cast(up_id_mask, theano.config.floatX)

        gate = T.nnet.sigmoid(T.batched_dot(b_ur_matrix, tmp.dimshuffle(0, 1, 'x')))
	gate = (gate.reshape((gate.shape[0], gate.shape[1])) + b_ur_slice) * up_id_mask.dimshuffle(0, 'x')

        if gate.ndim == 1:
            gate = gate.dimshuffle(0, 'x')

        h_up_vec = T.sum(h_up_vec * gate, axis = 0)
        #### Down
        tmp = hiddens[down_id, :]
        h_down_vec = T.batched_dot(bv_dr_matrix, tmp.dimshuffle(0, 1, 'x'))
	h_down_vec = h_down_vec.reshape((h_down_vec.shape[0], h_down_vec.shape[1])) + bv_dr_slice
        if down_id_mask.dtype != theano.config.floatX:
            down_id_mask = T.cast(down_id_mask, theano.config.floatX)
        gate = T.nnet.sigmoid(T.batched_dot(b_dr_matrix, tmp.dimshuffle(0, 1, 'x')))
	gate = (gate.reshape((gate.shape[0], gate.shape[1])) + b_dr_slice) * down_id_mask.dimshuffle(0, 'x')

        if gate.ndim == 1:
            gate = gate.dimshuffle(0, 'x')
        h_down_vec = T.sum(h_down_vec * gate, axis = 0)
        ### sum
        return self.activation(T.concatenate([h_loop_vec + self.b, h_up_vec + self.b, h_down_vec + self.b], axis = 0))


    def forward_all(self, hiddens, up_ids, up_id_masks, bv_ur_slices, bv_ur_matrixs, b_ur_slices, b_ur_matrixs, down_ids, down_id_masks, bv_dr_slices, bv_dr_matrixs, b_dr_slices, b_dr_matrixs, residual = False):
        h0 = T.zeros_like(T.concatenate([hiddens[0], hiddens[1], hiddens[0]], axis = 0))
        #h0 = T.zeros_like(hiddens[0])
        t_hiddens = T.ones_like(hiddens) * hiddens
        ids = T.arange(hiddens.shape[0])
        h, _ = theano.scan(
               fn = self.forward,
               sequences = [ids, up_ids, up_id_masks, bv_ur_slices, bv_ur_matrixs, b_ur_slices, b_ur_matrixs, down_ids, down_id_masks, bv_dr_slices, bv_dr_matrixs, b_dr_slices, b_dr_matrixs],
               outputs_info =  h0,
               non_sequences = t_hiddens
            )
        if residual:
            h = h + hiddens
        return h

    @property
    def params(self):
        return self.lst_params

    @params.setter
    def params(self, param_list):
        assert len(param_list) == len(self.lst_params)
        for p, q in zip(self.lst_params, param_list):
            p.set_value(q.get_value())


class GraphCNN(Layer):
    def __init__(self, n_in, n_out, activation=ReLU):
        self.n_in = n_in
        self.n_out = n_out
	self.create_parameters()
	self.activation = activation

    def create_parameters(self):
	n_d = self.n_in
	scale = np.sqrt(3.0 / n_d, dtype=theano.config.floatX)
	self.W_up = create_shared(random_init((n_d, n_d)) * scale , name='W_UP') 
	self.W_down = create_shared(random_init((n_d, n_d)) * scale, name='W_DOWN') 
	self.W_loop = create_shared(random_init((n_d, n_d)) * scale , name='W_LOOP')
	self.w_up = create_shared(random_init((n_d, n_d)) * scale , name='w_UP') 
	self.w_down = create_shared(random_init((n_d, n_d)) * scale, name='w_DOWN') 
	self.b_loop = create_shared(random_init((n_d, )) * scale, name='b_LOOP') 
	self.b = create_shared(np.ones((n_d, ), dtype=theano.config.floatX) * 0.01, name='b') 
	self.lst_params = [self.W_up, self.W_down, self.W_loop, self.w_up, self.w_down, self.b_loop, self.b]
	
    def forward(self, id, up_id, up_id_mask, bv_ur_slice, b_ur_slice, down_id, down_id_mask, bv_dr_slice, b_dr_slice, previous, hiddens):
	hidden = hiddens[id]
	#### Loop
	h_loop_vec = T.dot(hidden, self.W_loop) + self.b_loop
	#### Up
	tmp = hiddens[up_id, :]
	h_up_vec = T.dot(tmp, self.W_up) + bv_ur_slice
        if up_id_mask.dtype != theano.config.floatX:
            up_id_mask = T.cast(up_id_mask, theano.config.floatX)

	gate = T.nnet.sigmoid(T.dot(tmp, self.w_up) + b_ur_slice) * up_id_mask.dimshuffle(0, 'x')
	
	if gate.ndim == 1:
	    gate = gate.dimshuffle(0, 'x')
	
	h_up_vec = T.sum(h_up_vec * gate, axis = 0)
	#### Down
        tmp = hiddens[down_id, :]
        h_down_vec = T.dot(tmp, self.W_down) + bv_dr_slice
	if down_id_mask.dtype != theano.config.floatX:
            down_id_mask = T.cast(down_id_mask, theano.config.floatX)
        gate = T.nnet.sigmoid(T.dot(tmp, self.w_down) + b_dr_slice) * down_id_mask.dimshuffle(0, 'x')

	if gate.ndim == 1:
	    gate = gate.dimshuffle(0, 'x')
        h_down_vec = T.sum(h_down_vec * gate, axis = 0)
	### sum
	#vec_ = T.dot(T.concatenate([h_loop_vec, h_up_vec, h_down_vec], axis = 0), self.W) + self.b
	#gate =  T.nnet.sigmoid(T.dot(T.concatenate([h_loop_vec, h_up_vec, h_down_vec], axis = 0), self.W1) + self.b1)
	#return self.activation(vec_ * gate) 
	#return self.activation(T.concatenate([h_loop_vec + self.b, h_up_vec + self.b, h_down_vec + self.b], axis = 0))
	return self.activation(h_loop_vec + h_up_vec + h_down_vec + self.b)
	return T.concatenate([h_loop_vec, self.activation(h_up_vec + h_down_vec)], axis = 0)
	

    def forward_all(self, hiddens, up_ids, up_id_masks, bv_ur_slices, b_ur_slices, down_ids, down_id_masks, bv_dr_slices, b_dr_slices, residual = False):
	#h0 = T.zeros_like(T.concatenate([hiddens[0], hiddens[1], hiddens[0]], axis = 0))
	h0 = T.zeros_like(hiddens[0])
	t_hiddens = T.ones_like(hiddens) * hiddens
	ids = T.arange(hiddens.shape[0])
	h, _ = theano.scan(
               fn = self.forward,
               sequences = [ids, up_ids, up_id_masks, bv_ur_slices, b_ur_slices, down_ids, down_id_masks, bv_dr_slices, b_dr_slices],
               outputs_info =  h0,
               non_sequences = t_hiddens
            )
	if residual:
	    h = h + hiddens
	return h	

    @property
    def params(self):
        return self.lst_params

    @params.setter
    def params(self, param_list):
        assert len(param_list) == len(self.lst_params)
        for p, q in zip(self.lst_params, param_list):
            p.set_value(q.get_value())


class SentenceAttn(Layer):
    def __init__(self, n_in, n_out):
	self.n_in = n_in
	self.n_out = n_out
	self.create_parameters()

    def create_parameters(self):
	n_d = self.n_in
	#self.W = create_shared(random_init((n_d, n_d)), name="W")
	self.W = create_shared(random_init((3 * n_d, )), name="W")
	self.lst_params = [self.W]
    	
    def forward(self, x, mask, previous, pvec):
	W = self.W
	
	copyOnes = T.ones((x.shape[0], x.shape[0]))
	xC = x.dimshuffle(0, 'x', 1)
	pvecC = pvec.dimshuffle('x', 0, 1)
	
	dotP = xC * pvecC
	xCs = xC * copyOnes.dimshuffle(0, 1, 'x')
	pvecCs = pvecC * copyOnes.dimshuffle(0, 1, 'x')
	
	affMatrix = T.tanh( T.dot(T.concatenate([xCs, pvecCs, dotP], axis = 2), W))

	#x_ = T.dot(x, W)
	#pvec_ = pvec.dimshuffle(1, 0)
	
	#affMatrix = T.tanh(T.dot(x_, pvec_))
	#affMatrix = T.exp(affMatrix) ### sum -> exp or exp -> sum ?
	
	alpha = T.sum(affMatrix, axis = 1) * mask
	alphaS = T.sum(alpha)
	
	alpha = alpha / (alphaS  + 1e-6)
	
	sumV = T.sum(alpha.dimshuffle(0, 'x') * x, axis = 0)
	return sumV
	
    def forward_all(self, hiddens, masks, lens):
	tmp = hiddens.dimshuffle(1, 0, 2)
	last_hidden = tmp[-1]
	masks_ = masks.dimshuffle(1, 0)

	#padding = T.zeros_like(tmp[0]).dimshuffle('x', 0, 1)
	#padding_masks = T.zeros_like(masks_[0]).dimshuffle('x', 0)
	
	h0 = T.zeros((last_hidden.shape[1],))
	
	prev_hidden = tmp
	t_masks = masks_
	#prev_hidden = T.concatenate([padding, tmp], axis = 0)
	#t_masks = T.concatenate([padding_masks, masks_], axis = 0)
	
	h, _ = theano.scan(
                    fn = self.forward,
                    sequences = [prev_hidden, t_masks],
                    outputs_info =  h0,
		    non_sequences = last_hidden
         	)
	
	last_hidden_ = T.sum(last_hidden, axis = 0) / lens[-1] # (T.sum(last_masks) + 1e-6)
	h = T.concatenate([h[0:-1], last_hidden_.dimshuffle('x', 0)], axis = 0)
	return h

    @property
    def params(self):
        return self.lst_params

    @params.setter
    def params(self, param_list):
        assert len(param_list) == len(self.lst_params)
        for p, q in zip(self.lst_params, param_list):
            p.set_value(q.get_value())

'''
    This class implements the attention layer described in
        A Neural Attention Model for Abstractive Sentence Summarization
        (http://arxiv.org/pdf/1509.00685.pdf)

    This layer is uni-directional and non-recurrent.
'''
class BilinearAttentionLayer(Layer):
    def __init__(self, n_d, activation, weighted_output=True):
        self.n_d = n_d
        self.activation = activation
        self.weighted_output = weighted_output
        self.create_parameters()

    def create_parameters(self):
        n_d = self.n_d
        self.P = create_shared(random_init((n_d, n_d)), name="P")
        self.W_r = create_shared(random_init((n_d, n_d)), name="W_r")
        self.W_h = create_shared(random_init((n_d, n_d)), name="W_h")
        self.b = create_shared(random_init((n_d,)), name="b")
        self.lst_params = [ self.P, self.W_r, self.W_h, self.b ]

    '''
        One step of attention activation.

        Inputs
        ------

        h_before        : the state before attention at time/position t
        h_after_tm1     : the state after attention at time/position t-1; not used
                          because the current attention implementation is not
                          recurrent
        C               : the context to pay attention to
        mask            : which positions are valid for attention; specify this when
                          some tokens in the context are non-meaningful tokens such
                          as paddings

        Outputs
        -------

        return the state after attention at time/position t
    '''
    def forward(self, h_before, h_after_tm1, C, mask=None):
        # C is batch*len*d
        # h is batch*d
        # mask is batch*len

        # batch*1*d
        #M = T.dot(h_before, self.P).dimshuffle((0,'x',1))
        M = T.dot(h_before, self.P).reshape((h_before.shape[0], 1, h_before.shape[1]))

        # batch*len*1
        alpha = T.nnet.softmax(
                    T.sum(C * M, axis=2)
                )
        alpha = alpha.reshape((alpha.shape[0], alpha.shape[1], 1))
        if mask is not None:
            eps = 1e-8
            if mask.dtype != theano.config.floatX:
                mask = T.cast(mask, theano.config.floatX)
            alpha = alpha*mask.dimshuffle((0,1,'x'))
            alpha = alpha / (T.sum(alpha, axis=1).dimshuffle((0,1,'x'))+eps)

        # batch * d
        r = T.sum(C*alpha, axis=1)

        # batch * d
        if self.weighted_output:
            beta = T.nnet.sigmoid(
                    T.dot(r, self.W_r) + T.dot(h_before, self.W_h) + self.b
                )
            h_after = beta*h_before + (1.0-beta)*r
        else:
            h_after = self.activation(
                    T.dot(r, self.W_r) + T.dot(h_before, self.W_h) + self.b
                )
        return h_after

    '''
        Apply the attention-based activation to all input tokens x_1, ..., x_n

        Return the post-activation representations
    '''
    def forward_all(self, x, C, mask=None):
        # x is len1*batch*d
        # C is batch*len2*d
        # mask is batch*len2

        C2 = C.dimshuffle(('x',0,1,2))

        # batch*len1*d
        M = T.dot(x, self.P).dimshuffle((1,0,2))
        # batch*d*len2
        C3 = C.dimshuffle((0,2,1))

        alpha = T.batched_dot(M, C3).dimshuffle((1,0,2))
        alpha = T.nnet.softmax(
                    alpha.reshape((-1, C.shape[1]))
                )
        alpha = alpha.reshape((x.shape[0],x.shape[1],C.shape[1],1))
        if mask is not None:
            # mask is batch*len1
            if mask.dtype != theano.config.floatX:
                mask = T.cast(mask, theano.config.floatX)
            mask = mask.dimshuffle(('x',0,1,'x'))
            alpha = alpha*mask
            alpha = alpha / (T.sum(alpha, axis=2).dimshuffle((0,1,2,'x')) + 1e-8)

        # len1*batch*d
        r = T.sum(C2*alpha, axis=2)

        # len1*batch*d
        if self.weighted_output:
            beta = T.nnet.sigmoid(
                        T.dot(r, self.W_r) + T.dot(x, self.W_h) + self.b
                    )
            h = beta*x + (1.0-beta)*r
        else:
            h = self.activation(
                    T.dot(r, self.W_r) + T.dot(x, self.W_h) + self.b
                )

        '''
            The current version is non-recurrent, so theano scan is not needed.
            Use scan when recurrent attention is implemented
        '''
        #func = lambda h, r: self.one_step(h, None, r)
        #h, _ = theano.scan(
        #            fn = func,
        #            sequences = [ x, r ]
        #        )
        return h


    @property
    def params(self):
        return self.lst_params

    @params.setter
    def params(self, param_list):
        assert len(param_list) == len(self.lst_params)
        for p, q in zip(self.lst_params, param_list):
            p.set_value(q.get_value())


class CNN(Layer):

    '''
        CNN

        Inputs
        ------

            order           : CNN feature width
    '''
    def __init__(self, n_in, n_out, activation=tanh,
            order=1,  clip_gradients=False, direction = None):

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

    def forward(self, x, mask, hc):
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
            in_i_t = layers[i].forward(x)
            if i == 0:
                c_i_t =  in_i_t
            else:
                c_i_t =  in_i_t + c_im1_tm1
            
	    lst.append(T.cast(c_i_t * mask.dimshuffle(0, 'x'), 'float32'))
            c_im1_tm1 = c_i_tm1
            c_im1_t = c_i_t

        h_t = activation(c_i_t + self.bias)
        lst.append(T.cast(h_t * mask.dimshuffle(0, 'x'), 'float32'))
        if hc.ndim > 1:
            return T.concatenate(lst, axis=1)
        else:
            return T.concatenate(lst)

    def forward_all(self, x, masks = None, h0=None, return_c=False, direction = None):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out*(self.order+1)), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out*(self.order+1),), dtype=theano.config.floatX)

        if masks == None:
            masks = T.ones((x.shape[0], x.shape[1]), dtype = theano.config.floatX)
        h, _ = theano.scan(
                    fn = self.forward,
                    sequences = [x, masks],
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


'''
    This class implements the recurrent convolutional network model described in
        Retrieving Similar Questions with Recurrent Convolutional Models
        (http://arxiv.org/abs/1512.05726)
'''
class RCNN(Layer):

    '''
        RCNN

        Inputs
        ------

            order           : CNN feature width
            has_outgate     : whether to add a output gate as in LSTM; this can be
                              useful for language modeling
            mode            : 0 if non-linear filter; 1 if linear filter (default)
    '''
    def __init__(self, n_in, n_out, activation=tanh,
            order=1, has_outgate=False, mode=1, clip_gradients=False, direction = None):

        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.order = order
        self.clip_gradients = clip_gradients
        self.has_outgate = has_outgate
        self.mode = mode

        internal_layers = self.internal_layers = [ ]
        for i in range(order):
            input_layer = Layer(n_in, n_out, linear, has_bias=False, \
                    clip_gradients=clip_gradients)
            internal_layers.append(input_layer)

        forget_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
        internal_layers.append(forget_gate)

        self.bias = create_shared(random_init((n_out,)), name="bias")

        if has_outgate:
            self.out_gate = RecurrentLayer(n_in, n_out, sigmoid, clip_gradients)
            self.internal_layers += [ self.out_gate ]

    '''
        One step of recurrent

        Inputs
        ------

            x           : input token at current time/position t
            hc          : hidden/visible states at time/position t-1

        Outputs
        -------

            return hidden/visible states at time/position t
    '''
    def forward(self, x, mask, hc):
        order, n_in, n_out, activation = self.order, self.n_in, self.n_out, self.activation
        layers = self.internal_layers
        if hc.ndim > 1:
            h_tm1 = hc[:, n_out*order:]
        else:
            h_tm1 = hc[n_out*order:]

        forget_t = layers[order].forward(x, h_tm1)
        lst = [ ]
        for i in range(order):
            if hc.ndim > 1:
                c_i_tm1 = hc[:, n_out*i:n_out*i+n_out]
            else:
                c_i_tm1 = hc[n_out*i:n_out*i+n_out]
            in_i_t = layers[i].forward(x)
            if i == 0:
                c_i_t = forget_t * c_i_tm1 + (1-forget_t) * in_i_t
            elif self.mode == 0:
                c_i_t = forget_t * c_i_tm1 + (1-forget_t) * (in_i_t * c_im1_t)
            else:
                c_i_t = forget_t * c_i_tm1 + (1-forget_t) * (in_i_t + c_im1_tm1)
            lst.append(T.cast(c_i_t * mask.dimshuffle(0, 'x'), 'float32'))
	    c_im1_tm1 = c_i_tm1
            c_im1_t = c_i_t

        if not self.has_outgate:
            h_t = activation(c_i_t + self.bias)
        else:
            out_t = self.out_gate.forward(x, h_tm1)
            h_t = out_t * activation(c_i_t + self.bias)
	lst.append(T.cast(h_t * mask.dimshuffle(0, 'x'), 'float32'))
        if hc.ndim > 1:
            return T.concatenate(lst, axis=1)
        else:
            return T.concatenate(lst)

    '''
        Apply recurrent steps to input of all positions/time

        Inputs
        ------

            x           : input tokens x_1, ... , x_n
            h0          : initial states
            return_c    : whether to return hidden states in addition to visible
                          state

        Outputs
        -------

            return visible states (and hidden states) of all positions/time
    '''
    def forward_all(self, x, masks = None, h0=None, return_c=False, direction = None):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out*(self.order+1)), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out*(self.order+1),), dtype=theano.config.floatX)
        
	if masks == None:
	    masks = T.ones((x.shape[0], x.shape[1]), dtype = theano.config.floatX)
	h, _ = theano.scan(
                    fn = self.forward,
                    sequences = [x, masks],
                    outputs_info = [ h0 ]
                )
        if return_c:
            return h
        elif x.ndim > 1:
            return h[:,:,self.n_out*self.order:]
        else:
            return h[:,self.n_out*self.order:]

    def forward2(self, x, hc, f_tm1):
        order, n_in, n_out, activation = self.order, self.n_in, self.n_out, self.activation
        layers = self.internal_layers
        if hc.ndim > 1:
            h_tm1 = hc[:, n_out*order:]
        else:
            h_tm1 = hc[n_out*order:]

        forget_t = layers[order].forward(x, h_tm1)
        lst = [ ]
        for i in range(order):
            if hc.ndim > 1:
                c_i_tm1 = hc[:, n_out*i:n_out*i+n_out]
            else:
                c_i_tm1 = hc[n_out*i:n_out*i+n_out]
            in_i_t = layers[i].forward(x)
            if i == 0:
                c_i_t = forget_t * c_i_tm1 + (1-forget_t) * in_i_t
            elif self.mode == 0:
                c_i_t = forget_t * c_i_tm1 + (1-forget_t) * (in_i_t * c_im1_t)
            else:
                c_i_t = forget_t * c_i_tm1 + (1-forget_t) * (in_i_t + c_im1_tm1)
            lst.append(c_i_t)
            c_im1_tm1 = c_i_tm1
            c_im1_t = c_i_t

        if not self.has_outgate:
            h_t = activation(c_i_t + self.bias)
        else:
            out_t = self.out_gate.forward(x, h_tm1)
            h_t = out_t * activation(c_i_t + self.bias)
        lst.append(h_t)

        if hc.ndim > 1:
            return T.concatenate(lst, axis=1), forget_t
        else:
            return T.concatenate(lst), forget_t

    def get_input_gate(self, x, h0=None):
        if h0 is None:
            if x.ndim > 1:
                h0 = T.zeros((x.shape[1], self.n_out*(self.order+1)), dtype=theano.config.floatX)
                f0 = T.zeros((x.shape[1], self.n_out), dtype=theano.config.floatX)
            else:
                h0 = T.zeros((self.n_out*(self.order+1),), dtype=theano.config.floatX)
                f0 = T.zeros((self.n_out,), dtype=theano.config.floatX)

        [h, f], _ = theano.scan(
                    fn = self.forward2,
                    sequences = x,
                    outputs_info = [ h0,f0 ]
                )
        return 1.0-f

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

def log_sum_exp(x, axis=None):
    """
    Sum probabilities in the log-space.
    """
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

def TreeCRFForward(observations, transitions, sids, pids, flags, tids, margin, generate = False, viterbi=False,
            return_alpha=False, return_best_sequence=False):
    
    assert not return_best_sequence or (viterbi and not return_alpha)

    def recurrence(obs, sid, pid, tid, previous, transitions, ones):
        previous_t = previous[pid].dimshuffle(0, 'x')
	loss = T.concatenate([ones[:tid], [T.zeros_like(ones[tid])], ones[tid+1:]])
        obs = (obs + loss).dimshuffle('x', 0)
        if viterbi:
            scores = previous_t + obs + transitions
            tmp = scores.max(axis=0)
	    out = T.concatenate([previous[:sid, :], tmp.dimshuffle('x', 0), previous[sid+1:, :]], axis = 0)
            if return_best_sequence:
                out2 = scores.argmax(axis=0)
                return out, out2
            else:
                return out
        else:
	    tmp = log_sum_exp(previous_t + obs + transitions, axis=0)
	    previous = T.concatenate([previous[0:sid, :], tmp.dimshuffle('x', 0), previous[sid+1:, :]], axis = 0)
            return previous

    initial = T.zeros_like(observations) + observations
    ones = T.zeros_like(observations[0]) * margin
    if generate:
	ones = T.ones_like(observations[0]) * margin
	
    alpha, _ = theano.scan(
        fn=recurrence,
        outputs_info=(initial, None) if return_best_sequence else initial,
        sequences=[observations[1:], sids, pids, tids],
        non_sequences=[transitions, ones]
    )

    if return_alpha:
        return alpha[-1]
    elif return_best_sequence:
	tmp = alpha[1]
	tmp = tmp[pids[::-1]]
        sequence, _ = theano.scan(
            fn=lambda beta_i, previous: beta_i[previous],
            outputs_info=T.cast(T.argmax(alpha[0][-1][-1]), 'int32'),
            sequences=T.cast(tmp, 'int32')
        )
        sequence = T.concatenate([sequence[::-1], [T.argmax(alpha[0][-1][-1])]])
        return sequence
    else:
        if viterbi:
            return alpha[-1][-1].max(axis=0)
        else:
	    fn = lambda flag, vec: log_sum_exp(vec, axis = 0) * flag
	    fn2 = lambda flag, vec: vec * flag
	    #return log_sum_exp(T.sum(fn2(flags.dimshuffle(0, 'x'), alpha[-1]), axis = 0), axis = 0)
            return T.sum(fn(flags.dimshuffle(0, 'x'), alpha[-1]))

def CRFForward(observations, transitions,  viterbi=False,
            return_alpha=False, return_best_sequence=False):
    assert not return_best_sequence or (viterbi and not return_alpha)

    def recurrence(obs, previous, transitions):
        previous = previous.dimshuffle(0, 'x')
        obs = obs.dimshuffle('x', 0)
        if viterbi:
            scores = previous + obs + transitions
            out = scores.max(axis=0)
            if return_best_sequence:
                out2 = scores.argmax(axis=0)
                return out, out2
            else:
                return out
        else:
            return log_sum_exp(previous + obs + transitions, axis=0)

    initial = observations[0]
    alpha, _ = theano.scan(
        fn=recurrence,
        outputs_info=(initial, None) if return_best_sequence else initial,
        sequences=[observations[1:]],
        non_sequences=transitions
    )

    if return_alpha:
        return alpha
    elif return_best_sequence:
        sequence, _ = theano.scan(
            fn=lambda beta_i, previous: beta_i[previous],
            outputs_info=T.cast(T.argmax(alpha[0][-1]), 'int32'),
            sequences=T.cast(alpha[1][::-1], 'int32')
        )
        sequence = T.concatenate([sequence[::-1], [T.argmax(alpha[0][-1])]])
        return sequence
    else:
        if viterbi:
            return alpha[-1].max(axis=0)
        else:
            return log_sum_exp(alpha[-1], axis=0)

