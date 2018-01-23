import re
import os
import sys
import gzip
import numpy as np
import theano
import random 

labelD = {'O':0, 'B-ASP':1, 'I-ASP':2}
labelM = {0:'O', 1:'B-ASP', 2:'I-ASP'}


def shared(shape, name):
    if len(shape) == 1:
        value = np.zeros(shape)  # bias are initialized with zeros
    else:
        drange = np.sqrt(6. / (np.sum(shape)))
        value = drange * np.random.uniform(low=-1.0, high=1.0, size=shape)
    return theano.shared(value=value.astype(theano.config.floatX), name=name)

def say(s, stream=sys.stdout):
    stream.write("{}".format(s))
    stream.flush()

def load_embedding_iterator(path):
    file_open = gzip.open if path.endswith(".gz") else open
    with file_open(path) as fin:
        for line in fin:
            line = line.strip()
            if line:
                parts = line.split()
                word = parts[0]
                vals = np.array([ float(x) for x in parts[1:] ])
                yield word, vals

def words_load(train_sents, test_sents):
    wordLis = {}
    for sent in train_sents:
	tokens = sent.tokens
	for token in tokens:
	    t_str = token.t_str
	    t_size = len(wordLis)
	    if t_str not in wordLis:
		wordLis[t_str] = t_size

    for sent in test_sents:
        tokens = sent.tokens
        for token in tokens:
            t_str = token.t_str
            t_size = len(wordLis)
            if t_str not in wordLis:
                wordLis[t_str] = t_size

    return wordLis.keys()

def chars_load(wordLis):
    charLis = {}
    for word in wordLis:
        for char in word:
	    if char not in charLis:
		charLis[char] = 1
    return charLis.keys()


def rels_load(train_sents, test_sents):
    rel_lis = {}
    for sent in train_sents:
	tokens = sent.tokens
	for token in tokens:
	    rel = token.rel
	    r_rel = 'r_' + token.rel
	    r_size = len(rel_lis)
	    
	    u_rels = token.u_rels
	    d_rels = token.d_rels
	
	    rels = [rel, r_rel]
	    rels.extend(u_rels)
	    rels.extend(d_rels)
    	
	    for rel in rels:
		r_size = len(rel_lis)
		if rel not in rel_lis:
		    rel_lis[rel] = r_size
		
    for sent in test_sents:
        tokens = sent.tokens
        for token in tokens:
	    rel = token.rel
            r_rel = 'r_' + token.rel
            r_size = len(rel_lis)

            u_rels = token.u_rels
            d_rels = token.d_rels

            rels = [rel, r_rel]
            rels.extend(u_rels)
            rels.extend(d_rels)

            for rel in rels:
                r_size = len(rel_lis)
                if rel not in rel_lis:
                    rel_lis[rel] = r_size

    return rel_lis.keys()

def create_input(data, w_emb_layer, c_emb_layer, r_emb_layer):
    input = []
    
    words_ids = []
    chars_ids = []
    chars_len = []
    chars_masks = []
    label_ids = []
    up_ids = []
    up_rels = []
    up_id_masks = []
    down_ids = []
    down_rels = []
    down_id_masks = []
    
    max_c_num = 0
    max_u_num = 0
    max_d_num = 0
    words = []
    for token in data.tokens:
	t_str = token.t_str
	s_str = len(t_str)
	
	label = token.label
	assert label in labelD
	label_ids.append(labelD[label])

	u_ids = token.u_ids
	u_rels = token.u_rels
	d_ids = token.d_ids
	d_rels = token.d_rels
	
	if s_str > max_c_num:
	    max_c_num = s_str
	if len(u_ids) > max_u_num:
	    max_u_num = len(u_ids)
	if len(d_ids) > max_d_num:
	    max_d_num = len(d_ids)

	word = []
	for c_ in t_str:
	    word.append(c_)
	chars_ids.append(c_emb_layer.map_to_ids(word).tolist())
	chars_masks.append([1.0] * len(word))
	chars_len.append(s_str)
	words.append(t_str)
	up_ids.append(u_ids)
	up_rels.append(r_emb_layer.map_to_ids(u_rels).tolist())
	up_id_masks.append([1.0] * len(u_ids))
	down_ids.append(d_ids)
	down_rels.append(r_emb_layer.map_to_ids(d_rels).tolist())
	down_id_masks.append([1.0] * len(d_ids))

    for i in range(len(words)):
	chars_ids[i] = chars_ids[i] + [-1] * (max_c_num - len(chars_ids[i]))
	chars_masks[i] = chars_masks[i] + [0.0] * (max_c_num - len(chars_masks[i]))	
	up_ids[i] = up_ids[i] + [-1] * (max_u_num - len(up_ids[i]))
	up_rels[i] = up_rels[i] + [-1] * (max_u_num - len(up_rels[i]))
	up_id_masks[i] = up_id_masks[i] + [0.0] * (max_u_num - len(up_id_masks[i]))
	down_ids[i] = down_ids[i] + [-1] * (max_d_num - len(down_ids[i]))
	down_rels[i] = down_rels[i] + [-1] * (max_d_num - len(down_rels[i]))
	down_id_masks[i] = down_id_masks[i] + [0.0] * (max_d_num - len(down_id_masks[i]))
	
    words_ids = w_emb_layer.map_to_ids(words).tolist()
    
    words_ids = np.asarray(words_ids, dtype = np.int32)
    chars_ids = np.asarray(chars_ids, dtype = np.int32)
    chars_len = np.asarray(chars_len, dtype = np.float32)
    chars_masks = np.asarray(chars_masks, dtype = np.float32)
    up_ids = np.asarray(up_ids, dtype = np.int32)
    up_rels = np.asarray(up_rels, dtype = np.int32)
    up_id_masks = np.asarray(up_id_masks, dtype = np.float32)
    down_ids = np.asarray(down_ids, dtype = np.int32)
    down_rels = np.asarray(down_rels, dtype = np.int32)
    down_id_masks = np.asarray(down_id_masks, dtype = np.float32)
    label_ids = np.asarray(label_ids, dtype = np.int32)
    input.append(words_ids)
    input.append(chars_ids)
    input.append(chars_len)
    input.append(chars_masks)
    input.append(up_ids)
    input.append(up_rels)
    input.append(up_id_masks)
    input.append(down_ids)
    input.append(down_rels)
    input.append(down_id_masks)
    input.append(label_ids)

    return input

def evaluate(preds, dset):
    score = 0
    pnum = 0
    rnum = 0
    num = 0
	
    for ind, py in enumerate(preds):
	py = py.tolist()
	gy = dset[ind][-1].tolist()
	
	p_aspects = []
	g_aspects = []
	
	ind = 0
	slen = len(py)
	sta = end = 0
	while True:
	    if ind >= slen:
		break
	    if py[ind] == 1:
		sta = ind
	    else:
		if py[ind] == 0 and ind > 0 and py[ind - 1] != 0:
		    end = ind - 1
		    p_aspects.append((sta, end))	
	    ind += 1
	if py[ind - 1] != 0:
	    end = ind - 1
	    p_aspects.append((sta, end))
	ind = 0
	sta = end = 0
	while True:
            if ind >= slen:
                break
            if gy[ind] == 1:
                sta = ind
            else:
                if gy[ind] == 0 and ind > 0 and gy[ind - 1] != 0:
                    end = ind - 1
                    g_aspects.append((sta, end))
            ind += 1
        if gy[ind - 1] != 0:
            end = ind - 1
            g_aspects.append((sta, end))
	
	pnum += len(p_aspects)
	num += len(g_aspects)
	
	g_size = len(g_aspects)
	ind = 0
	for tup1 in p_aspects:
	    for tup2 in g_aspects:
		if tup1[0] == tup2[0] and tup1[1] == tup2[1]:
		    rnum += 1	

    recall = 1.0 * rnum / num 
    precise = 1.0 * rnum / (pnum + 1e-6)
    f1 = 2 * recall * precise / (recall + precise + 1e-6)
    
    outStr = str(rnum) + '#' + str(num) + '#' + str(pnum) +'\n' 
    outStr += 'Recall: %.5f \tPrecise: %.5f\tF1: %.5f' % (recall, precise, f1)
    print outStr
    return f1, precise, recall

