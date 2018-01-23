import sys

from data import Token, Sentence


class processor(object):
    def __init__(self, src):
	assert src != None
	self.corpus = []
	self.src = src
	
    def loadSrc(self):
	corpus = self.corpus
	src = self.src
	
	tokens = []
	id = 0
	with open(src, 'r') as fin:
	    for line in fin.readlines():
		if line == '\n':
		    tmp_tokens = list(tokens)
		    le = len(tmp_tokens)
		    for i, token in enumerate(tmp_tokens):
			h_id = token.h_id
			h_rel = token.rel
			
			if i < le - 1:
			    tmp_tokens[i].add_d_id_rel(i + 1, '@+1@')
			
			if i > 0:
			    tmp_tokens[i].add_u_id_rel(i - 1, '@-1@')
			
			if h_id != -1:
			    tmp_tokens[h_id].add_d_id_rel(i, h_rel) 
			
		    sent = Sentence(tmp_tokens)
		    corpus.append(sent)
		    tokens = []
		    
		    id = 0
		else:
		    items = line.strip().split()
		    t_str = items[0]
		    h_id = int(items[1])
		    rel = items[2]
		    label = items[3]
		    token = Token(id, t_str, h_id, rel, label)
		    tokens.append(token)
		    id += 1
	return corpus

#processor = processor(src=sys.argv[1])
#processor.loadSrc()
#processor.loadSrc()
#sset = processor.loadLabel(sys.argv[2])
#processor.checkNull(sset)
