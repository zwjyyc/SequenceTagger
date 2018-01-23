
class Token(object):
    def __init__(self, id, t_str, h_id, rel, label):
	self.t_str = t_str
	self.id = id
	self.h_id = h_id
	self.rel = rel
	self.label = label
	
	self.u_ids = [h_id]
	self.u_rels = [rel]
	
	self.d_ids = []
	self.d_rels = []
	
    def add_d_id_rel(self, id, rel):
	self.d_ids.append(id)
	self.d_rels.append('r_' + rel)

    def add_u_id_rel(self, id, rel):
	self.u_ids.append(id)
	self.u_rels.append(rel)
	
    def token2str(self):
	return self.t_str + ' ' + str(self.id) + ' ' + str(self.h_id) + ' ' + self.rel  + ' ' + self.label + ' ### childs : ' + ' '.join([str(id) for id in self.d_ids]) + ' ### rels : ' + ' '.join(self.d_rels)

class Sentence(object):
    def __init__(self, tokens):
	self.tokens = tokens
	self.dic = {}
	self.get_dic()

    def get_dic(self):
	tokens = self.tokens
	
