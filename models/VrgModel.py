#obj+rela+attr
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .CaptionModel import CaptionModel
from misc.utils import expand_feats


def sort_pack_padded_sequence(input, lengths):
    #print('1')#4
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    #print('2')#5
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    #print('3')#3
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)


class GNN(nn.Module):
    #print('4')
    def __init__(self, opt):
        #print('5')
        super(GNN, self).__init__()
        self.opt = opt
        in_dim = opt.rnn_size
        out_dim = opt.rnn_size

        if self.opt.rela_gnn_type==0:
            in_rela_dim = in_dim*3
        elif self.opt.rela_gnn_type==1:
            in_rela_dim = in_dim*2
        else:
            raise NotImplementedError()
        self.att_vecs = nn.Linear(in_dim*2, out_dim)
        self.gnn_attr = nn.Sequential(nn.Linear(in_dim*2, out_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(opt.drop_prob_lm))
        self.gnn_rela = nn.Sequential(nn.Linear(in_rela_dim, out_dim),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(opt.drop_prob_lm))

    def forward(self, obj_vecs, attr_vecs, rela_vecs, edges, rela_masks=None):
        #print('6')#9
        # for easily indexing the subject and object of each relation in the tensors
        #print('**********obj_vecs', obj_vecs.size())#obj_vecs (64, 61, 1000)
        #print('**********obj_vecs', obj_vecs)
        #print('**********rela_vecs', rela_vecs.size())#rela_vecs (64, 53, 1000)
        #print('**********rela_vecs', rela_vecs)
        #print('**********edges', edges.size())#edges (64, 53, 2)
        #print('**********edges', edges)
        #print('**********attr_vecs', attr_vecs.size())#attr_vecs (16, 54, 2000)
        attr_vecs = self.att_vecs(attr_vecs)#
        #print('**********attr_vecs', attr_vecs.size())#attr_vecs (16, 54, 1000)
        obj_vecs, attr_vecs, rela_vecs, edges, ori_shape = self.feat_3d_to_2d(obj_vecs, attr_vecs, rela_vecs, edges)
        #print('**********obj_vecs', obj_vecs.size())#obj_vecs (3904, 1000)
        #print('**********obj_vecs', obj_vecs)
        #print('**********rela_vecs', rela_vecs.size())#rela_vecs (3392, 1000)
        #print('**********rela_vecs', rela_vecs)
        #print('**********edges', edges.size())#edges (3392, 2)
        #print('**********edges', edges)
        #print('**********ori_shape', ori_shape)#ori_shape (64, 61)

        # obj
        new_obj_vecs = obj_vecs

        #attr
        #print('*******obj_vecs', obj_vecs.size())#obj_vecs (864, 1000)
        #print('*******attr_vecs', attr_vecs.size())#attr_vecs (864, 1000)

        new_attr_vecs = self.gnn_attr(torch.cat([obj_vecs, attr_vecs], dim=-1)) + attr_vecs
        #print('*******new_attr_vecs', new_attr_vecs.size())#new_attr_vecs (864, 2000)
        # rela
        # get node features for each triplet <subject, relation, object>
        s_idx = edges[:, 0].contiguous() # index of subject
        #print('**********s_idx', s_idx.size())#s_idx (3392,)
        #print('**********s_idx', s_idx)#yi hang []
        o_idx = edges[:, 1].contiguous() # index of object
        #print('**********o_idx', o_idx.size())#o_idx (3392,)
        #print('**********o_idx', o_idx)#yi hang []
        s_vecs = obj_vecs[s_idx]
        #print('**********s_vecs', s_vecs.size())#o_idx (3392, 1000)
        #print('**********s_vecs', s_vecs)
        o_vecs = obj_vecs[o_idx]
        #print('**********o_vecs', o_vecs.size())#o_vecs (3392, 1000)
        #print('**********o_vecs', o_vecs)

        if self.opt.rela_gnn_type == 0:
            t_vecs = torch.cat([s_vecs, rela_vecs, o_vecs], dim=1)
        elif self.opt.rela_gnn_type == 1:
            t_vecs = torch.cat([s_vecs + o_vecs, rela_vecs], dim=1)
        else:
            raise NotImplementedError()
        #print('**********t_vecs', t_vecs.size())#t_vecs (3392, 3000)
        #print('**********t_vecs', t_vecs)
        new_rela_vecs = self.gnn_rela(t_vecs)+rela_vecs
        #print('**********new_rela_vecs', new_rela_vecs.size())#new_rela_vecs (3392, 1000)
        #print('**********new_rela_vecs', new_rela_vecs)
        new_obj_vecs, new_attr_vecs, new_rela_vecs = self.feat_2d_to_3d(new_obj_vecs, new_attr_vecs, new_rela_vecs, rela_masks, ori_shape)
        #print('**********new_obj_vecs', new_obj_vecs.size())#new_obj_vecs (64, 61, 1000)
        #print('**********new_obj_vecs', new_obj_vecs)
        #print('**********new_rela_vecs', new_rela_vecs.size())#new_rela_vecs (64, 53, 1000)
        #print('**********new_rela_vecs', new_rela_vecs)
        return new_obj_vecs, new_attr_vecs, new_rela_vecs


    # def feat_3d_to_2d(self, obj_vecs, attr_vecs, rela_vecs, edges):
    def feat_3d_to_2d(self, obj_vecs, attr_vecs, rela_vecs, edges):
        #print('7')#10
        """
        convert 3d features of shape (B, N, d) into 2d features of shape (B*N, d)
        """
        B, No = obj_vecs.shape[:2]
        obj_vecs = obj_vecs.view(-1, obj_vecs.size(-1))
        #print('*******attr_vecs', attr_vecs.size())#attr_vecs (16, 54, 2000)
        attr_vecs = attr_vecs.view(-1, attr_vecs.size(-1))
        #print('*******attr_vecs', attr_vecs.size())#attr_vecs (864, 2000)
        rela_vecs = rela_vecs.view(-1, rela_vecs.size(-1))

        obj_offsets = edges.new_tensor(range(0, B * No, No))
        edges = edges + obj_offsets.view(-1, 1, 1)
        edges = edges.view(-1, edges.size(-1))
        return obj_vecs, attr_vecs, rela_vecs, edges, (B, No)


    # def feat_2d_to_3d(self, obj_vecs, attr_vecs, rela_vecs, rela_masks, ori_shape):
    def feat_2d_to_3d(self, obj_vecs, attr_vecs, rela_vecs, rela_masks, ori_shape):
        #print('8')#11
        """
        convert 2d features of shape (B*N, d) back into 3d features of shape (B, N, d)
        """
        B, No = ori_shape
        obj_vecs = obj_vecs.view(B, No, -1)
        attr_vecs = attr_vecs.view(B, No, -1)
        rela_vecs = rela_vecs.view(B, -1, rela_vecs.size(-1)) * rela_masks
        return obj_vecs, attr_vecs, rela_vecs


def build_embeding_layer(vocab_size, dim, drop_prob):
    #print('9')
    embed = nn.Sequential(nn.Embedding(vocab_size, dim),
                          nn.ReLU(),
                          nn.Dropout(drop_prob))
    return embed


class AttModel(CaptionModel):
    #print('10')
    def __init__(self, opt):
        #print('11')
        super(AttModel, self).__init__()
        self.opt = opt
        # self.geometry_relation = opt.geometry_relation
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_hid_size = opt.att_hid_size
        self.seq_per_img = opt.seq_per_img
        self.use_bn = getattr(opt, 'use_bn', 0)
        self.att_feat_size = opt.att_feat_size
        if opt.use_box:
            self.att_feat_size = self.att_feat_size + 5  # concat box position features
        # self.sg_label_embed_size = opt.sg_label_embed_size
        self.ss_prob = 0.0 # Schedule sampling probability

        self.embed = build_embeding_layer(self.vocab_size + 1, self.input_encoding_size, self.drop_prob_lm)
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))

        # lazily use the same vocabulary size for obj, attr and rela embeddings
        # num_objs = num_relas = 472
        # self.obj_embed = build_embeding_layer(num_objs, self.sg_label_embed_size, self.drop_prob_lm)
        # if not self.geometry_relation:
        #     self.rela_embed = build_embeding_layer(num_relas, self.sg_label_embed_size, self.drop_prob_lm)

        self.proj_obj = nn.Sequential(*[nn.Linear(self.rnn_size + self.rnn_size * 1,
                                                  self.rnn_size), nn.ReLU(), nn.Dropout(0.5)])
        self.proj_attr = nn.Sequential(*[nn.Linear(self.rnn_size,
                                                  self.rnn_size), nn.ReLU(), nn.Dropout(0.5)])
        self.proj_rela = nn.Sequential(*[nn.Linear(self.rnn_size,
                                                   self.rnn_size), nn.ReLU(), nn.Dropout(0.5)])
        self.gnn = GNN(opt)

        self.ctx2att_obj = nn.Linear(self.rnn_size, self.att_hid_size)
        self.ctx2att_attr = nn.Linear(self.rnn_size, self.att_hid_size)
        self.ctx2att_rela = nn.Linear(self.rnn_size, self.att_hid_size)

        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.logit_tag = nn.Linear(self.rnn_size, 3)
        self.init_weights()

        self.fuse_obj = nn.Linear(self.rnn_size, self.rnn_size, bias = False)
        self.fuse_att = nn.Linear(self.rnn_size, self.rnn_size, bias = False)
        self.fuse_att_obj = nn.Linear(self.rnn_size, self.rnn_size, bias = True)
        self.fuse_att_obj1 = nn.Linear(self.rnn_size, 1, bias = True)

    def init_weights(self):
        #print('12')
        initrange = 0.1
        self.embed[0].weight.data.uniform_(-initrange, initrange)
        #self.attr_embed[0].weight.data.uniform_(-initrange, initrange)
        # self.obj_embed[0].weight.data.uniform_(-initrange, initrange)
        # if not self.geometry_relation:
        #     self.rela_embed[0].weight.data.uniform_(-initrange, initrange)


    def init_hidden(self, bsz, weak_rela):
        #print('13')
        assert bsz == weak_rela.size(0), 'make sure of same batch size'
        rela_mask = weak_rela > 0

        weak_rela_embedding = torch.sum(self.embed(weak_rela) * rela_mask.unsqueeze(-1).float(), dim=1) / \
                              (torch.sum(rela_mask, dim=1, keepdim=True).float() + 1e-20)
        h = torch.stack([weak_rela_embedding for _ in range(self.num_layers)], 0)
        return (h, h)


    def _embed_vrg(self, obj_labels, attr_labels, rela_labels):
        #print('14')#7
        # obj_embed = self.obj_embed(obj_labels)
        obj_embed = self.embed(obj_labels)
        attr_embed = self.embed(attr_labels)
        # if self.geometry_relation:
        #     rela_embed = rela_labels
        # else:
        # rela_embed = self.rela_embed(rela_labels)
        rela_embed = self.embed(rela_labels)

        return obj_embed, attr_embed, rela_embed

    def _proj_vrg(self, obj_embed, attr_embed, rela_embed, att_feats):
        "project node features in paper"
        #print('15')#8
        #print('*******obj_embed', obj_embed.size())#obj_embed (16, 54, 1, 1000)
        #print('*******att_feats', att_feats.size())#att_feats (16, 54, 1000)
        obj_embed = obj_embed.view(obj_embed.size(0), obj_embed.size(1), -1)
        #print('*******obj_embed', obj_embed.size())#obj_embed (16, 54, 1000)
        obj_vecs = self.proj_obj(torch.cat([att_feats, obj_embed], dim=-1))+att_feats
        #print('*******attr_embed', attr_embed.size())#attr_embed (16, 54, 2, 1000) 2 or 3 is attr number
        attr_embed = attr_embed.view(attr_embed.size(0), attr_embed.size(1), -1)
        #print('*******attr_embed', attr_embed.size())#attr_embed (16, 54, 1000)
        attr_vecs = self.proj_obj(attr_embed)

        rela_vecs = self.proj_rela(rela_embed)
        return obj_embed, attr_embed, rela_embed
        """
	obj_embed = obj_embed.view(obj_embed.size(0), obj_embed.size(1), -1)
	fuse_obj = self.fuse_obj(obj_embed).view(obj_embed.size(0) * obj_embed.size(1), 1, self.rnn_size).repeat(1, obj_embed.size(1), 1)
        fuse_att = self.fuse_att(att_feats).view(att_feats.size(0), att_feats.size(1), self.rnn_size).repeat(att_feats.size(1), 1, 1)
	fuse_obj_att = torch.mul(fuse_obj, fuse_att)
	fuse_obj_att = F.relu(fuse_obj_att)
	fuse_obj_att = self.fuse_att_obj(fuse_obj_att)
	fuse_obj_att = self.fuse_att_obj1(fuse_obj_att).view(obj_embed.size(0), obj_embed.size(1), obj_embed.size(1))
	fuse_obj_att = F.softmax(fuse_obj_att, dim=-1)
        
	#obj_att = self.proj_obj(torch.cat([att_feats, obj_embed], dim=-1))
	#fuse_obj_att_obj_feats = torch.matmul(fuse_obj_att, obj_att)
        #obj_vecs = fuse_obj_att_obj_feats + att_feats

        #fuse_obj_att_att_feats = torch.matmul(fuse_obj_att, att_feats)
        #obj_vecs = fuse_obj_att_att_feats + att_feats
	
        fuse_obj_att_att_feats = torch.matmul(fuse_obj_att, obj_embed)
        obj_vecs = fuse_obj_att_att_feats + att_feats
        #obj_vecs = obj_att + att_feats

	rela_vecs = self.proj_rela(rela_embed)
        return obj_vecs,  rela_vecs
        """
    def _prepare_vrg_features(self, vrg_data, att_feats, att_masks):
        #print('16')#6
        """
        the raw data the are needed:
            - obj_labels: (B, No, ?)
            - rela_labels: (B, Nr, ?)
            - rela_triplets: (subj_index, obj_index, rela_label) of shape (B, Nr, 3)
            - rela_edges: LongTensor of shape (B, Nr, 2), where rela_edges[b, k] = [i, j] indicates the
                        presence of the relation triple: ( obj[b][i], rela[b][k], obj[b][j] ),
                        i.e. the k-th relation of the b-th sample which is between the i-th and j-th objects
        """
        obj_labels = vrg_data['obj_labels']
        attr_labels = vrg_data['attr_labels']
        #print('**********attr_labels', attr_labels.size())#attr_labels (16, 54, 2)
        rela_masks = vrg_data['rela_masks']
        rela_edges, rela_labels = vrg_data['rela_edges'], vrg_data['rela_feats']

        att_masks, rela_masks = att_masks.unsqueeze(-1), rela_masks.unsqueeze(-1)
        # node features
        obj_embed, attr_embed, rela_embed = self._embed_vrg(obj_labels, attr_labels, rela_labels)
        #print('**********attr_labels', attr_labels.size())#attr_labels (16, 54, 2)
        obj_vecs, attr_vecs, rela_vecs = self._proj_vrg(obj_embed, attr_embed, rela_embed, att_feats)
        #print('**********attr_vecs', attr_labels.size())#attr_vecs (16, 54, 2)
        # node embedding with simple gnns
        obj_vecs, attr_vecs, rela_vecs = self.gnn(obj_vecs, attr_vecs, rela_vecs, rela_edges, rela_masks)
        #print('**********prepare_attr_vecs', attr_labels.size())#attr_vecs (16, 54, 2)
        return obj_vecs, attr_vecs, rela_vecs


    def prepare_core_args(self, sg_data, fc_feats, att_feats, att_masks):
        #print('17')#2
        rela_masks = sg_data['rela_masks']
        # embed fc and att features
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        obj_feats, attr_feats, rela_feats = self._prepare_vrg_features(sg_data, att_feats, att_masks)

        p_obj_feats = self.ctx2att_obj(obj_feats)
        #print('attr_feats', attr_feats.size())#attr_feats (16, 54, 2000)
        p_attr_feats = self.ctx2att_attr(attr_feats)
        p_rela_feats = self.ctx2att_rela(rela_feats)

        core_args = [fc_feats, att_feats, obj_feats, attr_feats, rela_feats, \
                     p_obj_feats, p_attr_feats,  p_rela_feats, \
                     att_masks, rela_masks]
        return core_args


    def _forward(self, sg_data, fc_feats, att_feats, seq, weak_rela, att_masks=None):
        #print('18')#1
        core_args = self.prepare_core_args(sg_data, fc_feats, att_feats, att_masks)
        # make seq_per_img copies of the encoded inputs:  shape: (B, ...) => (B*seq_per_image, ...)
        core_args = expand_feats(core_args, self.seq_per_img)
        weak_rela = expand_feats([weak_rela], self.seq_per_img)[0]

        batch_size = fc_feats.size(0) * self.seq_per_img
        state = self.init_hidden(batch_size, weak_rela)
        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1)
        outputs_tag = fc_feats.new_zeros(batch_size, seq.size(1) - 1, 3)
        # teacher forcing
        for i in range(seq.size(1) - 1):
            # scheduled sampling
            if self.training and i >= 1 and self.ss_prob > 0.0:
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            # output, state = self.get_logprobs_state(it, state, core_args)
            output, output_tag, state = self.get_logprobs_state(it, state, core_args)
            outputs[:, i] = output
            outputs_tag[:, i] = output_tag

        return outputs, outputs_tag

    def get_logprobs_state(self, it, state, core_args):
        #print('19')#13 17
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, state, core_args)
        logprobs = F.log_softmax(self.logit(output), dim=1)
        logprobs_tag = F.log_softmax(self.logit_tag(output), dim=1)

        # return logprobs, state
        return logprobs, logprobs_tag, state

    # sample sentences with greedy decoding
    def _sample(self, sg_data, fc_feats, att_feats, weak_rela, att_masks=None, opt={}, _core_args=None):
        #print('20')
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        return_core_args = opt.get('return_core_args', False)
        expand_features = opt.get('expand_features', True)

        if beam_size > 1:
            return self._sample_beam(sg_data, fc_feats, att_feats, weak_rela, att_masks, opt)
        if _core_args is not None:
            # reuse the core_args calculated during generating sampled captions
            # when generating greedy captions for SCST,
            core_args = _core_args
        else:
            core_args = self.prepare_core_args(sg_data, fc_feats, att_feats, att_masks)

        # make seq_per_img copies of the encoded inputs:  shape: (B, ...) => (B*seq_per_image, ...)
        # should be True when training (xe or scst), False when evaluation
        if expand_features:
            if return_core_args:
                _core_args = core_args
            core_args = expand_feats(core_args, self.seq_per_img)
            weak_rela = expand_feats([weak_rela], self.seq_per_img)[0]
            batch_size = fc_feats.size(0)*self.opt.seq_per_img
        else:
            batch_size = fc_feats.size(0)

        state = self.init_hidden(batch_size, weak_rela)
        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)

            logprobs, _, state = self.get_logprobs_state(it, state, core_args)
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp
            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        returns = [seq, seqLogprobs]
        if return_core_args:
            returns.append(_core_args)
        return returns


    # sample sentences with beam search
    def _sample_beam(self, sg_data, fc_feats, att_feats, weak_relas, att_masks=None, opt={}):
        #print('21')
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        core_args = self.prepare_core_args(sg_data, fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            weak_rela = expand_feats([weak_relas[k: k + 1]], beam_size)[0]
            state = self.init_hidden(beam_size, weak_rela)
            sample_core_args = []
            for item in core_args:
                if type(item) is list or item is None:
                    sample_core_args.append(item)
                    continue
                else:
                    sample_core_args.append(item[k:k+1])
            sample_core_args = expand_feats(sample_core_args, beam_size)

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, _, state = self.get_logprobs_state(it, state, sample_core_args)

            self.done_beams[k] = self.beam_search(state, logprobs, sample_core_args, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

class AdaAtt_attention(nn.Module):
    def __init__(self, opt):
        super(AdaAtt_attention, self).__init__()
        self.input_encoding_size = opt.input_encoding_size#1000
        self.rnn_size = opt.rnn_size#1000
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_hid_size = opt.att_hid_size#512

        # fake region embed
        self.fr_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.input_encoding_size),
            nn.ReLU(), 
            nn.Dropout(self.drop_prob_lm))#1000 1000
        self.fr_embed = nn.Linear(self.input_encoding_size, self.input_encoding_size)
                                             #1000 512
        # h out embed
        self.ho_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.input_encoding_size),
            nn.Tanh(), 
            nn.Dropout(self.drop_prob_lm))#1000 1000
        self.ho_embed = nn.Linear(self.input_encoding_size, self.input_encoding_size)
                                    #1000 1000
        self.alpha_net = nn.Linear(self.rnn_size, 1)#512
        self.att2h = nn.Linear(self.rnn_size, self.rnn_size)
#2021.10.11
        self.conv_feat_embed = nn.Linear(512, self.rnn_size)

    def forward(self, h_out, fake_region, conv_feat, conv_feat_embed):

        # View into three dimensions
        
        att_size = conv_feat.numel() // conv_feat.size(0) // self.rnn_size
        conv_feat = conv_feat.view(-1, att_size, self.rnn_size)
        #print(conv_feat.size())
        conv_feat_embed = conv_feat_embed.view(-1, att_size, self.att_hid_size)
        #print(conv_feat_embed.size())
#2021.10.11
        conv_feat_embed = self.conv_feat_embed(conv_feat_embed)
        #print(conv_feat_embed.size())
        # view neighbor from bach_size * neighbor_num x rnn_size to bach_size x rnn_size * neighbor_num
        fake_region = self.fr_linear(fake_region)#st
        fake_region_embed = self.fr_embed(fake_region)#st

        h_out_linear = self.ho_linear(h_out)#ht
        h_out_embed = self.ho_embed(h_out_linear)#ht

        txt_replicate = h_out_embed.unsqueeze(1).expand(h_out_embed.size(0), att_size + 1, h_out_embed.size(1))#ht
        #print(fake_region.size())#(80, 1000)
        #print(self.input_encoding_size)#1000
        #print(conv_feat.size())#(80, 54, 1000)
        img_all = torch.cat([fake_region.view(-1,1,self.input_encoding_size), conv_feat], 1)
        #st and V 
        #print(fake_region_embed.size())#(80, 1000)
        #print(self.input_encoding_size)#1000
        #print(conv_feat_embed.size())#(80, 54, 512)

        #hA_zt = F.tanh(img_all + txt_replicate)#st(st+V) + ht
        #print('hA_zt', hA_zt.size())
        #hA_zt = F.dropout(hA_zt, self.drop_prob_lm, self.training)
        #print('hA_zt', hA_zt.size())

        img_all_embed = torch.cat([fake_region_embed.view(-1,1,self.rnn_size), conv_feat_embed], 1)#st and p_V
        
        hA = F.tanh(img_all_embed + txt_replicate)#st(st+p_V) + ht
        #print('hA', hA.size())#hA (80, 55, 1000)
        hA = F.dropout(hA,self.drop_prob_lm, self.training)#zt_extend
        #print('hA', hA.size())#(80, 55, 1000)

        #alpha_zt_concat = torch.cat([hA, hA_zt], dim=1)
        #print('alpha_zt_concat', alpha_zt_concat.size())

        hAflat = self.alpha_net(hA.view(-1, self.rnn_size))#512
        #print("hAflat", hAflat.size())#hAflat (4400, 1)
        PI = F.softmax(hAflat.view(-1, att_size + 1), dim=1) #c_alpha_t           ### (batch_size*seq_per_img*1)*(att_size + 1)  
        #print("PI", PI.size())#PI (80, 55)

        visAtt = torch.bmm(PI.unsqueeze(1), img_all).squeeze(1)#ct
        #print("visAtt", visAtt.size())#visAtt (80, 1000)

        beta_t = PI[:, -1:]
        #print("beta_t", beta_t.size())#beta_t (80, 1)
        c_t_hat = beta_t * fake_region + (1-beta_t) * visAtt
        #print("c_t_hat", c_t_hat.size())#c_t_hat (80, 1000)

        c_t_hat = c_t_hat.squeeze(1) 
        #print("c_t_hat", c_t_hat.size())
        #print("visAttdim", visAttdim.size())#visAttdim (80, 1000)

        return c_t_hat

class VRGCore(nn.Module):
    #print('22')
    def __init__(self, opt, use_maxout=False):
        #print('23')
        super(VRGCore, self).__init__()
        self.opt = opt
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1

        lang_lstm_in_dim = opt.rnn_size*4
        self.lang_lstm = nn.LSTMCell(lang_lstm_in_dim, opt.rnn_size) # h^1_t, \hat v

        self.attention_obj = Attention(opt)
        self.attention_attr = Attention(opt)
        self.attention_rela = Attention(opt)

        self.xt = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.fc = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.h_att = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.h_att1 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.xt2 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.h_att2 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.h_lang = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.h_att3 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.h_lang1 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.h_att4 = nn.Linear(opt.rnn_size, opt.rnn_size)

        self.fuse1 = nn.Linear(opt.rnn_size*2, opt.rnn_size)
        #self.fuse2 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.attention_fuse1 = Attention(opt)
        self.r_i2h = nn.Linear(opt.rnn_size*4, opt.rnn_size)
        self.r_h2h = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.adattention = AdaAtt_attention(opt)
        self.huu = nn.Linear(opt.rnn_size*3, opt.rnn_size)

    def forward(self, xt, state, core_args):
        #print('24')#14 18
        fc_feats, att_feats, obj_feats, attr_feats, rela_feats, \
        p_obj_feats, p_attr_feats, p_rela_feats, \
        att_masks, rela_masks = core_args
        prev_h = state[0][-1]
        #print('**********fc_feats_size:', fc_feats.size())#fc_feats_size: (320, 1000)
        #print('**********att_feats_size:', att_feats.size())#att_feats_size: (320, 61, 1000)
        #print('**********obj_feats_size:', obj_feats.size())#obj_feats_size: (320, 61, 1000)
        #print('**********rela_feats_size:', rela_feats.size())#rela_feats_size: (320, 53, 1000)

        #x = torch.sigmoid(self.xt(xt) + self.h_att(state[0][0]))
        #xt = torch.mul(x, xt)
        #f = torch.sigmoid(self.fc(fc_feats) + self.h_att1(state[0][0]))
        #fc_feats = torch.mul(f, fc_feats)

        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)#att_lstm_input_size: (80, 3000)
        #print('**********att_lstm_input_size:', att_lstm_input.size())
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))#h_att_size: (80, 1000)
        h_att = self.xt2(xt) + self.h_att2(h_att)
        #print('**********h_att_size:', h_att.size())
        lang_lstm_input = h_att

        att_obj = self.attention_obj(h_att, obj_feats, p_obj_feats, att_masks)#att_obj_size: (80, 1000)
        #print('**********att_obj_size:', att_obj.size())
        lang_lstm_input = torch.cat([lang_lstm_input, att_obj], 1)#lang_lstm_input_size: (80, 2000)

        att_attr = self.attention_attr(h_att, attr_feats, p_attr_feats, att_masks)
        lang_lstm_input = torch.cat([lang_lstm_input, att_attr], 1)

        #print('**********lang_lstm_input_att_obj_size:', lang_lstm_input.size())6+555
        att_rela = self.attention_rela(h_att, rela_feats, p_rela_feats, rela_masks)#att_rela_size: (80, 1000)
        #print('**********att_rela_size:', att_rela.size())
        lang_lstm_input = torch.cat([lang_lstm_input, att_rela], 1)#lang_lstm_input_size: (80, 3000)
        #print('**********lang_lstm_input_att_rela_size:', lang_lstm_input.size())
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        #h_lang = self.h_lang(h_lang) + self.h_att3(h_att)

#visual sentinel
        next_c = c_lang
        prev_h = state[0][1]
        tanh_mt = F.tanh(next_c)
        i2h = self.r_i2h(lang_lstm_input)#xt
        n5 = i2h + self.r_h2h(prev_h)
        st = F.sigmoid(n5) * tanh_mt
        visAttdim = self.adattention(h_lang, st, att_feats, p_obj_feats)
        output = self.huu(torch.cat((h_att, h_lang, visAttdim),1))#h_fuse1

        output = F.dropout(output, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))
        return output, state

class Attention(nn.Module):
    #print('25')
    def __init__(self, opt):
        #print('26')
        super(Attention, self).__init__()
        self.opt = opt
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.query_dim = self.rnn_size
        self.h2att = nn.Linear(self.query_dim, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)


    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        #print('27')#15 16
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = torch.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size

        weight = F.softmax(dot, dim=1)                       # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size
        return att_res

class VrgModel(AttModel):
    #print('28')
    def __init__(self, opt):
        #print('29')
        super(VrgModel, self).__init__(opt)
        self.num_layers = 2
        self.core = VRGCore(opt)

