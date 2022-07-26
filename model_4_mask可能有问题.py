import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import math
from category_id_map import CATEGORY_ID_LIST

class CrossAtt(nn.Module):
    """交叉注意力,num_hiddens应该怎么设"""

    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads, dropout, bias = False, **kwargs):
        super(CrossAtt, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.num_hiddens = num_hiddens
        self.bias = bias
        self.W_k = nn.Linear(key_size, num_hiddens, bias = bias) # 768 768
        self.W_q = nn.Linear(query_size, num_hiddens, bias = bias) # 768 768
        self.W_v = nn.Linear(value_size, num_hiddens, bias = bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias = bias)
        self.drop_out = nn.Dropout(dropout)

        # feedforward network
        self.dense1 = nn.Linear(768, 768)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(768, 768)

        # add&norm
        self.ln = nn.LayerNorm(768)

    def forward(self, queries, keys, values, mask_1, mask_2):
        queries = transpose_qkv(self.W_q(queries), self.num_heads) # ([16*3, 330, 768/3=256])
        keys = transpose_qkv(self.W_k(keys), self.num_heads) # ([16*3, 32, 256])
        values = transpose_qkv(self.W_v(values), self.num_heads) # ([16*3, 32, 768/3])

        valid_lens_1 = self.get_valid_len(mask_1) # 得到有效长度 ([16])
        valid_lens_2 = self.get_valid_len(mask_2) # 得到有效长度        
        if valid_lens_1 is not None:
        # 在轴0，将第⼀项（标量或者⽮量）复制num_heads次，
        # 然后如此复制第⼆项，然后诸如此类
            valid_lens_1 = torch.repeat_interleave(valid_lens_1, repeats=self.num_heads, dim=0) # ([48])，每一个元素重复3次
        if valid_lens_2 is not None:
            valid_lens_2 = torch.repeat_interleave(valid_lens_2, repeats=self.num_heads, dim=0) # ([48])，每一个元素重复3次
        output = self.DotProductAttention(queries, keys, values, valid_lens_1, valid_lens_2) # ([48, 330, 256])
        output_concat = transpose_output(output, self.num_heads) # ([16, 330, 768])
        Multi_att_output = self.W_o(output_concat)
        #至此完成multi-head attention的输出，输出维度([16, 330, 768])

        # feed forward network
        ffn_output = self.dense2(self.relu(self.dense1(Multi_att_output)))
        # add&norm
        return self.ln(self.drop_out(ffn_output) + Multi_att_output)


    def DotProductAttention(self, queries, keys, values, valid_lens_1, valid_lens_2):
        d = queries.shape[-1] # 768
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = self.masked_softmax(scores, valid_lens_1, valid_lens_2)
        return torch.bmm(self.drop_out(self.attention_weights), values)

    def masked_softmax(self, X, valid_lens_1, valid_lens_2): #input : X([48, 330, 32])
        """通过在最后⼀个轴上掩蔽元素来执⾏softmax操作"""
        # X:3D张量， valid_lens:1D或2D张量
        if valid_lens_1 is None and valid_lens_2 is None:
            return nn.functional.softmax(X, dim=-1)
        else:
            shape_1 = X.shape # 48 330 32
            valid_lens_1 = torch.repeat_interleave(valid_lens_1, shape_1[1]) # ([48]) -> ([15840]) 现在每一个元素重复990次
                # 最后⼀轴上被掩蔽的元素使⽤⼀个⾮常⼤的负值替换，从⽽其softmax输出为0
                # valid_lens_1中的元素大小是视频的有效长度，但重复的次数由文本决定
            # frame_mask
            X = self.sequence_mask(X.reshape(-1, shape_1[-1]), valid_lens_1, value=-1e6) # ([15840, 32])
            X = X.reshape(shape_1) # ([48 330 32])
            # title_mask
            X = X.permute(0, 2, 1) # 48 32 330
            shape_2 = X.shape #
            valid_lens_2 = torch.repeat_interleave(valid_lens_2, shape_2[1]) # ([1536])
            X = self.sequence_mask(X.reshape(-1, shape_2[-1]), valid_lens_2, value=-1e6) # 1536 330
            X = X.reshape(shape_2) # 48 32 330
            X = X.permute(0, 2, 1) # 48 330 32
        return nn.functional.softmax(X, dim=-1)
    
    def sequence_mask(self, X, valid_len, value=0): # input : X([15840, 32])
        """在序列中屏蔽不相关的项, X的维度不改变"""
        maxlen = X.size(1) # dim=1维度的大小,即32
        mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None] # ([15840, 32])
        X[~mask] = value

        return X

    def get_valid_len(self, X):
        """取得有效长度,输出一维tensor"""
        return torch.count_nonzero(X, dim = 1)
    
def transpose_qkv(X, num_heads): # input -> ([16, 32, 768<num_hiddens>])
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2 ,1, 3) # ([16, 3, 32, 256])
    return X.reshape(-1, X.shape[2], X.shape[3]) # 
    
def transpose_output(X, num_heads):
    """逆转transpose_qkv"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2]) # 16 3 330 256
    X = X.permute(0, 2, 1, 3) # 16 3 330 256
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache) # 创建tokenizer
        # cache_dir存放着预训练模型配置
        self.bertembeddings = self.bert.embeddings
        self.video_fc = nn.Linear(768, 768) # 几个意思?
        self.relu = nn.ReLU()

        self.crossAttention_visual = CrossAtt(768, 768, 768, num_hiddens = 768, num_heads = 3, dropout = 0.3)
        self.crossAttention_textual = CrossAtt(768, 768, 768, 768, 3, dropout = 0.3)

        self.classifier = nn.Linear(768, len(CATEGORY_ID_LIST))


    def forward(self, inputs, inference=False):

        text_embedding = self.bertembeddings(input_ids=inputs["title_input"]) # 这里应该是得到了embedding
        vision_embedding = self.relu(self.video_fc(inputs["frame_input"]))
        vision_embedding = self.bertembeddings(inputs_embeds=vision_embedding)

        text_random_attention_mask = inputs["title_mask"][:, None, None, :] # 
        text_random_attention_mask = (1.0 - text_random_attention_mask) * -10000.0

        text_bert_output = self.bert.encoder(text_embedding, attention_mask=text_random_attention_mask)["last_hidden_state"] # sequence_output ([16, 362, 768])
        #即将进入cross attention的是vision_embedding和text_bert_output

        #开始cross attention
        cro_visual_emb = self.crossAttention_visual(text_bert_output, vision_embedding, vision_embedding, inputs["frame_mask"], inputs["title_mask"]) # ([16, 330, 768])
        cro_textual_emb = self.crossAttention_textual(vision_embedding, text_bert_output, text_bert_output, inputs["title_mask"], inputs["frame_mask"]) # [(16, 32, 768)]
        
        sequence_output_crossAtt = torch.cat((cro_visual_emb, cro_textual_emb), dim = 1)
        # sequence_output_noAtt = torch.cat((vision_embedding, text_bert_output))
        combine_attention_mask = torch.cat((inputs["title_mask"], inputs["frame_mask"]), dim = 1)

        meanpooling = MeanPooling()
        final_embed_crossAtt = meanpooling(sequence_output_crossAtt, combine_attention_mask)
        # final_embed_noAtt = meanpooling(sequence_output_noAtt, combine_attention_mask)

        prediction_crossAtt = self.classifier(final_embed_crossAtt)
        # prediction_noAtt = self.classifier(final_embed_noAtt)

        if inference:
            return prediction_crossAtt
            # return torch.argmax(prediction, dim=1)
        else:
            # return self.cal_loss(0.4 * prediction_crossAtt + 0.6 * prediction_noAtt, inputs["label"])
            return self.cal_loss(prediction_crossAtt, inputs["label"])

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
