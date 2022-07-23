import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from category_id_map import CATEGORY_ID_LIST, lv2id_nums, category_id_to_lv1id, tuple_to_lv2id, lv2id_to_lv1id


class CrossAtt(nn.Module):
    """交叉注意力"""
    def __init__(self, query_size, key_size, value_size, num_hiddens, num_heads, dropout, bias = False, **kwargs):
        super(CrossAtt, self).__init__(**kwargs)
        num_heads = num_heads
        self.W_k = nn.Linear(key_size, num_hiddens, bias = bias)
        self.W_q = nn.Linear(query_size, num_hiddens, bias = bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias = bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias = bias)
        # self.drop_out = nn.Dropout(dropout)
   
    #加性注意力
    def AdditiveAttention(self, queries, keys, values):
        queries, keys = self.W_q(queries), self.W_k(keys) # 这是乘W后的
        # 维度扩展后,
        # queries的形状：(batch_size, , 1, num_hiddens)
        # keys的形状：(batch_size, 1, , num_hiddens)
        # 使用广播方式求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.W_v(features).squeeze(-1) # -1?
        self.attention_weights = nn.functional.softmax(scores, dim = -1) # -1?
        # 计算注意力分数(即attention层f(·))

    def forward(self, queries, keys, values):
        self.attention = self.AdditiveAttention(queries, keys, values)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(keys), self.num_heads)

        output = self.attention(queries, keys, values)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2 ,1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """逆转transpose_qkv"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.bertembeddings = self.bert.embeddings
        self.video_fc = nn.Linear(768, 768)
        self.relu = nn.ReLU()

        self.crossAttention_visual = CrossAtt(330, 32, 32, num_hiddens = 330, num_heads = 3, dropout = 0.3)
        self.crossAttention_textual = CrossAtt(32, 330, 330, 330, 3, 0.3)
        self.crossAttention_visual_mask = CrossAtt(330, 32, 32, 330, 3, 0.3)
        self.crossAttention_textual_mask = CrossAtt(32, 330, 330, 330, 3, 0.3)

        #self.classifier = nn.Linear(768, len(CATEGORY_ID_LIST))
        self.classifier = nn.Linear(768, 23)

    def forward(self, inputs, inference=False):
        # batch_size = 16
        text_embedding = self.bertembeddings(input_ids=inputs["title_input"]) # text_embedding([16, 330, 768])
        text_embedding = text_embedding.permute(0, 2, 1) # text_embedding([16, 768, 330])

        cls_mask = inputs["title_mask"][:, 0:1] # [16, 1] 
        text_mask = inputs["title_mask"][:, 1:] # [16, 329]
        # inputs["frame_mask"]:([16, 32])

        vision_embedding = self.relu(self.video_fc(inputs["frame_input"]))
        vision_embedding = self.bertembeddings(inputs_embeds=vision_embedding) # vision_embedding[16, 32, 768]
        vision_embedding = vision_embedding.permute(0, 2, 1) # vision_embedding[16, 678, 32]

        cro_visual_emb = self.crossAttention_visual(text_embedding, vision_embedding, vision_embedding)
        cro_textual_emb = self.crossAttention_textual(vision_embedding, text_embedding, text_embedding)

        cro_visual_emb_mask = self.crossAttention_visual_mask(inputs["title_mask"], inputs["frame_mask"], inputs["frame_mask"])
        cro_textual_emb_mask = self.crossAttention_textual_mask(inputs["frame_mask"], inputs["title_mask"], inputs["title_mask"]).permute(0, 2, 1)

        combine_embedding = torch.cat(
            [cro_visual_emb, cro_textual_emb], dim=2
        ).permute(0, 2, 1)
        combine_attention_mask = torch.cat(
            [cro_visual_emb_mask, cro_textual_emb_mask], dim=2
        ).permute(0, 2, 1)


        combine_random_attention_mask = combine_attention_mask[:, None, None, :]
        combine_random_attention_mask = (1.0 - combine_random_attention_mask) * -10000.0

        sequence_output = self.bert.encoder(
            combine_embedding, attention_mask=combine_random_attention_mask
        )["last_hidden_state"]
        #text_embedding = self.bert(inputs['title_input'], inputs['title_mask'])['last_hidden_state']
        



        
        meanpooling = MeanPooling()
        final_embed = meanpooling(sequence_output, combine_attention_mask)
        prediction = self.classifier(final_embed)
        
        

        if inference:
            return prediction
            # return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs["label"])

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        
        lv1_labels = torch.LongTensor([lv2id_to_lv1id(int(lv2id)) for lv2id in label]).cuda()

        loss = F.cross_entropy(prediction, lv1_labels)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (lv1_labels == pred_label_id).float().sum() / lv1_labels.shape[0]
        return loss, accuracy, pred_label_id, lv1_labels


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
