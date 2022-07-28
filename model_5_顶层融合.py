import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from config import parse_args
args = parse_args()
from category_id_map import CATEGORY_ID_LIST


class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache) # 创建tokenizer分词器
        # cache_dir存放着预训练模型配置
        self.bertembeddings = self.bert.embeddings
        self.video_fc = nn.Linear(768, 768) 
        self.relu = nn.ReLU()

        self.classifier = nn.Linear(768, len(CATEGORY_ID_LIST))


    def forward(self, inputs, inference=False):  
        if args.fix == False:
            text_embedding = self.bertembeddings(input_ids=inputs["title_input"]) # 这里应该是得到了embedding

            cls_embedding = text_embedding[:, 0:1, :]
            text_embedding = text_embedding[:, 1:, :]

            cls_mask = inputs["title_mask"][:, 0:1]
            text_mask = inputs["title_mask"][:, 1:]

            vision_embedding = self.relu(self.video_fc(inputs["frame_input"]))
            vision_embedding = self.bertembeddings(inputs_embeds=vision_embedding)

            combine_embedding = torch.cat(
                [cls_embedding, vision_embedding, text_embedding], dim=1
            )
            combine_attention_mask = torch.cat(
                [cls_mask, inputs["frame_mask"], text_mask], dim=1
            )
            combine_random_attention_mask = combine_attention_mask[:, None, None, :]
            combine_random_attention_mask = (1.0 - combine_random_attention_mask) * -10000.0

            sequence_output = self.bert.encoder(
                combine_embedding, attention_mask=combine_random_attention_mask
            )["last_hidden_state"]

            meanpooling = MeanPooling()
            final_embed = meanpooling(sequence_output, combine_attention_mask)

            prediction = self.classifier(final_embed)

            if inference:
                return prediction
                # return torch.argmax(prediction, dim=1)
            else:
                return self.cal_loss(prediction, inputs["label"])
            
        else:
            with torch.no_grad():
                text_embedding = self.bertembeddings(input_ids=inputs["title_input"]) # 这里应该是得到了embedding

                cls_embedding = text_embedding[:, 0:1, :]
                text_embedding = text_embedding[:, 1:, :]

                cls_mask = inputs["title_mask"][:, 0:1]
                text_mask = inputs["title_mask"][:, 1:]

            vision_embedding = self.relu(self.video_fc(inputs["frame_input"]))

            with torch.no_grad():
                vision_embedding = self.bertembeddings(inputs_embeds=vision_embedding)

                combine_embedding = torch.cat(
                    [cls_embedding, vision_embedding, text_embedding], dim=1
                )
                combine_attention_mask = torch.cat(
                    [cls_mask, inputs["frame_mask"], text_mask], dim=1
                )
                combine_random_attention_mask = combine_attention_mask[:, None, None, :]
                combine_random_attention_mask = (1.0 - combine_random_attention_mask) * -10000.0

                sequence_output = self.bert.encoder(
                    combine_embedding, attention_mask=combine_random_attention_mask
                )["last_hidden_state"]

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
