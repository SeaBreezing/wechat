import torch
import torch.nn as nn
from config import parse_args
args = parse_args()


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        # 初始化影子权重为初始权重
        if args.fix == True: # 
            for name, param in self.model.named_parameters():
                if name not in args.param_list: #
                    param.requeires_grad = False #
                    if param.requires_grad:
                        self.shadow[name] = param.data.clone()
            else: # 
                if param.requires_grad: # 
                    self.shadow[name] = param.data.clone() 

                    

    def update(self):
        if args.fix == True: # 
            for name, param in self.model.named_parameters():
                if name not in args.param_list: #
                    param.requeires_grad = False #
                    if param.requires_grad: 
                        assert name in self.shadow
                        new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                        self.shadow[name] = new_average.clone()
        else: # 
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        if args.fix == True: #
            for name, param in self.model.named_parameters():
                if name not in args.param_list: #
                    param.requeires_grad = False #
                    if param.requires_grad:
                        assert name in self.shadow
                        self.backup[name] = param.data
                        param.data = self.shadow[name]
        else: 
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    self.backup[name] = param.data
                    param.data = self.shadow[name]


    def restore(self):
        if args.fix == True: # 
            for name, param in self.model.named_parameters():
                if name not in args.param_list: #
                    param.requeires_grad = False #
                    if param.requires_grad:
                        assert name in self.backup
                        param.data = self.backup[name]
        else: # 
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.backup
                    param.data = self.backup[name]     
        self.backup = {}

        
class FGM:
    def __init__(self, model: nn.Module, eps=1.):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.backup = {}

    # only attack word embedding
    def attack(self, emb_name='word_embeddings'):
        # name_list = []
        # param_list = []
        if args.fix == True:
            for name, param in self.model.named_parameters():
                if name not in args.param_list: #
                    param.requeires_grad = False #
            # name_list.append(name)
            # param_list.append(param)
            # param_dict = dict(zip(name_list,param_list))
                    if param.requires_grad and emb_name in name:
                        self.backup[name] = param.data.clone()
                        norm = torch.norm(param.grad)
                        if norm and not torch.isnan(norm):
                            r_at = self.eps * param.grad / norm
                            param.data.add_(r_at)
        else:
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at) #"_"指inplace操作

        # return param_dict

    def restore(self, emb_name='word_embeddings'):
        if args.fix == True:
            if name not in args.param_list: #
                param.requeires_grad = False #
                for name, para in self.model.named_parameters():
                    if para.requires_grad and emb_name in name:
                        assert name in self.backup
                        para.data = self.backup[name]
        else:
            for name, para in self.model.named_parameters():
                if para.requires_grad and emb_name in name:
                    assert name in self.backup
                    para.data = self.backup[name]            

        self.backup = {}

