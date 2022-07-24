    def AdditiveAttention(self, queries, keys, values, valid_lens):
        """加性注意力""" 
        W_k_addi = nn.Linear(self.key_size / 3, self.num_hiddens, bias = self.bias)
        W_q_addi = nn.Linear(query_size, num_hiddens, bias = bias)
        W_v_addi = nn.Linear(value_size, num_hiddens, bias = bias) # 这里要重新看，后面的线性层都要改
        queries, keys = self.W_q(queries), self.W_k(keys) # 这是乘W后的
        # 维度扩展后,
        # queries的形状：(batch_size, query_size, 1, num_hiddens)
        # keys的形状：(batch_size, 1, key_size, num_hiddens)
        # 使用广播方式求和 features : (batch_size, query_size, key_size, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.W_v(features).squeeze(-1) # -1?
        self.attention_weights = self.masked_softmax(scores, valid_lens)
        # self.attention_weights = nn.functional.softmax(scores, dim = -1) # -1?
        # 计算注意力分数(即attention层f(·))
        return torch.bmm(self.dropout(self.attention_weights), values)
