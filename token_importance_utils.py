import torch

class AttentionBasedImportance:
    
    def __init__(self, inputs, tokenizer, attentions):
        
        if type(inputs["input_ids"]) == torch.Tensor:
            if inputs["input_ids"].device != torch.device(type='cpu'):
                self.input_ids = inputs["input_ids"].cpu()
            else:
                self.input_ids = inputs["input_ids"]
            self.input_ids = self.input_ids.numpy()
        self.tokenizer = tokenizer
        self.attentions = attentions
        self.layers_count = len(attentions)
        self.heads_count = attentions[0].shape[1]
        
    def get_attn_head_processor(self, layer, head, record_index, valid_tokens, tokens):
        
        attns = self.attentions[layer][record_index][head][:valid_tokens+1,:valid_tokens+1]
        
        """
        if type(attns) == torch.Tensor:
            if attns.device != torch.device(type='cpu'):
                attns = attns.cpu()
            attns = attns.numpy()
        """
        
        def attn_head_processor(sep_attn_upperbound=0.7):
            total_weight = attns.sum()
            sep_weight = attns[:,-1].sum() + attns[-1,:].sum() + attns[:,0].sum() + attns[0,:].sum()
            if (sep_weight > sep_attn_upperbound*total_weight):
                return None
            record_data = []
            for j in range(valid_tokens+1):
                token_attns = list(attns[j,:])
                record_data.append({"token": "%s "%tokens[j],
                                    "heat": list(map(lambda x: float(x), token_attns)),
                                   })
            return record_data
    
        return attn_head_processor
    
    def get_data_for_textual_heatmap(self, record_index, sep_attn_upperbound=0.7, spcific_layers_heads=[]):
        record = self.input_ids[record_index]
        tokens = self.tokenizer.convert_ids_to_tokens(record)
        valid_tokens = record.nonzero()[0].max()
        data = []
        tiles = []
        for i in range(self.layers_count):
            for h in range(self.heads_count):
                processor = self.get_attn_head_processor(i, h, record_index, valid_tokens, tokens)
                output = processor(sep_attn_upperbound)
                if output is not None:
                    data.append(output)
                    tiles.append("H-%s-%s"%(i, h))
        return data, tiles
        
