'''
This code leverages the Hugging Face GPTNeo model
'''

from transformers import GPTNeoConfig
from typing import List

class LowrankSubGPTNeoConfig(GPTNeoConfig):
    def __init__(self, num_layers=4,
                 attention_types=[[["global", "local"], 2]],
                 sa_ranks: List=[None, None, None, None],
                 mlp_ranks: List=[None, None, None, None],
                 **kwargs):

        super().__init__(num_layers=num_layers, attention_types=attention_types, **kwargs)
        
        assert len(sa_ranks) == num_layers, "The length of your list of self-attention ranks needs to equal the number of layers!"
        assert len(mlp_ranks) == num_layers, "The length of your list of MLP ranks needs to equal the number of layers!"

        self.sa_ranks = sa_ranks
        self.mlp_ranks = mlp_ranks
