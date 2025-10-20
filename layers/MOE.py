from torch import nn 

from mixture_of_experts import MoE


# NOTE : this is just an example implimentatoin of how we are going to impliment MOE in our code
# Directly import MoE from the liberary in the near future unless we are using something complex

moe = MoE(
    dim=512,
    num_experts=16,
    hidden_dim=2048,
    activation=nn.GELU,
    second_policy_train='random'
)
