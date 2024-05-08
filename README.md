# Low Rank Substitution

This repo is the code for a little transformer LLM experiment I did in Dec 2023.

At some point I came across [this post](https://medium.com/@edandwe/a-guide-to-craft-your-own-custom-hugging-face-model-ba9cd555a646) and it got me thinking about a similar idea: What happens if we try replacing all or some of the Q / K / V and FFW matrices with lower-rank matrices and train from scratch? Kind of like LoRA, but instead of augmenting the existing matrices with low-rank adapters we replace them altogether?

## More info

You can read more (including the results I found) at my blog: https://chrissarmstrong.github.io/seeking-manifold/Experiments/Low-Rank-Substitution
