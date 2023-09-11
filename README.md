# Echo Chamber Score
ECS (Echo Chamber Score) is a method to measure the echo chamber and polarization in social media.

**Paper:** Quantifying the Echo Chamber Effect: An Embedding Distance-based Approach (Accepted to ASONAM 2023 - [arXiv](https://arxiv.org/abs/2307.04668))

# Paper Abstract
The rise of social media platforms has facilitated the formation of echo chambers, which are online spaces where users predominantly encounter viewpoints that reinforce their existing beliefs while excluding dissenting perspectives. This phenomenon significantly hinders information dissemination across communities and fuels societal polarization. Therefore, it is crucial to develop methods for quantifying echo chambers. In this paper, we present the Echo Chamber Score (ECS), a novel metric that assesses the cohesion and separation of user communities by measuring distances between users in the embedding space. In contrast to existing approaches, ECS is able to function without labels for user ideologies and makes no assumptions about the structure of the interaction graph. To facilitate measuring distances between users, we propose EchoGAE, a self-supervised graph autoencoder-based user embedding model that leverages users' posts and the interaction graph to embed them in a manner that reflects their ideological similarity. To assess the effectiveness of ECS, we use a Twitter dataset consisting of four topics - two polarizing and two non-polarizing. Our results showcase ECS's effectiveness as a tool for quantifying echo chambers and shedding light on the dynamics of online discourse.



# Cite

```tex
@misc{Alatawi2023Quantifying,
  title = {Quantifying the Echo Chamber Effect: An Embedding Distance-Based Approach},
  author = {Alatawi, Faisal and Sheth, Paras and Liu, Huan},
  year = {2023},
  month = jul,
  number = {arXiv:2307.04668},
  eprint = {2307.04668},
  primaryclass = {cs},
  publisher = {arXiv},
  urldate = {2023-07-11},
  archiveprefix = {arxiv}
}
```
