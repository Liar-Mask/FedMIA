# FedMIA-Repository

### This is the official pytorch implementation of the paper:

  ** Evaluating Membership Inference Attacks and Defenses in Federated Learning **

## Abstract 

Membership Inference Attacks (MIAs) poses a growing threat to privacy preservation in federated learning. The semi-honest attacker, e.g., the server, may determine whether a particular sample belongs to a target client according to the observed model information. This paper conducts an evaluation of existing MIAs and corresponding defense strategies. Our evaluation on MIAs reveals two important findings about the trend of MIAs. Firstly, combining model information from multiple communication rounds (Multi-temporal) enhances the overall effectiveness of MIAs compared to utilizing model information from a single epoch.  Secondly, incorporating models from non-target clients (Multi-spatial) significantly improves the effectiveness of MIAs, particularly when the clients' data is homogeneous. This highlights the importance of considering the temporal and spatial model information in MIAs. Next, we assess the effectiveness via privacy-utility tradeoff for two type defense mechanisms against MIAs: Gradient Perturbation and Data Replacement. Our results demonstrate that Data Replacement mechanisms achieve a more optimal balance between preserving privacy and maintaining model utility. Therefore, we recommend the adoption of Data Replacement methods as a defense strategy against MIAs.


## Getting started 


