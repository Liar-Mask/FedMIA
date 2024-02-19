# FedMIA-Repository

### This is the official pytorch implementation of the paper:

- **Evaluating Membership Inference Attacks and Defenses in Federated Learning**


## Description

Membership Inference Attacks (MIAs) poses a growing threat to privacy preservation in federated learning. The semi-honest attacker, e.g., the server, may determine whether a particular sample belongs to a target client according to the observed model information. **This paper conducts an evaluation of existing MIAs and corresponding defense strategies.** Our evaluation on MIAs reveals **two important findings** about the trend of MIAs. **Firstly**, combining model information from multiple communication rounds (Multi-temporal) enhances the overall effectiveness of MIAs compared to utilizing model information from a single epoch.  **Secondly**, incorporating models from non-target clients (Multi-spatial) significantly improves the effectiveness of MIAs, particularly when the clients' data is homogeneous. This highlights the importance of considering the temporal and spatial model information in MIAs. Next, we assess the effectiveness via privacy-utility tradeoff for two type defense mechanisms against MIAs: Gradient Perturbation and Data Replacement. Our results demonstrate that Data Replacement mechanisms achieve a more optimal balance between preserving privacy and maintaining model utility. Therefore, we recommend the adoption of Data Replacement methods as a defense strategy against MIAs.


## Getting started 

### Preparation

Before executing the project code, please prepare the Python environment according to the `requirement.txt` file. We set up the environment with `python 3.8` and `torch 1.8.1`. 

In the experiment, we utilized two image classification datasets: [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) and [Dermnet](www.dermnet.com). CIFAR-100 contains 50,000 images and 100 categories. Dermnet includes 23,000 dermoscopic images with 23 categories.

### How to run


#### 1. Basic Training under Federated Learning:
The basic command to run our repository is in `run.sh`. We provide two models, AlexNet and ResNet, based on the above two data sets. 

During model training, the Membership Disclosure Measure (MDM) information, i.e., **data loss, cosine similarity and gradient norm**, is saved for subsequent attack implementation.

#### 2. Attacks:

We conducted a comprehensive comparison of various baseline attack methods and we classified existing methods into three types, i.e. **MIA I, MIA II and  MIA III**, based on whether the attacks utilize *temporal information and spatial information*. 

|               | Measurement    | Temporal <br> Information | Spatial <br>  Information |
| ------------- | -------------- | -------------------- | ------------------- |
| Loss-I [1]    | Data Loss      | Single               | Single              |
| Cos-I [2]     | Cos Similarity | Single               | Single              |
| Grad-Norm [3] | Gradient Norm  | Single               | Single              |
| Los-II [4]    | Data Loss      | Multi                | Single              |
| Cos-II  [2]   | Cos Similarity | Single               | Single              |
| Grad-Diff [2] | Cos similarity | Multi                | Single              |
| Loss-III Ours | Cos similarity | Multi                | Multi               |
| Cos-III Ours  | Cos similarity | Multi                | Multi               |

These attacks are implemented in `mia_attack_auto.py`.


#### 3. Defenses


Several defense methods have been proposed to mitigate Membership Inference Attacks (MIAs) in federated learning. These defense methods can be categorized into two main categories: gradient perturbation and data replacement. 
- **Gradient perturbation methods** aims to protect membership by adding perturbation on uploading gradients, including differential privacy [5,6], gradient quantization [7,8], and gradient sparsification [9,10,11]; 
- **Data replacement** aims to protect membership by modifying the training data, including Mixup[12], InstaHide [13].

These defenses are controlled by the parameter `defense`, `dp` and other parameters.


### References

1. Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh
 Jha. Privacy risk in machine learning: Analyzing the connection to overfitting. In 2018 IEEE 31st computer security foundations symposium (CSF), 2018.

2. Jiacheng Li, Ninghui Li, and Bruno Ribeiro. Effective passive membership inference attacks in federated learning against overparameterized models. In The Eleventh International Conference on Learning Representations, 2022.

3. Milad Nasr, Reza Shokri, and Amir Houmansadr. Comprehensive privacy analysis of deep learning: Passive and active white-box inference attacks against centralized and federated learning. In 2019 IEEE symposium on security and privacy (SP), 2019.

4. Yuhao Gu, Yuebin Bai, and Shubin Xu. Cs-mia: Membership
510 inference attack based on prediction confidence series in federated learning. Journal of Information Security and
Applications, 2022.

5. Robin C Geyer, Tassilo Klein, and Moin Nabi. Differentially private federated learning: A client level perspective. arXiv preprint arXiv:1712.07557, 2017.

6. Qinqing Zheng, Shuxiao Chen, Qi Long, and Weijie Su. Federated f-differential privacy. In International Conference on Artificial Intelligence and Statistics, 2021.

7. Amirhossein Reisizadeh, Aryan Mokhtari, Hamed Hassani, Ali Jadbabaie, and Ramtin Pedarsani. Fedpaq: A 
communication-efficient federated learning method with
periodic averaging and quantization. In International Conference on Artificial Intelligence and Statistics, 2020.

8. Farzin Haddadpour, Mohammad Mahdi Kamani, Aryan Mokhtari, and Mehrdad Mahdavi. Federated learning with compression: Unified analysis and sharp guarantees. In International Conference on Artificial Intelligence and Statistics, 2021.

9. Otkrist Gupta and Ramesh Raskar. Distributed learning of
deep neural network over multiple agents. Journal of Network and Computer Applications, 2018.

10. Reza Shokri and Vitaly Shmatikov. Privacy-preserving deep learning. In Proceedings of the 22nd ACM SIGSAC conference on computer and communications security, 2015.

11. Chandra Thapa, Pathum Chamikara Mahawaga Arachchige,
Seyit Camtepe, and Lichao Sun. Splitfed: When federated
learning meets split learning. In Proceedings of the AAAI Conference on Artificial Intelligence, 2022.

12. Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David
Lopez-Paz. mixup: Beyond empirical risk minimization.
arXiv preprint arXiv:1710.09412, 2017.

13. Yangsibo Huang, Zhao Song, Kai Li, and Sanjeev Arora. Instahide: Instance-hiding schemes for private distributed learning. In International conference on machine learning, 2020.