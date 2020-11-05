# Biometric Backdoors: A Poisoning Attack Against Unsupervised Template Updating

This repository contains the code for the paper "[Biometric Backdoors: A Poisoning Attack Against Unsupervised Template Updating](https://arxiv.org/pdf/1905.09162.pdf)" published in the Proceedings of the [5th IEEE European Symposium on Security and Privacy](https://www.ieee-security.org/TC/EuroSP2020/).
This work is a collaboration between [Giulio Lovisotto](https://github.com/giuliolovisotto/) and [Simon Eberz](https://www.cs.ox.ac.uk/people/simon.eberz/) from the [System Security Lab](http://www.cs.ox.ac.uk/groups/seclab/) at University of Oxford.


## Idea
In this work, we investigated a new attack vector for biometric authentication, by modeling an attacker who exploits the unsupervised *template update* procedure.

During template update, a biometric recognition system will add new samples to the known template (i.e., the user's stored biometric reference) under a certain policy. 
These added samples will help the system recognize the user in the future. This procedure
is in place in consumer biometric systems such as Apple FaceId and Siri.

However, when such updates are computed in an unsupervised manner, 
i.e., without the supervision of a human, the procedure can be exploited by the
adversary to cause a *drift* in the user template in latent space: by planting a *biometric backdoor*.
If the drift is crafted carefully then it will lead to the adversary being able to
inconspicuously access the system.

<p align="center"><img src="/github_images/poisoning_concept.gif" width="50%"></p>

## Attack Realization

Modern biometric systems often rely on deep neural networks (DNN) to extract
input features which are later used for recognition. 
Unfortunately, DNNs have been shown to be vulnerable to adversarial examples (AE): 
input with subtle changes which mislead the network into wrong predictions.
One way to realize such AE in practice is to wear a pair of glasses which have
a coloured frame, as introduced in ["Accessorize to a Crime"](https://dl.acm.org/doi/10.1145/2976749.2978392).
We adapt the methods of that paper to work in our poisoning use-case, by improving
the method's resilience to *input uncertainty* and  *failed authentication attempts*.

<p align="center"><img src="/github_images/attack_steps.gif" width="80%"></p>

## Result 

We evaluate the attack on several system configurations, based on three different
network architectures: FaceNet, VGGFace, ResNet-50.
The attack leads to the adversary being able to impersonate the user 
after less than 10 injection attempts in 70\% of cases.
The attack is also inconspicuous with respect to the false reject rate for the victim:
even after the backdoor is planted the victim can still use the system without
being rejected.

We also evaluate the performace in a black-box setting where the adversary does not
know the target network and uses a different network as a surrogate, showing that 
the attack can transfer across different architectures.

<p align="center"><img src="/github_images/transferability.png" width="80%"></p>

## Countermeasure

We design a countemeasure whose goal is to stop the template drift necessary for the
attack. The countermeasure checks whether consecutive template updates have similar
*angular similarity* in latent space. We measure the tradeoff between thwarting the 
attack and not impeding legitimate template updates by using a set of attributes
intra-user variation factors automatically labeled from the Google Vision API.

We show that the countermeasure has a 7-14\% equal error rate against legitimate 
updates, leading to 99\% of attacks being detected after a couple of injection attempts.

<p align="center"><img src="/github_images/countermeasure.gif" width="65%"></p>

## Resources

IEEE Euro S&P presentation [slides](https://github.com/ssloxford/biometric-backdoor/blob/master/images/talk.pdf).  
IEEE Euro S&P presentation [video](https://www.youtube.com/watch?v=h3s21WnJWYk).  
[![Conference presentation video](https://img.youtube.com/vi/h3s21WnJWYk/0.jpg)](https://www.youtube.com/watch?v=h3s21WnJWYk)

## Code
The conda environment used for this project is saved in
[biometric-backdoor.yml](https://github.com/ssloxford/biometric-backdoor/blob/master/biometric-backdoor.yml).
At the time (~2018), I also used the following libraries (that are not included in the environment):
 * [deepface](https://github.com/serengil/deepface) installed from source
 * [dlib](https://github.com/davisking/dlib) installed from source
 * [cleverhans](https://github.com/tensorflow/cleverhans) installed from source
 * [Google Vision API](https://cloud.google.com/vision) with the python client
 * [MAX-Facial-Age-Estimator](https://github.com/IBM/MAX-Facial-Age-Estimator) downloaded in a docker container

 
The code is contained in the following folders:
```
.
├── face                  # Face-related scripts, most of preprocessing
│   ├── adversarial       # Scripts to run evaluation on face
├── matchers              # Implementation of the three matchers
└── images                # Code to generate images used in the paper
```

Be aware that if you attempt to re-use this code you have to create `config.json` and
`face/config.json` based on the templates and on your local configuration.

### Models

In the paper I used four different models: (1) [FaceNet](https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view), (2) [FaceNet-CASIA](https://drive.google.com/file/d/1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz/view), (3) ResNet-50 and (4) VGG16.
Models 1 and 2 are taken from [https://github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet). Models 3 and 4 are taken from an old version of the `deepface` pip package, where they are called `FaceRecognizerResnet` and `FaceRecognizerVGG`, respectively.


## Citation
If you use this repository please cite the paper as follows:
```
@INPROCEEDINGS{9230411,
  author={G. {Lovisotto} and S. {Eberz} and I. {Martinovic}},
  booktitle={2020 IEEE European Symposium on Security and Privacy (EuroS\&P)}, 
  title={Biometric Backdoors: A Poisoning Attack Against Unsupervised Template Updating}, 
  year={2020},
  volume={},
  number={},
  pages={184-197},
  doi={10.1109/EuroSP48549.2020.00020}}
```
NB! An older arxiv version of the paper uses the id `lovisotto2019biometric`.

## Contributors
 * [Giulio Lovisotto](https://github.com/giuliolovisotto/)
 * [Simon Eberz](https://www.cs.ox.ac.uk/people/simon.eberz/)

## Acknowledgements

This work was generously supported by a grant from Mastercard and by the Engineering and Physical Sciences Research Council grant number EP/N509711/1.
 

