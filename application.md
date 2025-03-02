# Privacy-preserving Federated models with Extreme Learning Machines

Extreme Learning Machines (ELM) are fast and scalable machine learning models [1]. They have a unique property of having an explicit formula for the best solution that requires no iterative improvements, unlike other projection based models like logistic regression or multi-layer neural networks. This exact solution makes the model very fast, and enables applications that iteratively process new data instead of iteratively training a model on the same set of data. 

Federated learning [2] is an approach, or an arrangement, of a machine learning task where the data resides permanently on multiple remove machines called "clients". The central machine called "server" is tasked with computing a model without copying the data into its storage over the network. Despite seeming unusual, this arrangement corresponds to the majority of machine learning use cases in the real world - because the data is either personal and poses privacy or legal risks when shared (the "data leakage" incidents that happen regularly across the world), or the data is too big to transfer and store at a central location. The latter use case holds almost always, as the price of multiple simple and small devices is much lower than the price of a centralised data storage and processing even with the cloud optimisations, as this is how hardware scaling works.

The typical approach to federated learning is exchanging model weights that are partially trained ("updated") by clients on their local parts of the overall "data". Server gathers weights from multiple, combines them in some manner, and sends the combined weights for a number of clients for another round of training. The weights will converge over multiple rounds, and represent the model that has the knowledge of the whole dataset without explicitly sharing the data.

An ELM application to federated learning follows a different philosophy. Instead of sharing the weights, ELM shares an intermediate data representation about the relations between different data features - or more precisely, between the features of a non-linear high dimensional projection of the data. The original data cannot be derived from this intermediate representation because the non-linear projection is not reversible with the finite precision of physical computers, and the shared data includes only the lengths and angles between the new high-dimensional feature vectors instead of the features themselves. This means there are infinitely many data reconstructions corresponding to the shared information, and the data cannot be reconstructed reliably.

In addition, recent research [3] discovered that the ELM model tolerates reasonable amount of added noise to the shared data representation. This serves as another privacy preserving feature. But a second use case for the noise is controlling the "quality" of the shared information, enabling scenarios like "share full data with other devices of the same company, while sharing a lower quality information publicly that is still useful but won't give competitors an advantage".


## WS1: Open-source federated ELM based onto an established open-source framework

We want to create and publish afederated learning version of ELM integrated into an established federated learning framework. The reason of using an established framework is to avoid duplicating the work on communication between models and clients, enable support for multiple federated approaches, easy support for different languages and technologies e.g. for clients on mobile phones or edge devices, and fair comparison with other federated learning approaches. An established framework will significantly simplify an implementation of any real federated learning system that would utilize ELM as its core model.

We preliminary chose Flower [3] as the framework, because it is fully open-source without a proprietary version, has an active community, a very actively developed code base in Github, and an excellent website with tutorials enabling any developer to start using it reasonably fast.

The deliverables would be:

- An open source federated ELM implementation based on an established framework, with code published on GitHub with an open source license. Code includes comparison to other methods, examples and tutorials for its use.
- A paper with an implementation details.
- (optional) A paper implementing a federated learning system based on the framework, aiming at clients being smartphones or edge devices, in a similar setup to a medical domain. Focus on data privacy, evaluate the effectiveness of privacy preserving function of federated ELM.


## WS2: Federated ELM application to crowdsourced soil mapping for public wellfare

Our prior works on machine learning for soil mapping [4] created a mobile client with an edge computing system that can create and display a predicted soil map based on a combination of an established dataset and user inputs. The use case focused on acid sulfate soils detection in southwestern coast of Finland.

Acid sulfate soils are widespread in several other countries including Europe and Australia. A well functioning mobile client makes us (Arcada) a valuable research partner for international geological institutions in research for predicting acid sulfate soils in their countries. Other applications are equally possible because the tool is problem-agnostic, and can take any map layers and labelled points for input.

The deliverables would be:

- A joint project with a foreign research institution (general university of geological institutions) researching the improvements in predictions for their application domain like acid sulfate soils, using the mobile map client for crowdsourcing data input
- An adaptation of federated ELM for the mobile client that enables federated training withot exchanging the labelled map points explicitly between the clients and a server


## References

1. Akusok, Anton, et al. "High-performance extreme learning machines: a complete toolbox for big data applications." IEEE Access 3 (2015): 1011-1025.

2. Beutel, Daniel J., et al. "Flower: A friendly federated learning research framework." arXiv preprint arXiv:2007.14390 (2020).

3. Akusok, Anton, et al. "Data Obfuscation Scenarios for Batch ELM in Federated Learning Applications." SRE2024: 21st International Conference on Smart Technologies & Education (2024).

4. Akusok, Anton, et al. "Native interaction experience for computational maps with mobile devices." Proceedings of the 16th International Conference on PErvasive Technologies Related to Assistive Environments. 2023.