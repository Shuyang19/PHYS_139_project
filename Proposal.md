# Background/motivation/introduction

 - Background:
   
In high-energy physics, the identification and analysis of particle jets is fundamental to understanding the outcomes of particle collisions, such as those occurring at the Large Hadron Collider (LHC). Jets are collimated sprays of particles resulting from the hadronization of quarks and gluons. These jets contain valuable information about the fundamental interactions and particles involved in the collision processes. 

Traditionally, jet tagging was limited on light-flavor particles, but since the large center-of-mass avaliable in LHC collisions, heavy particles would also be invovled, and boosted-jet topologies are accessible, which makes jat-tagging became a more complex task.

Among the numerous particles produced in such collisions, the Higgs boson plays a pivotal role due to its association with the mechanism that gives mass to other particles.

Identifying high-transverse-momentum Higgs bosons decaying into bottom quark-antiquark pairs (𝐻→𝑏𝑏ˉH→b bˉ) is particularly challenging. This process is often obscured by the overwhelming background of ordinary jets from quark and gluon interactions. Traditional jet tagging algorithms rely on specific features and thresholds to differentiate between these signals and background noise. However, as the complexity and volume of collision data increase, these conventional methods face limitations in performance and accuracy.

 - Motivation:
   
The accurate identification of Higgs bosons decaying into bottom quark-antiquark pairs is crucial for advancing our understanding of the Standard Model and exploring potential new physics beyond it. Improvements in this area can lead to more precise measurements of the Higgs boson properties, including its couplings and self-interactions, which are essential for verifying theoretical predictions and discovering any deviations that could hint at new physics phenomena.

Deep learning techniques, especially those utilizing graph neural networks (GNNs), have shown great promise in capturing the complex interactions within jets. Unlike traditional methods, GNNs can effectively model the relational structure between particles, making them well-suited for tasks like jet tagging. By leveraging the latest advancements in deep learning and graph-based representations, we aim to develop an algorithm that significantly enhances the identification performance of 𝐻→𝑏𝑏ˉH→b bˉdecays, thereby providing a powerful tool for high-energy physics research.

 - Introduction:
   
In this project, we propose an advanced algorithm based on an interaction network to identify high-transverse-momentum Higgs bosons decaying to bottom quark-antiquark pairs and distinguish them from ordinary jets that arise from quark and gluon interactions at short distances. Our approach models the jet as a combination of particle-to-particle and particle-to-vertex interactions, allowing for a detailed representation of the jet substructure.

The algorithm's inputs include features of the reconstructed charged particles within a jet and the secondary vertices associated with these particles. By describing the jet shower through these interactions, the model is trained to learn a jet representation optimized for the classification task. The training is performed on simulated datasets of realistic LHC collisions provided by the CMS Collaboration through the CERN Open Data Portal.

Our interaction network approach outperforms existing state-of-the-art algorithms in identifying 𝐻→𝑏𝑏ˉH→b bˉdecays, demonstrating a significant improvement in performance. This is achieved by utilizing an extended feature representation and a graph-based model that captures the intricate details of the jet structure. The success of this method highlights the potential of deep learning techniques in revolutionizing particle identification and paves the way for more accurate and efficient analyses in high-energy physics.

Through this project, we aim to contribute to the experimental precision of 𝐻→𝑏𝑏ˉH→b bˉmeasurements, which are crucial for probing the Higgs boson's properties and exploring new physics beyond the Standard Model. The developed algorithm can serve as a valuable tool for future analyses at the LHC and other particle colliders, enhancing our ability to uncover the fundamental secrets of the universe.

# Dataset description

 ## Dataset Components
 
 - Simulated Events:

The dataset should consist of fully simulated events of proton-proton collisions at the LHC, provided by the CMS Collaboration or similar sources.
These events should include both signal events (Higgs bosons decaying to bottom quark-antiquark pairs, 𝐻→𝑏𝑏ˉH→b bˉ) and background events (ordinary jets from quark and gluon interactions).

 - Particle Features:

Detailed information about the charged particles within each jet.
Features should include:
Transverse momentum (𝑝𝑇p T)
Energy(𝐸E)
Pseudorapidity (𝜂η)
Azimuthal angle (𝜙ϕ)
Impact parameters (transverse 𝑑0d0, longitudinal 𝑧0z 0)
Track quality indicators
Covariance matrix entries for the track parameters
Particle identification (PID) flags if available
Secondary Vertices (SVs):

Information about reconstructed secondary vertices associated with each jet.

 - Features should include:
Transverse displacement (𝑑𝑥𝑦d xy)
Longitudinal displacement (𝑑𝑧d z)
Number of associated tracks
Invariant mass
Energy
Cosine of the angle between the vertex flight direction and the jet axis (cos𝜃cosθ)

 - Jet Features:

Overall properties of the jets, such as:
Jet transverse momentum (𝑝𝑇p T)
Jet mass (including soft-drop mass)
Jet area
Number of constituent particles
Subjet information (from jet clustering algorithms)
N-subjettiness variables (𝜏𝑁τ N)
Energy correlation functions

 - Event Metadata:

Information about the entire event, such as:
Number of primary vertices
Total event energy
Global event variables (e.g., missing transverse energy)

 - Labels:
Ground truth labels indicating whether a jet is from a Higgs boson decay (𝐻→𝑏𝑏ˉH→b bˉ) or a background process (QCD jet).
These labels are used for supervised training of the algorithm.

 ## Dataset Sources:
 
 - CERN Open Data Portal: Provides access to simulated collision data from CMS, including events with Higgs boson decays and QCD background processes.

 - MC Generators: Monte Carlo generators like MADGRAPH5_aMC@NLO and PYTHIA can be used to generate signal and background events, followed by detailed detector simulations with GEANT4 to mimic the CMS detector response.

# Main task/challenge/problem

 ## Main Task
The primary task of this project is to develop a robust and highly accurate algorithm based on an interaction network to identify high-transverse-momentum Higgs bosons decaying into bottom quark-antiquark pairs (𝐻→𝑏𝑏ˉH→b bˉ) from a large background of ordinary jets produced by quark and gluon interactions. This involves several steps:

 - Data Collection and Preparation:

Acquire high-quality simulated datasets of proton-proton collisions from the LHC, including detailed information on jet constituents and secondary vertices.
Preprocess the data to create a suitable format for training and evaluation of the interaction network model.

 - Model Design and Implementation:

Develop an interaction network that can effectively model the complex relationships between particles within a jet and between particles and secondary vertices.
Implement the network using a suitable machine learning framework, such as PyTorch Geometric, ensuring efficient handling of graph-structured data.

 - Training and Validation:

Train the interaction network on the prepared dataset, using advanced optimization techniques to ensure convergence and prevent overfitting.
Validate the model on a separate subset of the data to monitor its performance and make necessary adjustments to the architecture and hyperparameters.

 - Performance Evaluation:

Evaluate the model's performance using metrics such as accuracy, area under the ROC curve (AUC), and background rejection rate.
Compare the performance of the interaction network with existing state-of-the-art algorithms to quantify improvements.

 - Mass Decorrelation:

Implement techniques to decorrelate the model's output from the jet mass to ensure unbiased selection in physics analyses.
Test and validate the effectiveness of these decorrelation methods.

 - Robustness and Scalability:

Ensure the model's robustness against varying conditions, such as different levels of pileup (multiple overlapping collisions).
Scale the model to handle large datasets efficiently, potentially using distributed computing resources.

## Challenges and Problems

 - Data Volume and Quality:

Handling and processing large volumes of high-dimensional collision data is computationally intensive and requires significant storage and memory resources.
Ensuring the quality and realism of the simulated data to accurately reflect the conditions of actual LHC collisions is critical for the model's performance in real-world scenarios.

 - Feature Selection and Engineering:

Identifying the most relevant features from the complex and high-dimensional data is challenging and requires domain expertise in high-energy physics.
Proper feature engineering is crucial for improving the model's ability to distinguish between signal and background jets.
Model-Related Challenges

 - Complexity of Interactions:

Modeling the intricate interactions within jets, including both particle-particle and particle-vertex interactions, requires sophisticated network architectures.
Designing and implementing an effective interaction network that can learn from graph-structured data is technically challenging.

 - Training Stability and Efficiency:

Training deep learning models on large and complex datasets can be unstable and prone to overfitting.
Efficient training requires careful tuning of hyperparameters, optimization techniques, and the use of hardware accelerators like GPUs.
Performance and Evaluation Challenges

 - Balancing Accuracy and Generalization:

Achieving high accuracy on the training data while ensuring the model generalizes well to unseen data is a delicate balance.
Overfitting to the training data can lead to poor performance on actual collision data, undermining the model's utility.

 - Mass Decorrelation:

Ensuring the model's output is decorrelated from the jet mass is essential to prevent bias in physics analyses.
Implementing and validating decorrelation techniques without significantly compromising the model's performance is challenging.
Computational and Resource Challenges

 - Computational Resources:

The project requires substantial computational resources for training, validating, and testing the model, especially when scaling to large datasets.
Access to high-performance computing clusters or cloud-based resources is often necessary to handle the computational demands.
Integration and Deployment:

Integrating the developed model into existing high-energy physics analysis workflows requires compatibility with the tools and frameworks used in the field.
Deploying the model for real-time inference in experimental setups demands optimized and efficient implementations.


# Method(s) to be used/explored (e.g. RNNs, GNNs, …)

Graph Neural Networks (GNNs)
Interaction Networks (INs)
Interaction Networks are a type of Graph Neural Network (GNN) specifically designed to model interactions within a system of objects. They are particularly well-suited for tasks in high-energy physics where the relationships between particles are crucial for understanding jet substructure.

 - Overview:

Interaction Networks treat particles and secondary vertices within a jet as nodes in a graph. The interactions between these nodes are represented as edges.
These networks can effectively capture the complex dependencies between particles and vertices, learning a detailed representation of the jet structure.

 - Components:

Node Features: Include properties of particles (e.g., momentum, energy, impact parameters) and secondary vertices (e.g., displacement, number of tracks).
Edge Features: Represent interactions between particles and between particles and vertices (e.g., distance measures, relative momentum).

 - Architecture:

Particle-Particle Interactions: Modeled using a Multi-Layer Perceptron (MLP) that processes pairs of particle features.
Particle-Vertex Interactions: Modeled using another MLP that processes combined features of particles and vertices.
Aggregation: The results of these interactions are aggregated to update the node features, capturing the overall structure of the jet.

 - Advantages:

Flexibility: Can handle a variable number of particles and vertices in each jet.
Expressiveness: Able to learn complex, non-linear interactions between particles and vertices, leading to better performance in distinguishing signal from background.
Graph Convolutional Networks (GCNs)
Graph Convolutional Networks extend the concept of convolution from grid-based data (e.g., images) to graph-structured data, making them useful for tasks involving relational data like particle jets.

 - Overview:

Graph Convolutional Networks apply convolution operations to nodes in a graph, aggregating information from neighboring nodes to learn node representations.
Suitable for capturing local interactions within a jet, enhancing the identification of specific jet substructures. 

 - Architecture:

Node Features: Similar to INs, include particle and vertex properties.
Convolutional Layers: Perform local aggregation of node features using learned filters, propagating information across the graph.

 - Advantages:

Scalability: Efficiently handles large graphs by leveraging sparse operations.
Locality: Focuses on local neighborhoods, making it effective for capturing fine-grained jet substructure details.
Recurrent Neural Networks (RNNs)
Long Short-Term Memory Networks (LSTMs)
LSTMs are a type of Recurrent Neural Network (RNN) that are particularly effective at learning from sequential data, which can be useful for modeling the sequence of particles in a jet.

 - Overview:

LSTMs use gates to control the flow of information, allowing them to maintain long-term dependencies and mitigate the vanishing gradient problem common in traditional RNNs.

 - Application:

Sequential Features: Treat the sequence of particles within a jet as a time series, where each particle is a time step.
Feature Processing: Process particle features sequentially, capturing dependencies along the jet's particle sequence.

 - Advantages:

Temporal Dependencies: Effectively models long-range dependencies between particles in a jet.
Memory: Maintains a memory of previous particles, which can be crucial for understanding the jet's evolution.
Convolutional Neural Networks (CNNs)
CNNs are typically used for image data but can be adapted to work with jet data by representing the jet as an image-like structure.

 - Overview:

Jet Images: Represent jets as 2D histograms where pixel values correspond to particle properties (e.g., transverse momentum).
Convolutional Layers: Apply convolutional filters to these jet images to extract spatial features.

 - Application:

Image Representation: Convert the jet into a 2D image based on particle coordinates and properties.
Feature Extraction: Use convolutional layers to extract features from the jet image, focusing on spatial patterns.

 - Advantages:

Spatial Features: Captures spatial features of the jet, useful for identifying characteristic jet shapes and substructures.
Established Techniques: Leverages well-established techniques in computer vision for jet analysis.
Method Integration and Comparison

 - Integration:

Combine the strengths of different methods to improve overall performance. For example, use GNNs for relational data processing and CNNs for spatial feature extraction.
Develop hybrid models that incorporate multiple neural network architectures, leveraging their complementary strengths.

 - Comparison:

Compare the performance of various models (INs, GCNs, LSTMs, CNNs) on the task of 𝐻→𝑏𝑏ˉH→b bˉidentification.
Evaluate models based on metrics such as accuracy, area under the ROC curve (AUC), and background rejection rate.

# Expected outcomes/deliverables (e.g. jet tagging AUC > X)

 ## Expected Outcomes
 
 - Improved Jet Tagging Performance:

Achieve a significant improvement in the area under the receiver operating characteristic curve (AUC) for the identification of Higgs bosons decaying to bottom quark-antiquark pairs (𝐻→𝑏𝑏ˉH→b bˉ) compared to existing state-of-the-art methods.

 - Target AUC: Aim for an AUC greater than 0.99, indicating high accuracy in distinguishing signal jets from background jets.
   
 - High Background Rejection Rate:

Demonstrate a high background rejection rate, particularly at low false positive rates (FPR).
Target Background Rejection: Achieve a background rejection factor (1/FPR) of over 1000 at a true positive rate (TPR) of 30%, and over 500 at a TPR of 50%.

 - Robustness to Pileup:

Ensure the model maintains high performance in the presence of pileup (multiple overlapping collisions), which is common in LHC data.
Target Robustness: Show minimal degradation in performance metrics (AUC, background rejection) as the number of reconstructed primary vertices increases.

 - Mass Decorrelation:

Successfully implement mass decorrelation techniques to ensure the model’s output is independent of the jet mass, preventing bias in physics analyses.
Target Decorrelation: Achieve a high decorrelation metric (e.g., 1/DJS > 1000) while maintaining competitive jet tagging performance.
Deliverables

 - Trained Interaction Network Model:

Provide a fully trained interaction network model, optimized for the 𝐻→𝑏𝑏ˉH→b bˉ jet tagging task.
The model should be capable of making predictions with high accuracy and efficiency.

 - Source Code and Documentation:

Deliver the complete source code for the interaction network model, including the data preprocessing, training, validation, and testing scripts.
Provide detailed documentation explaining the model architecture, training process, and instructions for replicating the results.

 - Performance Evaluation Report:

Compile a comprehensive report detailing the model's performance on various metrics, including AUC, background rejection rates, and robustness to pileup.
Include comparisons with existing state-of-the-art jet tagging algorithms to highlight improvements.

 - Mass Decorrelation Techniques:

Implement and document the mass decorrelation methods used to ensure unbiased jet tagging.
Provide an analysis of the effectiveness of these techniques, including their impact on jet tagging performance.

 - Data Handling and Preparation Pipeline:

Develop and deliver a pipeline for data handling and preparation, tailored to the CMS open data and simulated datasets used in this project.
Ensure the pipeline is modular and reusable for future projects involving similar datasets.

 - Integration with Experimental Workflows:

Ensure the trained model and accompanying tools are compatible with existing experimental workflows used in high-energy physics.
Provide guidelines for integrating the model into these workflows, facilitating its use in real-world analyses at the LHC.

 - Published Research Paper:

Publish the findings of the project in a peer-reviewed journal or conference, detailing the methodology, results, and implications for high-energy physics.
Include a thorough discussion of the advantages of the interaction network approach and potential areas for future research.
Additional Deliverables

 - User-Friendly Interface:

Develop a user-friendly interface for running the model on new datasets, allowing researchers to easily apply the trained model to their own data.
Provide example scripts and use cases to demonstrate the model's application.

 - Performance Monitoring and Visualization Tools:

Create tools for monitoring the model’s performance during training and evaluation, including real-time visualization of metrics such as loss, accuracy, and AUC.
Include tools for visualizing the learned representations and interactions within the jets, aiding in the interpretability of the model.

