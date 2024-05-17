## Background/Motivation/Introduction
In high-energy physics, identifying and analyzing particle jets is essential for understanding particle collisions at the Large Hadron Collider (LHC). Jets, formed from quark and gluon hadronization, contain valuable information about fundamental interactions and particles like the Higgs boson. Identifying high-transverse-momentum Higgs bosons decaying into bottom quark-antiquark pairs (𝐻→𝑏𝑏ˉ) is challenging due to background noise from ordinary jets.

### Motivation
Accurately identifying Higgs bosons decaying into bottom quark-antiquark pairs advances our understanding of the Standard Model and new physics. Improved identification aids precise measurements of Higgs boson properties. Deep learning techniques, particularly graph neural networks (GNNs), show promise in capturing complex jet interactions. By leveraging GNNs, we aim to enhance 𝐻→𝑏𝑏ˉ decay identification, providing a powerful tool for high-energy physics research.

### Introduction
We propose an advanced algorithm using an interaction network to identify high-transverse-momentum Higgs bosons decaying into bottom quark-antiquark pairs and distinguish them from ordinary jets. Our approach models the jet as a combination of particle-to-particle and particle-to-vertex interactions, allowing detailed representation of jet substructure. The algorithm uses features of reconstructed charged particles and secondary vertices within a jet, trained on simulated LHC collision datasets from the CMS Collaboration. Our interaction network outperforms state-of-the-art algorithms in identifying 𝐻→𝑏𝑏ˉ decays, highlighting deep learning's potential in particle identification and enhancing high-energy physics analyses.

## Dataset Description
Dataset Components
Simulated Events: Fully simulated proton-proton collisions, including 𝐻→𝑏𝑏ˉ and background events.
Particle Features: Detailed information on charged particles, including transverse momentum, energy, pseudorapidity, azimuthal angle, impact parameters, track quality indicators, and particle identification flags.
Secondary Vertices (SVs): Information on reconstructed secondary vertices, including transverse and longitudinal displacement, number of tracks, invariant mass, energy, and cosine of the angle between the vertex flight direction and jet axis.
Jet Features: Overall jet properties, such as jet transverse momentum, mass, area, number of particles, subject information, N-subjettiness variables, and energy correlation functions.
Event Metadata: Information on the entire event, including the number of primary vertices, total event energy, and global event variables.
Labels: Ground truth labels indicating whether a jet is from a Higgs boson decay or a background process.
Dataset Sources
CERN Open Data Portal: Access to simulated collision data from CMS, including events with Higgs boson decays and QCD background processes.
MC Generators: Use Monte Carlo generators like MADGRAPH5_aMC@NLO and PYTHIA to generate signal and background events, followed by detailed detector simulations with GEANT4.

## Main Task/Challenge/Problem

### Main Task
Develop a robust algorithm using an interaction network to identify high-transverse-momentum Higgs bosons decaying into bottom quark-antiquark pairs from a large background of ordinary jets. This involves data collection and preparation, model design and implementation, training and validation, performance evaluation, mass decorrelation, and ensuring robustness and scalability.

### Challenges and Problems
Data Volume and Quality: Handling large volumes of high-dimensional collision data and ensuring data quality.
Feature Selection and Engineering: Identifying relevant features from complex data requires domain expertise.
Model Complexity: Designing effective interaction networks for graph-structured data.
Training Stability and Efficiency: Training deep models on large datasets requires careful tuning and computational resources.
Performance and Generalization: Balancing accuracy and generalization, and ensuring mass decorrelation.
Computational Resources: Substantial resources needed for training, validation, and testing.

## Methods to be Used/Explored
Graph Neural Networks (GNNs): Model relational structures within jets.
Interaction Networks (INs): Capture interactions between particles and vertices.
Components: Node features (particle and vertex properties), edge features (interactions).
Architecture: Model particle-particle and particle-vertex interactions using MLPs, aggregating results to update node features.
Graph Convolutional Networks (GCNs): Apply convolution to graph data, aggregating node information from neighbors.
Recurrent Neural Networks (RNNs): Use LSTMs to capture sequential dependencies in particle jets.
Convolutional Neural Networks (CNNs): Adapt for jet data by representing jets as image-like structures.

## Expected Outcomes/Deliverables
### Expected Outcomes
Improved Jet Tagging Performance: Achieve AUC > 0.99 for 𝐻→𝑏𝑏ˉ identification.
High Background Rejection Rate: Background rejection factor > 1000 at 30% TPR, > 500 at 50% TPR.
Robustness to Pileup: Maintain high performance with varying levels of pileup.
Mass Decorrelation: Achieve high decorrelation metric (e.g., 1/DJS > 1000) while maintaining jet tagging performance.
### Deliverables
Trained Interaction Network Model: Optimized for 𝐻→𝑏𝑏ˉ jet tagging.
Source Code and Documentation: Complete code and documentation for model replication.
Performance Evaluation Report: Detailed report on model performance and comparisons.
Mass Decorrelation Techniques: Implement and document methods for unbiased tagging.
Data Handling and Preparation Pipeline: Modular pipeline for CMS open data and simulated datasets.
Integration with Experimental Workflows: Guidelines for model integration into high-energy physics workflows.
Published Research Paper: Findings published in a peer-reviewed journal or conference.


## Schedule
A: Molan Li
B: Yaran Yang 
C: Shuyang Zhang
D: Michael Zhang
E: Christian Amezcua  
- Week 8
1. Create github(done) 
2. Preprocess data (A+B): 
Check the CERN Open Data Portal, choosing the collision data and simulated data of H → b¯b jets.
Split data into training and testing data.
3. Exploratory data analysis(C+D+E):
Visualizing data(create histogram).
Generate statics information.
Record finding and potential features.
- Week 9
1. Define Metrics (Done)
2. Create simple benchmark (A + D):
Build a simple model, train the model using the data.
Tune hyperparameters to ensure model convergence and prevent overfitting.
3. Train and Evaluate (B + E):
Train the simple benchmark model.
Evaluate its performance (Compare performance against the metrics).

- Week 10
1. Create more advanced model (A + B + C)
Build more expert IN model.
2. Train and Compare Performance (C + D + E)
Train IN model (and DDB model).
Evaluate and compare performance (IN, DDB, benchmark model).
Optimize the model architecture for performance and scalability.
Identify results and make a conclusion (All people).
## Reference
E. A. Moreno et al., “Interaction networks for the identification of boosted H →
bb decays”
[Link](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.012010)


