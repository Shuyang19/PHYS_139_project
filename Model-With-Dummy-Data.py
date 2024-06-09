import os
import h5py
import numpy as np
import itertools
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d

# Function to create dummy data
def create_dummy_data(filename, num_jets=100, n_particles=30, n_vertices=5):
    params_3 = [
        'sv_ptrel', 'sv_erel', 'sv_phirel', 'sv_etarel', 'sv_deltaR',
        'sv_pt', 'sv_mass', 'sv_ntracks', 'sv_normchi2', 'sv_dxy', 'sv_dxysig',
        'sv_d3d', 'sv_d3dsig', 'sv_costhetasvpv'
    ]
    params_2 = [
        'track_ptrel', 'track_erel', 'track_phirel', 'track_etarel', 'track_deltaR',
        'track_drminsv', 'track_drsubjet1', 'track_drsubjet2', 'track_dz', 'track_dzsig',
        'track_dxy', 'track_dxysig', 'track_normchi2', 'track_quality', 'track_dptdpt',
        'track_detadeta', 'track_dphidphi', 'track_dxydxy', 'track_dzdz', 'track_dxydz',
        'track_dphidxy', 'track_dlambdadz', 'trackBTag_EtaRel', 'trackBTag_PtRatio',
        'trackBTag_PParRatio', 'trackBTag_Sip2dVal', 'trackBTag_Sip2dSig', 'trackBTag_Sip3dVal',
        'trackBTag_Sip3dSig', 'trackBTag_JetDistVal'
    ]

    labels_qcd = {"label_QCD_b": 53, "label_QCD_bb": 51, "label_QCD_c": 54, "label_QCD_cc": 52, "label_QCD_others": 55}
    label_H_bb = 41

    with h5py.File(filename, 'w') as f:
        for param in params_2:
            f.create_dataset(param, data=np.random.rand(num_jets, n_particles))
        
        for param in params_3:
            f.create_dataset(param, data=np.random.rand(num_jets, n_vertices))
        
        f.create_dataset("fj_label", data=np.random.choice(list(labels_qcd.values()) + [label_H_bb], size=num_jets))
        f.create_dataset("event_no", data=np.arange(num_jets))

# Close the file if it is open and delete it
filename = "new_dummy_data.h5"
if os.path.exists(filename):
    os.remove(filename)

# Create the dummy data
create_dummy_data(filename)

# Load the dummy data
h5 = h5py.File(filename, 'r')

params_3 = [
    'sv_ptrel', 'sv_erel', 'sv_phirel', 'sv_etarel', 'sv_deltaR',
    'sv_pt', 'sv_mass', 'sv_ntracks', 'sv_normchi2', 'sv_dxy', 'sv_dxysig',
    'sv_d3d', 'sv_d3dsig', 'sv_costhetasvpv'
]

params_2 = [
    'track_ptrel', 'track_erel', 'track_phirel', 'track_etarel', 'track_deltaR',
    'track_drminsv', 'track_drsubjet1', 'track_drsubjet2', 'track_dz', 'track_dzsig',
    'track_dxy', 'track_dxysig', 'track_normchi2', 'track_quality', 'track_dptdpt',
    'track_detadeta', 'track_dphidphi', 'track_dxydxy', 'track_dzdz', 'track_dxydz',
    'track_dphidxy', 'track_dlambdadz', 'trackBTag_EtaRel', 'trackBTag_PtRatio',
    'trackBTag_PParRatio', 'trackBTag_Sip2dVal', 'trackBTag_Sip2dSig', 'trackBTag_Sip3dVal',
    'trackBTag_Sip3dSig', 'trackBTag_JetDistVal'
]

labels_qcd = {"label_QCD_b": 53, "label_QCD_bb": 51, "label_QCD_c": 54, "label_QCD_cc": 52, "label_QCD_others": 55}
label_H_bb = 41
num_jets = h5["event_no"].shape[0]
n_particles = h5["track_dxydxy"].shape[-1]
n_vertices = h5["sv_phirel"].shape[-1]

# track features
X = []

# SV features
Y = []

# labels
y = []
edge_indices = []
for jet in range(num_jets):
    if h5["fj_label"][jet] in list(labels_qcd.values()):
        y.append([0, 1])
    elif h5["fj_label"][jet] == label_H_bb:
        y.append([1, 0])
    else:
        continue
    
    x = []
    for feature in params_2:
        x.append(h5[feature][jet])
    X.append(np.array(x).reshape(n_particles, -1))
    
    y1 = []
    for feature in params_3:
        y1.append(h5[feature][jet])
    Y.append(np.array(y1).reshape(n_vertices, -1))
    
    # complete graph has n_particles*(n_particles-1)/2 edges, here we double count each edge, so has  n_particles*(n_particles-1) total edges
    pairs = np.stack([[m, n] for (m, n) in itertools.product(range(n_particles), range(n_particles)) if m != n])
    # edge_index = torch.tensor(pairs, dtype=torch.long)
    edge_index = pairs.transpose()
    edge_indices.append(edge_index)
    
    if jet > 10:
        break

# Convert lists to numpy arrays
X = np.array(X)
Y = np.array(Y)
y = np.array(y)

# Convert numpy arrays to tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train = torch.tensor(X, device=device)
Y_train = torch.tensor(Y, device=device)
y_train = torch.tensor(y, device=device)

# Define the model
class IN(nn.Module):
    def __init__(self, hidden=60, N_p=30, params=60, N_v=14, params_v=5, De=20, Do=24, softmax=True):
        super(IN, self).__init__()  # Call the superclass's __init__ method
        self.hidden = hidden
        self.Np = N_p  # N
        self.P = params
        self.Npp = self.Np * (self.Np - 1)  # Nr
        self.Nv = N_v
        self.S = params_v
        self.Npv = self.Np * self.Nv  # Nt
        self.De = De
        self.Do = Do
        self.softmax = softmax

        self.fr_pp = Seq(Lin(2 * self.P, self.hidden), ReLU(), Lin(self.hidden, self.De))
        self.fr_pv = Seq(Lin(self.P + self.S, self.hidden), ReLU(), Lin(self.hidden, self.De))
        self.fo = Seq(Lin(self.P + self.De, self.hidden), ReLU(), Lin(self.hidden, self.Do))
        self.fc = nn.Linear(self.Do, 1)

        self.Rr, self.Rs = self.assign_matrices(self.Np, self.Npp)
        self.Rk, self.Rv = self.assign_matrices_SV(self.Np, self.Npv, self.Nv)

    def assign_matrices(self, N, Nr):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        indices = torch.tensor(list(itertools.product(range(N), range(N))), dtype=torch.long, device=device)
        mask = indices[:, 0] != indices[:, 1]
        indices = indices[mask]
        Rr = torch.zeros(N, Nr, device=device)
        Rs = torch.zeros(N, Nr, device=device)
        Rr[indices[:, 0], torch.arange(Nr)] = 1
        Rs[indices[:, 1], torch.arange(Nr)] = 1
        return Rr, Rs

    def assign_matrices_SV(self, N, Nt, Nv):
        indices = torch.tensor(list(itertools.product(range(N), range(Nv))), dtype=torch.long)
        Rk = torch.zeros(N, Nt)
        Rv = torch.zeros(Nv, Nt)
        Rk[indices[:, 0], torch.arange(Nt)] = 1
        Rv[indices[:, 1], torch.arange(Nt)] = 1
        if torch.cuda.is_available():
            Rk = Rk.cuda()
            Rv = Rv.cuda()
        return Rk, Rv

    def forward(self, x, y):
        Orr = x @ self.Rr
        Ors = x @ self.Rs
        B = torch.cat([Orr, Ors], 1)
        E = self.fr_pp(B.view(-1, 2 * self.P)).view(-1, self.Npp, self.De)
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar_pp = (E @ self.Rr.t().contiguous())
        del E
        Ork = x @ self.Rk
        Orv = y @ self.Rv
        C = torch.cat([Ork, Orv], 1)
        C = C.transpose(1, 2).contiguous()
        F = self.fr_pv(C.view(-1, self.P + self.S)).view(-1, self.Npv, self.De)
        del C
        F = torch.transpose(F, 1, 2).contiguous()
        Ebar_pv = (F @ self.Rk.t().contiguous())
        del F
        Ebar = Ebar_pp + Ebar_pv
        del Ebar_pp
        del Ebar_pv
        Ebar = torch.transpose(Ebar, 1, 2).contiguous()
        J = torch.cat([x, Ebar], 2)
        
        # Debug print to check the shape before reshaping
        print(f"Shape of J before reshaping: {J.shape}")

        O = self.fo(J.view(-1, J.shape[-1])).view(-1, self.Np, self.Do)

        # Debug print to check the shape after reshaping
        print(f"Shape of O after reshaping: {O.shape}")

        del J
        O = torch.transpose(O, 1, 2).contiguous()
        O = torch.sum(O, 2)
        O = self.fc(O)
        if self.softmax:
            O = torch.sigmoid(O)
        return O

# Initialize and print the model
hidden = 60
N_p = n_particles
params = len(params_2)
N_v = len(params_3)
params_v = n_vertices
De = 20
Do = 24

model = IN(hidden=hidden, N_p=N_p, params=params, N_v=N_v, params_v=params_v, De=De, Do=Do).to(device)
print(model)

# Forward pass with dummy data
output = model(X_train.float(), Y_train.float())
print(output)
