import h5py
import numpy as np

def create_dummy_data(filename, num_jets=100, n_particles=30, n_vertices=5):
    params_3 = ['sv_ptrel', 'sv_erel', 'sv_phirel', 'sv_etarel', 'sv_deltaR', 'sv_pt', 'sv_mass', 'sv_ntracks', 'sv_normchi2', 'sv_dxy', 'sv_dxysig', 'sv_d3d', 'sv_d3dsig', 'sv_costhetasvpv']
    params_2 = ['track_ptrel', 'track_erel', 'track_phirel', 'track_etarel', 'track_deltaR', 'track_drminsv', 'track_drsubjet1', 'track_drsubjet2', 'track_dz', 'track_dzsig', 'track_dxy', 'track_dxysig', 'track_normchi2', 'track_quality', 'track_dptdpt', 'track_detadeta', 'track_dphidphi', 'track_dxydxy', 'track_dzdz', 'track_dxydz', 'track_dphidxy', 'track_dlambdadz', 'trackBTag_EtaRel', 'trackBTag_PtRatio', 'trackBTag_PParRatio', 'trackBTag_Sip2dVal', 'trackBTag_Sip2dSig', 'trackBTag_Sip3dVal', 'trackBTag_Sip3dSig', 'trackBTag_JetDistVal']
    labels_qcd = {"label_QCD_b": 53, "label_QCD_bb": 51, "label_QCD_c": 54, "label_QCD_cc": 52, "label_QCD_others": 55}
    label_H_bb = 41

    with h5py.File(filename, 'w') as f:
        for param in params_2:
            f.create_dataset(param, data=np.random.rand(num_jets, n_particles))
        for param in params_3:
            f.create_dataset(param, data=np.random.rand(num_jets, n_vertices))
        f.create_dataset("fj_label", data=np.random.choice(list(labels_qcd.values()) + [label_H_bb], size=num_jets))
        f.create_dataset("event_no", data=np.arange(num_jets))

filename = "new_dummy_data.h5"
create_dummy_data(filename)

import h5py
import numpy as np
import itertools
import torch

filename = "new_dummy_data.h5"
h5 = h5py.File(filename, 'r')

params_3 = ['sv_ptrel', 'sv_erel', 'sv_phirel', 'sv_etarel', 'sv_deltaR', 'sv_pt', 'sv_mass', 'sv_ntracks', 'sv_normchi2', 'sv_dxy', 'sv_dxysig', 'sv_d3d', 'sv_d3dsig', 'sv_costhetasvpv']
params_2 = ['track_ptrel', 'track_erel', 'track_phirel', 'track_etarel', 'track_deltaR', 'track_drminsv', 'track_drsubjet1', 'track_drsubjet2', 'track_dz', 'track_dzsig', 'track_dxy', 'track_dxysig', 'track_normchi2', 'track_quality', 'track_dptdpt', 'track_detadeta', 'track_dphidphi', 'track_dxydxy', 'track_dzdz', 'track_dxydz', 'track_dphidxy', 'track_dlambdadz', 'trackBTag_EtaRel', 'trackBTag_PtRatio', 'trackBTag_PParRatio', 'trackBTag_Sip2dVal', 'trackBTag_Sip2dSig', 'trackBTag_Sip3dVal', 'trackBTag_Sip3dSig', 'trackBTag_JetDistVal']
labels_qcd = {"label_QCD_b": 53, "label_QCD_bb": 51, "label_QCD_c": 54, "label_QCD_cc": 52, "label_QCD_others": 55}
label_H_bb = 41
num_jets = h5["event_no"].shape[0]
n_particles = h5["track_dxydxy"].shape[-1]
n_vertices = h5["sv_phirel"].shape[-1]

X, Y, y = [], [], []
for jet in range(num_jets):
    if h5["fj_label"][jet] in list(labels_qcd.values()):
        y.append([0, 1])
    elif h5["fj_label"][jet] == label_H_bb:
        y.append([1, 0])
    else:
        continue
    
    x = [h5[feature][jet] for feature in params_2]
    X.append(np.array(x).reshape(n_particles, -1))
    
    y1 = [h5[feature][jet] for feature in params_3]
    Y.append(np.array(y1).reshape(n_vertices, -1))
    
    if jet > 10:
        break

X, Y, y = np.array(X), np.array(Y), np.array(y)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train, Y_train, y_train = torch.tensor(X, device=device), torch.tensor(Y, device=device), torch.tensor(y, device=device)

import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

class IN(nn.Module):
    def __init__(self, hidden=60, N_p=30, params=60, N_v=14, params_v=5, De=20, Do=24, softmax=True):
        super(IN, self).__init__()
        self.hidden = hidden
        self.Np = N_p
        self.P = params
        self.Npp = self.Np * (self.Np - 1)
        self.Nv = N_v
        self.S = params_v
        self.Npv = self.Np * self.Nv
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
        indices = torch.tensor(list(itertools.product(range(N), range(N))), dtype=torch.long, device=device)
        mask = indices[:, 0] != indices[:, 1]
        indices = indices[mask]
        Rr = torch.zeros(N, Nr, device=device)
        Rs = torch.zeros(N, Nr, device=device)
        Rr[indices[:, 0], torch.arange(Nr, device=device)] = 1
        Rs[indices[:, 1], torch.arange(Nr, device=device)] = 1
        return Rr, Rs

    def assign_matrices_SV(self, N, Nt, Nv):
        indices = torch.tensor(list(itertools.product(range(N), range(Nv))), dtype=torch.long, device=device)
        Rk = torch.zeros(N, Nt, device=device)
        Rv = torch.zeros(Nv, Nt, device=device)
        Rk[indices[:, 0], torch.arange(Nt, device=device)] = 1
        Rv[indices[:, 1], torch.arange(Nt, device=device)] = 1
        return Rk, Rv

    def forward(self, x, y):
        Orr = x @ self.Rr
        Ors = x @ self.Rs
        B = torch.cat([Orr, Ors], 1)
        E = self.fr_pp(B.view(-1, 2 * self.P)).view(-1, self.Npp, self.De)
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar_pp = (E @ self.Rr.t().contiguous())
        Ork = x @ self.Rk
        Orv = y @ self.Rv
        C = torch.cat([Ork, Orv], 1)
        C = C.transpose(1, 2).contiguous()
        F = self.fr_pv(C.view(-1, self.P + self.S)).view(-1, self.Npv, self.De)
        F = torch.transpose(F, 1, 2).contiguous()
        Ebar_pv = (F @ self.Rk.t().contiguous())
        Ebar = Ebar_pp + Ebar_pv
        Ebar = torch.transpose(Ebar, 1, 2).contiguous()
        J = torch.cat([x, Ebar], 2)
        O = self.fo(J.view(-1, J.shape[-1])).view(-1, self.Np, self.Do)
        O = torch.transpose(O, 1, 2).contiguous()
        O = torch.sum(O, 2)
        O = self.fc(O)
        if self.softmax:
            O = torch.sigmoid(O)
        return O

hidden = 60
N_p = n_particles
params = len(params_2)
N_v = len(params_3)
params_v = n_vertices
De = 20
Do = 24

model = IN(hidden=hidden, N_p=N_p, params=params, N_v=N_v, params_v=params_v, De=De, Do=Do).to(device)

from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt

output = model(X_train.float(), Y_train.float())

# Convert one-hot labels to binary labels
y_true = y_train.argmax(axis=1).cpu().detach().numpy()

# Calculate accuracy
predictions = (output.cpu().detach().numpy() > 0.5).astype(int)
accuracy = accuracy_score(y_true[:output.shape[0]], predictions)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true[:output.shape[0]], output.cpu().detach().numpy().flatten())
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Print accuracy
print(f'Accuracy: {accuracy:.2f}')
