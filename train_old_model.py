# %%
import torch.nn as nn
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import wandb
import pandas as pd


class ProteinProteinInteractionPrediction(nn.Module):
    def __init__(self,mod_embed,prot_embed,dim=20,layer_gnn=2):
        super(ProteinProteinInteractionPrediction, self).__init__()
        self.prot_embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.mod_embed_fingerprint = nn.Embedding(nmod_fingerprint, dim)
        self.mod_W_gnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_gnn)])
        self.prot_gnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_gnn)])
        self.prot_W1_attention = nn.Linear(dim, dim)
        self.prot_W2_attention = nn.Linear(dim, dim)
        self.prot_w = nn.Parameter(torch.zeros(dim))
        self.W1_attention = nn.Linear(dim, dim)
        self.W2_attention = nn.Linear(2*dim, dim)  # Modified to accept concatenated protein vector
        self.w = nn.Parameter(torch.zeros(dim))
        self.W_out = nn.Sequential(
            nn.Linear(2*dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        
        self.mod_embed = mod_embed
        self.prot_embed = prot_embed

        self.rdkit_linear = nn.Linear(2048, dim)

    def gnn(self, xs1, A1, type):
        for i in range(layer_gnn):
            if type == "mod":
                hs1 = torch.relu(self.mod_W_gnn[i](xs1))
            elif type == "prot":
                hs1 = torch.relu(self.prot_gnn[i](xs1))
            xs1 = torch.matmul(A1, hs1)

        return xs1
    
    def mutual_attention(self, h1, h2):
        x1 = self.W1_attention(h1)
        x2 = self.W2_attention(h2)

        m1 = x1.size()[0]
        m2 = x2.size()[0]

        c1 = x1.repeat(1, m2).view(m1, m2, dim)
        c2 = x2.repeat(m1, 1).view(m1, m2, dim)

        d = torch.tanh(c1 + c2)
        alpha = torch.matmul(d, self.w).view(m1, m2)

        b1 = torch.mean(alpha, 1)
        p1 = torch.softmax(b1, 0)
        s1 = torch.matmul(torch.t(x1), p1).view(-1, 1)

        b2 = torch.mean(alpha, 0)
        p2 = torch.softmax(b2, 0)
        s2 = torch.matmul(torch.t(x2), p2).view(-1, 1)

        return torch.cat((s1, s2), 0).view(1, -1), p1, p2
    def prot_mutual_attention(self, h1, h2):
        x1 = self.prot_W1_attention(h1)
        x2 = self.prot_W2_attention(h2)

        m1 = x1.size()[0]
        m2 = x2.size()[0]

        c1 = x1.repeat(1, m2).view(m1, m2, dim)
        c2 = x2.repeat(m1, 1).view(m1, m2, dim)

        d = torch.tanh(c1 + c2)
        alpha = torch.matmul(d, self.prot_w).view(m1, m2)

        b1 = torch.mean(alpha, 1)
        p1 = torch.softmax(b1, 0)
        s1 = torch.matmul(torch.t(x1), p1).view(-1, 1)

        b2 = torch.mean(alpha, 0)
        p2 = torch.softmax(b2, 0)
        s2 = torch.matmul(torch.t(x2), p2).view(-1, 1)

        return torch.cat((s1, s2), 0).view(1, -1), p1, p2

    def forward(self, inputs, train = True):
        fingerprints1, adjacency1, fingerprints2, adjacency2, fingerprints3, adjacency3, smiles = inputs

        """Protein vector with GNN."""
        x_fingerprints1 = self.mod_embed_fingerprint(fingerprints1)
        x_fingerprints2 = self.prot_embed_fingerprint(fingerprints2)
        x_fingerprints3 = self.prot_embed_fingerprint(fingerprints3)
        if self.mod_embed == "gnn":
            x_mod = self.gnn(x_fingerprints1, adjacency1, "mod")
        elif self.mod_embed == "rdkit":
            # Implement RDKit molecular fingerprinting
            from rdkit import Chem
            from rdkit.Chem import AllChem

            # Convert SMILES to RDKit mol object
            mol = Chem.MolFromSmiles(smiles)
            # Generate Morgan fingerprint
            radius = 2  # Radius of 2 is equivalent to ECFP4
            nBits = 2048  # Number of bits in the fingerprint
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            
            # Convert bit vector to PyTorch tensor
            x_mod = torch.tensor(morgan_fp.ToBitString()).float().view(1, -1)
            x_mod = self.rdkit_linear(x_mod)            
        elif self.mod_embed == "graphmvp":
            x_mod = self.graphmvp_embed(fingerprints1)
        elif self.mod_embed == "infograph":
            x_mod = self.infograph_embed(fingerprints1)

        if self.prot_embed == "gnn":
            x_protein2 = self.gnn(x_fingerprints2, adjacency2, "prot")
            x_protein3 = self.gnn(x_fingerprints3, adjacency3, "prot")
        # x_mod = N, dim

        """Concatenate protein vectors"""
        x_proteins, p1, p2 = self.prot_mutual_attention(x_protein2, x_protein3)
        # print(f"shape of x_proteins: {x_proteins.shape}")
        # print(f"shape of x_protein2: {x_protein2.shape}")
        # print(f"shape of x_protein3: {x_protein3.shape}")
        """Protein vector with mutual-attention."""
        y, p1, p2 = self.mutual_attention(x_mod, x_proteins)

        z_interaction = self.W_out(y)
        # ADD CHEMICAL FEATURE VECTOR HERE 
        return z_interaction, p1, p2

    def __call__(self, data, train=True):
        inputs, t_interaction = data[:-1], data[-1]
        z_interaction, p1, p2 = self.forward(inputs, train)
        if train:
            loss = F.cross_entropy(z_interaction, t_interaction)
            return loss
        else:
            z = F.softmax(z_interaction, 1).to("cpu").data[0].numpy()
            t = int(t_interaction.to("cpu").data[0].numpy())
            return z, t, p1, p2

# %%

class Trainer(object):
    def __init__(self, model, lr = 1e-4):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        self.model.train()  # Set model to training mode
        sampling = random.choices(dataset, k=3000)
        # sampling = dataset
        loss_total = 0
        for data in tqdm(sampling):
            try:
                mod, p1, p2, interaction, family = data
                A1 = np.load(
                        prot_data[prot_data['protein'] == p1].iloc[0]['adj_path'],
                        allow_pickle=True
                    )
                A2 = np.load(
                    prot_data[prot_data['protein'] == p2].iloc[0]['adj_path'],
                    allow_pickle=True
                )

                P1 = np.load(
                    prot_data[prot_data['protein'] == p1].iloc[0]['fp_path'],
                    allow_pickle=True
                )
                P2 = np.load(
                    prot_data[prot_data['protein'] == p2].iloc[0]['fp_path'],
                    allow_pickle=True
                )
                mod_fp = np.load(
                    mod_data[mod_data['inchikey'] == mod].iloc[0]['fp_path'],
                    allow_pickle=True
                )
                mod_adj = np.load(
                    mod_data[mod_data['inchikey'] == mod].iloc[0]['adj_path'],
                    allow_pickle=True
                )
                smiles = mod_data[mod_data['inchikey'] == mod].iloc[0]['smiles']
            except Exception as e:
                # print(f"failed for {e}")
                continue
            protein1 = torch.LongTensor(P1.astype(np.float32))
            protein2 = torch.LongTensor(P2.astype(np.float32))
            adjacency1 = torch.FloatTensor(A1.astype(np.float32))
            adjacency2 = torch.FloatTensor(A2.astype(np.float32))
            mod_fp = torch.LongTensor(mod_fp.astype(np.float32))
            mod_adj = torch.FloatTensor(mod_adj.astype(np.float32))
            interaction = torch.LongTensor([interaction.astype(int)])

            # comb = (protein1.to(device), adjacency1.to(device), protein2.to(device), adjacency2.to(device), interaction.to(device))
            # with torch.no_grad():
            # prot_loss, prot_embed = self.prot_model(comb, train=True)
            
            comb = (mod_fp.to(device), mod_adj.to(device), 
                    protein1.to(device), adjacency1.to(device), 
                    protein2.to(device), adjacency2.to(device), smiles.to(device),
                    interaction.to(device))
            loss = self.model(comb)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        return loss_total / len(sampling)


# %%


class Tester(object):
    def __init__(self, model, wandb_project_name=False):
        self.model = model

        # Initialize wandb if a project name is provided
        # if wandb_project_name:
        #     wandb.init(project=wandb_project_name)
        #     self.use_wandb = True
        # else:
        #     self.use_wandb = False

    def test(self, dataset, epoch=None):
        # sampling = dataset
        sampling = random.choices(dataset, k=500)
        z_list, t_list = [], []
        for data in tqdm(sampling):
            try:
                mod, p1, p2, interaction, _ = data
                A1 = np.load(
                    prot_data[prot_data['protein'] == p1].iloc[0]['adj_path'],
                    allow_pickle=True
                )
                A2 = np.load(
                    prot_data[prot_data['protein'] == p2].iloc[0]['adj_path'],
                    allow_pickle=True
                )
                P1 = np.load(
                    prot_data[prot_data['protein'] == p1].iloc[0]['fp_path'],
                    allow_pickle=True
                )
                P2 = np.load(
                    prot_data[prot_data['protein'] == p2].iloc[0]['fp_path'],
                    allow_pickle=True
                )
                mod_fp = np.load(
                    mod_data[mod_data['inchikey'] == mod].iloc[0]['fp_path'],
                    allow_pickle=True
                )
                mod_adj = np.load(
                    mod_data[mod_data['inchikey'] == mod].iloc[0]['adj_path'],
                    allow_pickle=True
                )
            except Exception as e:
                # print(f"failed for {mod}, {p1}, and {p2}: {e}")
                continue

            protein1 = torch.LongTensor(P1.astype(np.float32))
            protein2 = torch.LongTensor(P2.astype(np.float32))
            adjacency1 = torch.FloatTensor(A1.astype(np.float32))
            adjacency2 = torch.FloatTensor(A2.astype(np.float32))
            mod_fp = torch.LongTensor(mod_fp.astype(np.float32))
            mod_adj = torch.FloatTensor(mod_adj.astype(np.float32))
            interaction = torch.LongTensor([interaction.astype(int)])

            # comb = (protein1.to(device), adjacency1.to(device), protein2.to(device), adjacency2.to(device), interaction.to(device))
            # with torch.no_grad():
            # prot_loss, prot_embed = self.prot_model(comb, train=True)
            
            comb = (mod_fp.to(device), mod_adj.to(device), 
                    protein1.to(device), adjacency1.to(device), 
                    protein2.to(device), adjacency2.to(device), 
                    interaction.to(device))
            z, _, _, _ = self.model(comb, train=False)
            # print(z,interaction)
            # print(z,torch.argmax(torch.FloatTensor(z)).item())
            z_list.append(z)
            t_list.append(interaction)

        score_list, label_list = [], []
        for z in z_list:
            score_list.append(z[1].item())
            label_list.append(torch.argmax(torch.FloatTensor(z)).item())

        labels = np.array(label_list)
        y_true = np.array([t.item() for t in t_list])
        y_pred = np.array(score_list)

        (
            tp,
            fp,
            tn,
            fn,
            accuracy,
            precision,
            sensitivity,
            recall,
            specificity,
            MCC,
            F1_score,
            Q9,
            ppv,
            npv,
        ) = self.calculate_performace(len(sampling), labels, y_true)
        roc_auc_val = roc_auc_score(y_true, y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        auc_val = auc(fpr, tpr)

        # Log results to wandb
        # if self.use_wandb:
        #     wandb.log({
        #         "Epoch": epoch,
        #         "Accuracy": accuracy,
        #         "Precision": precision,
        #         "Recall": recall,
        #         "Sensitivity": sensitivity,
        #         "Specificity": specificity,
        #         "MCC": MCC,
        #         "F1 Score": F1_score,
        #         "ROC AUC": roc_auc_val,
        #         "AUC": auc_val,
        #         "TP": tp,
        #         "FP": fp,
        #         "TN": tn,
        #         "FN": fn,
        #         "PPV": ppv,
        #         "NPV": npv,
        #     })

        return (
            accuracy,
            precision,
            recall,
            sensitivity,
            specificity,
            MCC,
            F1_score,
            roc_auc_val,
            auc_val,
            Q9,
            ppv,
            npv,
            tp,
            fp,
            tn,
            fn,
        )

    def result(
        self,
        epoch,
        time,
        loss,
        accuracy,
        precision,
        recall,
        sensitivity,
        specificity,
        MCC,
        F1_score,
        roc_auc_val,
        auc_val,
        Q9,
        ppv,
        npv,
        tp,
        fp,
        tn,
        fn,
        file_name,
    ):
        with open(file_name, "a") as f:
            result = map(
                str,
                [
                    epoch,
                    time,
                    loss,
                    accuracy,
                    precision,
                    recall,
                    sensitivity,
                    specificity,
                    MCC,
                    F1_score,
                    roc_auc_val,
                    auc_val,
                    Q9,
                    ppv,
                    npv,
                    tp,
                    fp,
                    tn,
                    fn,
                ],
            )
            f.write("\t".join(result) + "\n")

    def calculate_performace(self, test_num, pred_y, labels):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        test_num = len(pred_y)
        for index in range(test_num):
            if labels[index] == 1:
                if labels[index] == pred_y[index]:
                    tp = tp + 1
                else:
                    fn = fn + 1
            else:
                if labels[index] == pred_y[index]:
                    tn = tn + 1
                else:
                    fp = fp + 1

        if (tp + fn) == 0:
            q9 = float(tn - fp) / (tn + fp + 1e-06)
        if (tn + fp) == 0:
            q9 = float(tp - fn) / (tp + fn + 1e-06)
        if (tp + fn) != 0 and (tn + fp) != 0:
            q9 = 1 - float(np.sqrt(2)) * np.sqrt(float(fn * fn) / ((tp + fn) * (tp + fn)) + float(fp * fp) / ((tn + fp) * (tn + fp)))

        Q9 = (float)(1 + q9) / 2
        accuracy = float(tp + tn) / test_num
        precision = float(tp) / (tp + fp + 1e-06)
        sensitivity = float(tp) / (tp + fn + 1e-06)
        recall = float(tp) / (tp + fn + 1e-06)
        specificity = float(tn) / (tn + fp + 1e-06)
        ppv = float(tp) / (tp + fp + 1e-06)
        npv = float(tn) / (tn + fn + 1e-06)
        F1_score = float(2 * tp) / (2 * tp + fp + fn + 1e-06)
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

        return tp, fp, tn, fn, accuracy, precision, sensitivity, recall, specificity, MCC, F1_score, Q9, ppv, npv

    def save_model(self, model, file_name):
        torch.save(model.state_dict(), file_name)

# %%
z = torch.FloatTensor(np.array([0.1, 0.2], dtype=np.float32))
print(z, torch.argmax(z,dim=0))

# %%

# get data
prot_data = pd.read_csv("prot_data.csv")
mod_data = pd.read_csv("mod_data.csv")
train_data = pd.read_csv("interaction_data.csv")
examples = np.array(train_data.values.tolist())
# examples = np.array(random.choices(examples_all, k=500))

# setup folders
prot_fp_folder = "protein_fingerprints"
mod_fp_folder = "mod_fingerprints"
prot_fp_dict = np.load("protein_fingerprints/prot_fingerprint_dict.pickle",allow_pickle=True)
mod_fp_dict = np.load("mod_fingerprints/mod_fingerprint_dict.pickle",allow_pickle=True)


n_fingerprint = len(prot_fp_dict) + 100
nmod_fingerprint = len(mod_fp_dict) + 100


### Hyperparameters ###

radius         = 1
dim        = 100
layer_gnn      = 2
lr             = 1e-4
lr_decay       = 0.5
decay_interval = 10
iteration      = 30

import wandb
import timeit
import torch

# Initialize wandb run
# wandb.init(project="promisegat4")  # Replace with your actual entity name

fold_count = 1


# %%

for train, test in kfold.split(examples):
    dataset_train = examples[train]  # mod, prot1, prot2, int, int_family
    dataset_test = examples[test]

    start = timeit.default_timer()

    model = ProteinProteinInteractionPrediction(dim=dim, layer_gnn=layer_gnn, mod_embed="gnn", prot_embed = "gnn").to(device)
    trainer = Trainer(model, lr = lr)
    file_model = "ppim/model/" + "sep1_model_fold_" + str(fold_count)
    file_result = "ppim/result/" + "sep1_results_fold_" + str(fold_count) + ".txt"

    wandb.init(project="promisegat4")
    # Log file paths to wandb
    wandb.config.update({
        "fold": fold_count,
        "model_path": file_model,
        "result_path": file_result,
        "protein_embed": "gcn",
        "mod_embed": "gcn",
        "radius": radius,
        "dim": dim,
        "layer_gnn": layer_gnn,
        "lr": lr,
        "lr_decay": lr_decay,
        "decay_interval": decay_interval,
        "iteration": iteration
    })

    for epoch in range(iteration):
        if (epoch + 1) % decay_interval == 0:
            trainer.optimizer.param_groups[0]["lr"] *= lr_decay

        loss = trainer.train(dataset_train)
        print(f"finished with loss {loss}")

        # Log training loss and GPU usage to wandb
        gpu_usage = torch.cuda.memory_allocated(device=device) / 1024 ** 3  # in GB
        wandb.log({"epoch": epoch, "loss": loss, "gpu_usage_gb": gpu_usage, "fold": fold_count})

        tester = Tester(model)
        (
            accuracy,
            precision,
            recall,
            sensitivity,
            specificity,
            MCC,
            F1_score,
            roc_auc_val,
            auc_val,
            Q9,
            ppv,
            npv,
            tp,
            fp,
            tn,
            fn,
        ) = tester.test(dataset_test, epoch=epoch)
        end = timeit.default_timer()
        time = end - start

        # Log results to wandb
        wandb.log({
            "epoch": epoch,
            "time": time,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "MCC": MCC,
            "F1_score": F1_score,
            "ROC_AUC": roc_auc_val,
            "AUC": auc_val,
            "Q9": Q9,
            "PPV": ppv,
            "NPV": npv,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "fold": fold_count
        })

        tester.result(
            epoch,
            time,
            loss,
            accuracy,
            precision,
            recall,
            sensitivity,
            specificity,
            MCC,
            F1_score,
            roc_auc_val,
            auc_val,
            Q9,
            ppv,
            npv,
            tp,
            fp,
            tn,
            fn,
            file_result,
        )
        tester.save_model(model, file_model)


        print("Epoch: " + str(epoch))
        print("Accuracy: " + str(accuracy))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("Sensitivity: " + str(sensitivity))
        print("Specificity: " + str(specificity))
        print("MCC: " + str(MCC))
        print("F1-score: " + str(F1_score))
        print("ROC-AUC: " + str(roc_auc_val))
        print("AUC: " + str(auc_val))
        print("Q9: " + str(Q9))
        print("PPV: " + str(ppv))
        print("NPV: " + str(npv))
        print("TP: " + str(tp))
        print("FP: " + str(fp))
        print("TN: " + str(tn))
        print("FN: " + str(fn))
        print("\n")

        torch.manual_seed(1234)
    wandb.finish()
    fold_count += 1


