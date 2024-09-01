import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import grad
from torch import nn
from torchmetrics import Metric
from model.layer.mlp_layer import MLPNet
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
class SINN_ODMetrics(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("accuracies", default=[], dist_reduce_fx="mean")
        self.add_state("f1_scores", default=[], dist_reduce_fx="mean")
        self.add_state("loss", default=[], dist_reduce_fx="mean")
        
    def update(self, mode=None, loss=None, pred_labels=None, target_labels=None):
        if mode == 'val':
            self.loss.append(loss)
        else:
            accuracy = accuracy_score(target_labels.cpu().numpy(), pred_labels)
            self.accuracies.append(accuracy)
            
            f1 = f1_score(target_labels.cpu().numpy(), pred_labels, average='macro')
            self.f1_scores.append(f1)
        
    def compute(self, mode):
        if mode == 'val':
            return {
            'loss': torch.mean(torch.stack(self.loss)),
        }
        else:
            return {
                'accuracy': sum(self.accuracies)/len(self.accuracies),
                'f1_score': sum(self.f1_scores)/len(self.f1_scores)
            }



def gradients_mse(ode_in, ode_out, rhs):
    gradients = diff_gradient(ode_out, ode_in)  ## Left hand side of ODE $\tilde{x}_u(t)/dt$
    ODE_loss = (gradients - rhs).pow(2).sum(-1)
    return ODE_loss


def diff_gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def gumbel_softmax(logits, temperature=0.2):
    device = logits.device
    eps = 1e-20
    u = torch.rand(logits.shape, device=device)
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)
    
    
class SINN(nn.Module):
    def __init__(self, num_users=1, network_type='sinn', act_type='relu', hidden_channels=256, num_layers=3, nclasses=None, type_odm='degroot', alpha=1.0, beta=0.1, K=1, ):
        super().__init__()
        self.U = num_users
        self.network_type = network_type
        self.type_odm = type_odm
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.J = 1
        self.nclasses = nclasses
        self.net = MLPNet(num_users=self.U, 
                          num_layers=num_layers,
                          hidden_features=hidden_channels, outermost_linear=act_type, nonlinearity=act_type)
        self.val2label = nn.Linear(1, nclasses)
        self.init_ode_params()

    def init_ode_params(self):
        if self.type_odm == "degroot":
            self.M = nn.Parameter(torch.rand(self.U, self.K) / self.U)
            self.Q = nn.Parameter(torch.rand(self.U, self.K) / self.U)
        elif self.type_odm == "sbcm":
            self.rho = nn.Parameter(torch.ones(1))
        elif self.type_odm == "bcm":
            self.mu = nn.Parameter(torch.ones(1))
            self.threshold = nn.Parameter(torch.tensor([1.]))
            self.sigma = nn.Parameter(torch.tensor([1.]))
        elif self.type_odm == "fj":
            self.s_u = nn.Parameter(torch.zeros(self.U))

    def forward(self, batch):
        times = batch['ti']
        uids = batch['ui']
        output = self.net(times, uids)
        opinion_label = self.val2label(output)
        batch['opinion'] = output
        batch['opinion_label'] = opinion_label
        
        return batch

    def ode_constraints(self, batch):
        device = batch['ui'].device
        tau_j = torch.rand(self.J, device=device).unsqueeze(1).requires_grad_(True)
        users = torch.arange(self.U, device=device).unsqueeze(1)
        taus = tau_j.repeat(users.shape[0],1)
        _vector_x = self.net(taus, users)
        vector_x = torch.transpose(torch.reshape(_vector_x, (self.U, self.J)), 1, 0)
        user_id = torch.randint(self.U-1, (1,1), device=device)
        x_u = self.net(tau_j, user_id)
    
        if self.type_odm=="degroot":
            m_u = torch.index_select(torch.abs(self.M),0,user_id[:,0])
            Q = torch.abs(self.Q)
            a_u = torch.matmul(m_u, torch.transpose(Q,1,0))

            ## Right hand side (rhs) of Equation (5)
            rhs_ode = torch.matmul(a_u, vector_x.T)

            ## Regularization term $\mathcal{R}(\Lambda)$
            regularizer = self.beta * (torch.norm(self.M) + torch.norm(self.Q))
        if self.type_odm=="sbcm":
            distance = torch.abs(x_u - vector_x)

            ## Probability of user $u$ selecting user $v$ as an interaction partner at time $\tau_j$
            p_uv = (distance + 1e-12).pow(self.rho)

            ## Differentiable one-hot approximation $\tilde{z}_u^t$ in Equation (9)
            tilde_z_ut = self.sampling(p_uv)

            ## Right hand side (rhs) of Equation (10)
            rhs_ode = tilde_z_ut * (x_u - vector_x)
            rhs_ode = rhs_ode.sum(-1)

            ## Regularization term $\mathcal{R}(\Lambda)$
            regularizer = self.beta * torch.zeros(1)

        if self.type_odm=="bcm":
            mu = torch.abs(self.mu)
            distance = torch.abs(x_u - vector_x)

            ## Prepare ODE parameters 
            sigma = torch.abs(self.sigma)
            threshold = torch.abs(self.threshold)

            ## Right hand side (rhs) of Equation (8)
            rhs_ode = mu * torch.sigmoid( sigma*(threshold - distance) ) * (x_u - vector_x) 
            rhs_ode = rhs_ode.sum(-1)

            ## Regularization term $\mathcal{R}(\Lambda)$
            regularizer = self.beta * (torch.norm(sigma) + torch.norm(mu)) 

        if self.type_odm=="fj":
            ## Initial opinions of $U$ users 
            initial_opinion_gt = torch.index_select(batch['initial'][:1,:], 1, user_id[:,0]) 

            ## Get user $u$'s susceptibility to persuasion
            s_u = torch.gather(torch.abs(self.s_u),0,user_id[:,0])  

            ## Right hand side (rhs) of Equation (7)
            rhs_ode = s_u * vector_x.sum(-1) + (1.-s_u) * initial_opinion_gt - x_u

            ## Regularization term $\mathcal{R}(\Lambda)$
            regularizer = self.beta * torch.norm(s_u)
        rhs_ode = torch.reshape(rhs_ode, (-1,self.J))

        ### Compute ODE loss $\mathcal{L}_{ode}$
        ode_constraints = gradients_mse(tau_j, x_u, rhs_ode)
        ode_constraint = self.alpha * ode_constraints
        return regularizer, ode_constraint, tilde_z_ut

    
    def sampling(self,vec):
        vec = F.softmax(vec, dim=1)
        logits = gumbel_softmax(vec, 0.1)
        return logits
    
    def loss(self, batch):
        pred_opinion_label = batch['opinion_label'] ## Predicted opinion label
        gt_latent_opinion = batch['ground_truth_opinion']
        loss_func = nn.CrossEntropyLoss()
        data_loss = loss_func(pred_opinion_label, gt_latent_opinion[:,0].long())
        if self.network_type == 'sinn':
            regularizer, ode_constraint, tilde_z_ut = self.ode_constraints(batch)
            return data_loss + ode_constraint.mean() + regularizer.mean()
        elif self.network_type == 'nn':
            return data_loss
    
    def prediction2label(self, x):
        f_x = np.exp(x) / np.sum(np.exp(x), axis=-1)[:,None]
        label = np.argmax(f_x, axis=-1)
        return label

    def prediction(self, batch):
        batch = self(batch)
        dfs = []
        att_dfs = []
        zu_dfs = []
        test_ui = batch['ui'].cpu().detach().numpy().flatten()#[0]
        test_ti = batch['ti'].cpu().detach().numpy().flatten()
        test_oi = batch["ground_truth_opinion"].cpu().detach().numpy().flatten()
        test_pred = batch['opinion'].cpu().detach().numpy().flatten()
        if 'opinion_label' in batch.keys():
            test_pred_label = self.prediction2label(batch['opinion_label'].detach().cpu().numpy())
            test_oi = test_oi/(self.nclasses-1)
        else: 
            test_pred_label = test_pred * (self.nclasses-1)
        tmpdf = pd.DataFrame(data = np.c_[test_ui, test_ti, test_oi, test_pred, test_pred_label], columns=["user","time","gt","pred","pred_label"])
        dfs.append(tmpdf)
        if 'zu' in batch.keys() and not batch['zu'] is None:
            zu_pred = batch['zu'].detach().numpy()
            print(zu_pred.shape, test_ui.shape)
            if test_ui.shape[0]==zu_pred.shape[0]:
                zu_tmpdf = pd.DataFrame(data = np.c_[test_ui[:,np.newaxis], zu_pred], columns=["user"]+list(range(zu_pred.shape[1]))) 
                zu_dfs.append(zu_tmpdf)
        return dfs