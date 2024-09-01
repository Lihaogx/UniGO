import torch
from torch import nn



class MLPNet(nn.Module):
    def __init__(self, num_users, num_layers, hidden_features, out_features=1,
                 outermost_linear='sigmoid', nonlinearity='relu'):
        super(MLPNet, self).__init__()

        nls = {'relu': nn.ReLU(inplace=True), 
               'sigmoid': nn.Sigmoid(), 
               'tanh': nn.Tanh(), 
               'selu': nn.SELU(inplace=True), 
               'softplus': nn.Softplus(), 
               'elu': nn.ELU(inplace=True)}

        nl = nls[nonlinearity]
        nl_outermost = nls[outermost_linear]

        self.hidden_features = hidden_features

        self.embed_users = [] 
        self.embed_users.append(nn.Sequential(
                nn.Embedding(num_users, hidden_features), nl
        ))
        for i in range(num_layers-1):
            self.embed_users.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nl
            ))
        self.embed_users = nn.Sequential(*self.embed_users)

        self.embed_times = []
        self.embed_times.append(nn.Sequential(
            nn.Linear(1, hidden_features), nl
        ))
        for i in range(num_layers-1):
            self.embed_times.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nl
            ))
        self.embed_times = nn.Sequential(*self.embed_times)

        self.net = []
        self.net.append(nn.Sequential(
            nn.Linear(hidden_features*2, hidden_features), nl
        ))
        for i in range(num_layers):
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nl
            ))
        self.net.append(nn.Sequential(
                nn.Linear(hidden_features, out_features), nl_outermost
        ))
        self.net = nn.Sequential(*self.net)

        
    def forward(self, times, users, profs=None, params=None, **kwargs):

        x = self.embed_times(times.float())
        y = self.embed_users(users.long())
        y = torch.squeeze(y, dim=1)


        combined = torch.cat([x, y], dim=-1)
        output = self.net(combined)

        return output


