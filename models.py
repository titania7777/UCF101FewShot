import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from torchvision.models.video import r2plus1d_18
from torchvision.models.video.resnet import BasicBlock, Conv2Plus1D
from utils import freeze_all, freeze_layer, freeze_bn, initialize_linear, initialize_3d
# torch.backends.cudnn.enabled = False



class R2Plus1D(nn.Module):
    def __init__(self, way=5, shot=1, query=5, metric="cosine"):
        super(R2Plus1D, self).__init__()
        self.way = way
        self.shot = shot
        self.query = query
        self.metric = metric

        # r2plus1d_18
        model = r2plus1d_18(pretrained=True)
        
        # encoder(freezing)
        self.encoder_freeze = nn.Sequential(
            model.stem,
            model.layer1,
            model.layer2,
            model.layer3,
        )
        self.encoder_freeze.apply(freeze_all)

        # encoder(for cosine similarity)
        if self.metric == "cosine" or self.metric == "euclidean":
            self.encoder_tune = nn.Sequential(
                model.layer4,
                nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
            )

        # ralation module
        if self.metric == "relation":
            self.relation1 = nn.Sequential(
                BasicBlock(512, 256, Conv2Plus1D, stride=2, downsample=self._downsample(512, 256)),
                nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
            )

            self.relation2 = nn.Sequential(
                nn.Linear(256, 128),
                nn.Softplus(),
                nn.Linear(128, 1),
            )
            self.relation2.apply(initialize_linear)

        # scaler
        self.scaler = nn.Parameter(torch.tensor(5.0))
    
    def _downsample(self, inplanes, outplanes):
        return nn.Sequential(
            nn.Conv3d(inplanes, outplanes, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm3d(outplanes),
        )

    def forward(self, shot, query):
        x = torch.cat((shot, query), dim=0)

        # encoder
        x = x.transpose(1, 2).contiguous() # b, c, d, h, w
        x = self.encoder_freeze(x)

        if self.metric == "cosine" or self.metric == "euclidean":
            x = self.encoder_tune(x).squeeze()
            shot, query = x[:shot.size(0)], x[shot.size(0):]

            # make prototype
            shot = shot.reshape(self.shot, self.way, -1).mean(dim=0)

            # euclidean distance
            if self.metric == "euclidean":
                shot = shot.unsqueeze(0).repeat(self.way*self.query, 1, 1)
                query = query.unsqueeze(1).repeat(1, self.way, 1)
                logits = -((shot - query)**2).sum(dim=-1)
            
            # cosine similarity
            if self.metric == "cosine":
                shot = F.normalize(shot, dim=-1)
                query = F.normalize(query, dim=-1)
                logits = torch.mm(query, shot.t())

        if self.metric == "relation":
            # b, c, d, h, w
            shot, query = x[:shot.size(0)], x[shot.size(0):]
            shot = shot.reshape([self.shot, self.way] + list(shot.size()[1:])).sum(dim=0)
            
            # q, s(way), c, d, h, w
            #        shot  shot  shot
            # query ---o-----x-----x--
            # query ---x-----o-----x--
            # query ---x-----x-----o--

            # change shot shape
            shot = shot.unsqueeze(0).repeat(self.way*self.query, 1, 1, 1, 1, 1)
            shot = shot.reshape([-1] + list(shot.size()[2:]))

            # change query shape
            query = query.unsqueeze(1).repeat(1, self.way, 1, 1, 1, 1)
            query = query.reshape([-1] + list(query.size()[2:]))
            
            relation_pair = torch.cat((shot, query), dim=1) # relation pair (cat by channels)
            logits = self.relation1(relation_pair).squeeze()
            logits = self.relation2(logits)
            logits = logits.reshape(self.way*self.query, self.way)

        return logits * self.scaler

class Resnet(nn.Module):
    def __init__(self, way=5, shot=1, query=5, hidden_size=1024, num_layers=1, bidirectional=True, metric="cosine"):
        super(Resnet, self).__init__()
        self.way = way
        self.shot = shot
        self.query = query
        self.metric = metric

        # resnet18(freezing)
        model = resnet18(pretrained=True)
        self.encoder_freeze = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
        )
        self.encoder_freeze.apply(freeze_all)

        self.last_dim = model.fc.in_features

        # gru
        self.gru = nn.GRU(input_size=self.last_dim, hidden_size=hidden_size, batch_first=Truem, dropout=0.5 if num_layers > 1 else 0, bidirectional=bidirectional)
        # self.gru = nn.GRU(self.last_dim, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)

        # linear
        self.linear = nn.Linear(int(hidden_size*2) if bidirectional else hidden_size, hidden_size)
        self.linear.apply(initialize_linear)

        # scaler
        self.scaler = nn.Parameter(torch.tensor(5.0))

    def forward(self, shot, query):
        x = torch.cat((shot, query), dim=0)
        b, d, c, h, w = x.shape

        # encoder
        x = x.view(b * d, c, h, w)
        x = self.encoder_freeze(x)

        # lstm
        x = x.view(b, d, self.last_dim)
        x = (self.lstm(x)[0]).mean(1) # this may be helful for generalization
        # linear
        x = self.linear(x)
        
        shot, query = x[:shot.size(0)], x[shot.size(0):]

        # make prototype
        shot = shot.reshape(self.shot, self.way, -1).mean(dim=0)

        # cosine similarity
        shot = F.normalize(shot, dim=-1)
        query = F.normalize(query, dim=-1)        
        logits = torch.mm(query, shot.t())

        return logits * self.scaler