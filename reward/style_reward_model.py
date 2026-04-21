import torch
import torch.nn as nn
import torchvision.models as models

class StyleRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights='ResNet34_Weights.IMAGENET1K_V1')
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        

        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img1, img2):
        feat1 = self.feature_extractor(img1)
        feat2 = self.feature_extractor(img2)
        
        combined_features = torch.cat((feat1, feat2), dim=1)
        

        score = self.classifier_head(combined_features)
        
        return score.squeeze(-1)