import torch
import torch.nn as nn
import torch.nn.functional as F

#models
class MLP(nn.Module):
    def __init__(self, input_dim, h1, h2, output_dim):
        """
        Args:
            input_dim (int): number of input features
            hidden_dim (int): number of hidden units
            output_dim (int): number of output units
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(h2, output_dim),
        )

    def forward(self, x):
        # return F.log_softmax(self.layers(x), dim=1)
        try:
            return self.layers(x)
        except:
            x = x.to(torch.float32)
            return self.layers(x)

class MLPSoftmax(MLP):
    def __init__(self, input_dim, h1, h2, output_dim):
        super().__init__(input_dim, h1, h2, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = super().forward(x)  # Get the output from the parent MLP class
        x = self.softmax(x)     # Apply the softmax layer
        return x


class ResNet(nn.Module):
    def __init__(self,):
        super().__init__()
        # Load the pretrained ResNet-18 model
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

        # Remove the last fully connected layer to get features
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        return self.feature_extractor(x)

class CombinedModel(nn.Module):
    def __init__(self, resnet_feature_extractor, mlp_mocap, mlp_mocap_output_dim,
                 mlp_footp, mlp_footp_output_dim, num_classes):
        super().__init__()

        self.resnet_feature_extractor = resnet_feature_extractor
        self.mlp_mocap = mlp_mocap
        self.mlp_footp = mlp_footp

        input_dim = 0
        if resnet_feature_extractor is not None:
            input_dim += 512
        if mlp_mocap is not None:
            input_dim += mlp_mocap_output_dim
        if mlp_footp is not None:
            input_dim += mlp_footp_output_dim

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.bn3 = nn.BatchNorm1d(num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image_input, mlp_mocap_input, mlp_footp_input):
        features = []
        if self.resnet_feature_extractor is not None:
            resnet_features = self.resnet_feature_extractor(image_input)
            resnet_features = resnet_features.view(resnet_features.size(0), -1)
            features.append(resnet_features)

        if self.mlp_mocap is not None:
            mlp_mocap_features = self.mlp_mocap(mlp_mocap_input)
            features.append(mlp_mocap_features)

        if self.mlp_footp is not None:
            mlp_footp_features = self.mlp_footp(mlp_footp_input)
            features.append(mlp_footp_features)

        combined_features = torch.cat(features, dim=1)

        x = self.relu(self.bn1(self.fc1(combined_features)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))
        x = self.softmax(x)
        
        return x

# class CombinedModel(nn.Module):
#     def __init__(self, resnet_feature_extractor, mlp_mocap, mlp_mocap_output_dim, mlp_footp, mlp_footp_output_dim, num_classes):
#         super().__init__()
        
#         self.resnet_feature_extractor = resnet_feature_extractor
#         self.mlp_mocap = mlp_mocap
#         self.mlp_footp = mlp_footp
        
#         self.fc1 = nn.Linear(512 + mlp_mocap_output_dim + mlp_footp_output_dim, 256)
#         self.bn1 = nn.BatchNorm1d(256)  
#         self.fc2 = nn.Linear(256, 128)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.fc3 = nn.Linear(128, num_classes)
#         self.bn3 = nn.BatchNorm1d(num_classes)

#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, image_input, mlp_mocap_input, mlp_footp_input):
#         resnet_features = self.resnet_feature_extractor(image_input)
#         resnet_features = resnet_features.view(resnet_features.size(0), -1)  
        
#         mlp_mocap_features = self.mlp_mocap(mlp_mocap_input)
#         mlp_footp_features = self.mlp_footp(mlp_footp_input)
        
#         combined_features = torch.cat((resnet_features, mlp_mocap_features, mlp_footp_features), dim=1)
        
#         x = self.relu(self.bn1(self.fc1(combined_features)))  
#         x = self.relu(self.bn2(self.fc2(x)))  
#         x = self.bn3(self.fc3(x))
#         x = self.softmax(x)  # Apply softmax at the end
        
#         return x

# class CombinedModel(nn.Module):
#     def __init__(self, resnet_feature_extractor, mlp_mocap, mlp_mocap_output_dim, mlp_footp, mlp_footp_output_dim, num_classes):
#         super().__init__()
        
#         self.resnet_feature_extractor = resnet_feature_extractor
#         self.mlp_mocap = mlp_mocap
#         self.mlp_footp = mlp_footp
        
#         self.fc1 = nn.Linear(512 + mlp_mocap_output_dim + mlp_footp_output_dim, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, num_classes)

#         self.relu = nn.ReLU()
        
#     def forward(self, image_input, mlp_mocap_input,mlp_footp_input):
#         resnet_features = self.resnet_feature_extractor(image_input)
#         resnet_features = resnet_features.view(resnet_features.size(0), -1)  # Flatten the features
        
#         mlp_mocap_features = self.mlp_mocap(mlp_mocap_input)
#         mlp_footp_features = self.mlp_footp(mlp_footp_input)
        
#         combined_features = torch.cat((resnet_features, mlp_mocap_features, mlp_footp_features), dim=1)
        
#         x = self.relu(self.fc1(combined_features))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
        
#         return x
