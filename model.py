import torch
import torch.nn as nn
import torchvision.models as models

class CNNFeatureExtractor(nn.Module):
    def __init__(self, model_name="resnet34"):
        super(CNNFeatureExtractor, self).__init__()
        self.model = models.resnet34(pretrained=True)
        
        # Unfreeze the last few layers for fine-tuning
        for param in list(self.model.parameters())[:-10]:  
            param.requires_grad = True

        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-2])
        self.feature_dim = 512  # ResNet34 output features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.shape
        x = x.view(batch_size * sequence_length, C, H, W)
        features = self.feature_extractor(x)
        features = self.avgpool(features)
        features = features.view(batch_size, sequence_length, -1)
        return features

class CNNLSTMClassifier(nn.Module):
    def __init__(self, model_name="resnet34", hidden_dim=256, num_classes=10):
        super(CNNLSTMClassifier, self).__init__()
        self.cnn = CNNFeatureExtractor(model_name=model_name)
        self.lstm = nn.LSTM(input_size=self.cnn.feature_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.5)
        self.layer_norm = nn.LayerNorm(hidden_dim)  # Normalization for stability
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        lstm_out, _ = self.lstm(features)
        lstm_out = self.layer_norm(lstm_out[:, -1, :])  # Apply normalization
        out = self.fc(lstm_out)
        return out

# Save Model Checkpoint for Uploading to GitHub
if __name__ == "__main__":
    model = CNNLSTMClassifier()
    torch.save(model.state_dict(), "cnn_lstm_classifier.pth")
