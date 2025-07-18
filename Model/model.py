import torch
import torch.nn as nn

class SeedModel(nn.Module):
    def __init__(self, cnn_backbone, transformer_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.cnn = cnn_backbone  # BirdNET or custom CNN
        for param in self.cnn.parameters():
            param.requires_grad = False  # Freeze CNN at seed stage

        self.projector = nn.Linear(cnn_output_dim, transformer_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=512,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn(x)  # shape: [B, C, H, W]
        x = x.mean(dim=2)  # average over frequency → shape: [B, C, T]
        x = x.permute(0, 2, 1)  # [B, T, C] for Transformer
        x = self.projector(x)  # project to transformer dim
        x = self.transformer(x)  # [B, T, D]
        x = x.mean(dim=1)  # pool across time
        return self.classifier(x).squeeze(1)  # [B]

