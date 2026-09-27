"""
Spatial forensic branch using ConvNeXt-Tiny backbone.
Input: Aligned face crops (224x224 RGB)
Output: BranchOutput with 256-dim embedding and binary manipulation logit.
"""


import torch
from torch import nn
from torchvision import models

from deepfake_forensics.branches.contract import BranchOutput


class SpatialBranch(nn.Module):
    """
    Spatial forensic branch using ConvNeXt-Tiny.

    Architecture:
    - Backbone: ConvNeXt-Tiny (pretrained on ImageNet)
    - Backbone output: 768-dim global average pooled features
    - Projection: 768 -> 256 (with LayerNorm + ReLU)
    - Classifier: 256 -> 1 (binary logit for AUTHENTIC vs MANIPULATED)
    """

    def __init__(
        self,
        pretrained: bool = True,
        embedding_dim: int = 256,
        backbone_out_dim: int = 768,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Load ConvNeXt-Tiny backbone
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)

        # Remove the original classifier head
        self.backbone.classifier = nn.Identity()

        # Projection head: 768 -> 256
        self.projection = nn.Sequential(
            nn.LayerNorm(backbone_out_dim),
            nn.Linear(backbone_out_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        # Binary classifier: 256 -> 1 (logit)
        self.classifier = nn.Linear(embedding_dim, 1)

        # Initialize projection and classifier
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection and classifier weights."""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> BranchOutput:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 224, 224) - aligned face crops

        Returns:
            BranchOutput with embedding, evidence_score (logit), and placeholders
        """
        # Backbone features: (B, 768, 1, 1) -> flatten to (B, 768)
        features = self.backbone(x)
        features = features.flatten(1)

        # Project to embedding space: (B, 256)
        embedding = self.projection(features)

        # Binary classification logit: (B, 1)
        evidence_score = self.classifier(embedding)

        batch_size = x.shape[0]
        device = x.device

        # Placeholder reliability and uncertainty (to be implemented in future)
        reliability = torch.ones(batch_size, 1, device=device)
        uncertainty = torch.zeros(batch_size, 1, device=device)

        return BranchOutput(
            embedding=embedding,
            evidence_score=evidence_score,
            reliability=reliability,
            uncertainty=uncertainty,
            attribution=None,
            metadata={"branch": "spatial", "backbone": "convnext_tiny"},
        )

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding without classification."""
        features = self.backbone(x)
        features = features.flatten(1)
        return self.projection(features)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get binary classification logits."""
        embedding = self.get_embedding(x)
        return self.classifier(embedding)
