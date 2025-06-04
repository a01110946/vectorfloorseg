# VectorFloorSeg Implementation - Phase 4: Model Implementation

## Overview

This phase implements the core VectorFloorSeg architecture including the Modulated Graph Attention Network, Two-Stream architecture, and all supporting components. This is the heart of the VectorFloorSeg system.

## Prerequisites

- Completed Phase 1: Environment Setup
- Completed Phase 2: Project Structure Setup
- Completed Phase 3: Data Preparation Pipeline
- Active virtual environment: `vectorfloorseg_env`

## 4.1 Modulated Graph Attention Layer

```python
# File: src/models/modulated_gat.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.typing import Adj, OptTensor, PairTensor
from typing import Optional, Tuple, Union
import math

class ModulatedGATLayer(MessagePassing):
    """
    Modulated Graph Attention Layer for VectorFloorSeg.
    
    This layer implements the key innovation of VectorFloorSeg - using edge features
    from the dual stream to modulate attention computation in the primal stream.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        share_weights: bool = False,
        edge_dim: Optional[int] = None,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights
        self.edge_dim = edge_dim
        
        # NEW CODE: Core linear transformations
        self.lin_query = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_key = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_value = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        if not share_weights:
            self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
            self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # NEW CODE: Modulation network - core innovation
        if edge_dim is not None:
            self.modulation_net = nn.Sequential(
                nn.Linear(edge_dim, heads * out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(heads * out_channels, heads * out_channels * out_channels),
                nn.Sigmoid()  # Ensure positive modulation weights
            )
        else:
            self.modulation_net = None
        
        # NEW CODE: Edge feature processing
        if edge_dim is not None:
            self.edge_proj = nn.Linear(edge_dim, heads, bias=False)
        else:
            self.edge_proj = None
        
        # Standard attention parameters
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Output projection for residual connection
        self.out_proj = nn.Linear(heads * out_channels if concat else out_channels, 
                                 in_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.lin_query.weight, gain=1.414)
        nn.init.xavier_uniform_(self.lin_key.weight, gain=1.414)
        nn.init.xavier_uniform_(self.lin_value.weight, gain=1.414)
        
        if not self.share_weights:
            nn.init.xavier_uniform_(self.lin_src.weight, gain=1.414)
            nn.init.xavier_uniform_(self.lin_dst.weight, gain=1.414)
        
        nn.init.xavier_uniform_(self.att_src, gain=1.414)
        nn.init.xavier_uniform_(self.att_dst, gain=1.414)
        
        if self.edge_proj is not None:
            nn.init.xavier_uniform_(self.edge_proj.weight, gain=1.414)
        
        if self.modulation_net is not None:
            for module in self.modulation_net:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=1.414)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: Union[torch.Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        dual_edge_features: OptTensor = None,
        size: Optional[Tuple[int, int]] = None,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with modulated attention.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge features from current stream
            dual_edge_features: Edge features from dual stream (for modulation)
            size: Size of source and target nodes
            return_attention_weights: Whether to return attention weights
        """
        H, C = self.heads, self.out_channels
        
        # Handle input tensor(s)
        if isinstance(x, torch.Tensor):
            x_src = x_dst = x
        else:
            x_src, x_dst = x
        
        # Linear transformations
        query = self.lin_query(x_dst).view(-1, H, C)
        key = self.lin_key(x_src).view(-1, H, C)
        value = self.lin_value(x_src).view(-1, H, C)
        
        # Add self-loops if specified
        if self.add_self_loops:
            if isinstance(edge_index, torch.Tensor):
                # For SparseTensor, we need to use set_diag
                num_nodes = x_src.size(0)
                if x_dst is not x_src:
                    num_nodes = max(num_nodes, x_dst.size(0))
                
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, num_nodes=num_nodes
                )
                
                # Handle dual edge features for self-loops
                if dual_edge_features is not None:
                    # Add identity features for self-loops
                    num_self_loops = num_nodes
                    if dual_edge_features.dim() == 2:
                        self_loop_features = torch.zeros(
                            num_self_loops, dual_edge_features.size(1),
                            device=dual_edge_features.device,
                            dtype=dual_edge_features.dtype
                        )
                        dual_edge_features = torch.cat([dual_edge_features, self_loop_features], dim=0)
        
        # Propagate messages
        out, attention_weights = self.propagate(
            edge_index, 
            query=query, 
            key=key, 
            value=value,
            edge_attr=edge_attr,
            dual_edge_features=dual_edge_features,
            size=size,
            return_attention_weights=return_attention_weights
        )
        
        # Concatenate or average attention heads
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        # Add bias
        if self.bias is not None:
            out += self.bias
        
        # Residual connection
        out = self.out_proj(out) + x_dst
        
        if return_attention_weights:
            return out, attention_weights
        else:
            return out
    
    def message(
        self, 
        query_i: torch.Tensor,
        key_j: torch.Tensor, 
        value_j: torch.Tensor,
        edge_attr: OptTensor = None,
        dual_edge_features: OptTensor = None,
        index: torch.Tensor = None,
        ptr: OptTensor = None,
        size_i: Optional[int] = None,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """
        Compute messages with modulated attention mechanism.
        
        This is where the core innovation happens - dual edge features
        modulate the attention computation.
        """
        # Basic attention computation
        alpha = (query_i * key_j).sum(dim=-1)  # [num_edges, heads]
        
        # NEW CODE: Apply modulation from dual stream
        if dual_edge_features is not None and self.modulation_net is not None:
            # Generate modulation weights
            modulation_weights = self.modulation_net(dual_edge_features)
            # Reshape: [num_edges, heads * out_channels * out_channels]
            modulation_weights = modulation_weights.view(-1, self.heads, self.out_channels, self.out_channels)
            
            # Apply modulation to query-key interaction
            # This is the key insight: dual edge features influence attention
            query_modulated = torch.einsum('ehd,ehdc->ehc', query_i, modulation_weights)
            alpha = (query_modulated * key_j).sum(dim=-1)
        
        # Add edge features to attention if available
        if edge_attr is not None and self.edge_proj is not None:
            edge_contribution = self.edge_proj(edge_attr)
            alpha = alpha + edge_contribution
        
        # Apply leaky ReLU and softmax
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply attention to values
        out = alpha.unsqueeze(-1) * value_j
        
        if return_attention_weights:
            self._alpha = alpha
        
        return out
    
    def aggregate(
        self, 
        inputs: torch.Tensor, 
        index: torch.Tensor,
        ptr: OptTensor = None,
        dim_size: Optional[int] = None
    ) -> torch.Tensor:
        """Aggregate messages."""
        return super().aggregate(inputs, index, ptr, dim_size)

class MultiHeadModulatedGAT(nn.Module):
    """Multi-head version of Modulated GAT with layer normalization."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_in_channels = hidden_channels
            layer_out_channels = hidden_channels if i < num_layers - 1 else out_channels
            layer_heads = num_heads if i < num_layers - 1 else 1
            layer_concat = True if i < num_layers - 1 else False
            
            self.gat_layers.append(
                ModulatedGATLayer(
                    layer_in_channels,
                    layer_out_channels // layer_heads if layer_concat else layer_out_channels,
                    heads=layer_heads,
                    concat=layer_concat,
                    dropout=dropout,
                    edge_dim=edge_dim
                )
            )
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_channels if i < num_layers - 1 else out_channels)
                for i in range(num_layers)
            ])
        
        # Output projection
        self.output_proj = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: OptTensor = None,
        dual_edge_features: OptTensor = None,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """Forward pass through multiple GAT layers."""
        
        # Input projection
        x = self.input_proj(x)
        
        attention_weights = []
        
        for i, gat_layer in enumerate(self.gat_layers):
            # Apply GAT layer
            if return_attention_weights and i == len(self.gat_layers) - 1:
                x, attn_weights = gat_layer(
                    x, edge_index, edge_attr, dual_edge_features, 
                    return_attention_weights=True
                )
                attention_weights.append(attn_weights)
            else:
                x = gat_layer(x, edge_index, edge_attr, dual_edge_features)
            
            # Apply layer normalization
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            
            # Apply dropout (except last layer)
            if i < len(self.gat_layers) - 1:
                x = self.dropout(x)
        
        # Output projection
        x = self.output_proj(x)
        
        if return_attention_weights:
            return x, attention_weights
        else:
            return x
```

## 4.2 Two-Stream Architecture

```python
# File: src/models/two_stream_gnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import Dict, Tuple, Optional, List
import logging

from .modulated_gat import MultiHeadModulatedGAT
from .backbone import get_backbone_model

class TwoStreamGNN(nn.Module):
    """
    Two-Stream Graph Neural Network for VectorFloorSeg.
    
    This implements the core architecture with:
    1. Primal stream: processes line segments (edges as entities)
    2. Dual stream: processes regions (regions as entities)
    3. Cross-stream modulation via edge correspondence
    """
    
    def __init__(
        self,
        # Input dimensions
        primal_input_dim: int = 66,
        dual_input_dim: int = 66,
        edge_feature_dim: int = 128,
        
        # Architecture
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        
        # Task-specific
        num_room_classes: int = 12,
        
        # Backbone for image features
        backbone_name: str = "resnet101",
        use_backbone: bool = True,
        backbone_pretrained: bool = True,
        
        # Training
        boundary_loss_weight: float = 0.5
    ):
        super().__init__()
        
        self.primal_input_dim = primal_input_dim
        self.dual_input_dim = dual_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_room_classes = num_room_classes
        self.boundary_loss_weight = boundary_loss_weight
        self.use_backbone = use_backbone
        
        self.logger = logging.getLogger(__name__)
        
        # NEW CODE: Backbone CNN for image features
        if use_backbone:
            self.backbone = get_backbone_model(
                backbone_name, 
                pretrained=backbone_pretrained,
                feature_dim=hidden_dim // 4  # Reduce backbone feature contribution
            )
            backbone_feature_dim = hidden_dim // 4
        else:
            self.backbone = None
            backbone_feature_dim = 0
        
        # NEW CODE: Input embedding layers
        self.primal_input_embedding = nn.Sequential(
            nn.Linear(primal_input_dim + backbone_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.dual_input_embedding = nn.Sequential(
            nn.Linear(dual_input_dim + backbone_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # NEW CODE: Edge feature embedding
        self.primal_edge_embedding = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.dual_edge_embedding = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # NEW CODE: Two-stream GAT networks
        self.primal_stream = MultiHeadModulatedGAT(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            edge_dim=hidden_dim  # For dual edge modulation
        )
        
        self.dual_stream = MultiHeadModulatedGAT(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            edge_dim=hidden_dim  # For primal edge modulation
        )
        
        # NEW CODE: Edge update networks
        self.primal_edge_update = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),  # node_i + node_j + edge_feat
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.dual_edge_update = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # NEW CODE: Task-specific heads
        self.boundary_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary: boundary or not
        )
        
        self.room_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_room_classes)
        )
        
        # NEW CODE: Edge correspondence mapping
        self.edge_correspondence_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        primal_data: Data,
        dual_data: Data,
        image: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through two-stream network.
        
        Args:
            primal_data: Primal graph (line segments)
            dual_data: Dual graph (regions)
            image: Optional rasterized floorplan image for backbone features
            return_attention: Whether to return attention weights
        """
        
        batch_size = primal_data.batch.max().item() + 1 if primal_data.batch is not None else 1
        
        # NEW CODE: Extract backbone features if available
        backbone_features = None
        if self.use_backbone and self.backbone is not None and image is not None:
            backbone_features = self.backbone(image)  # [batch_size, feature_dim, H, W]
        
        # NEW CODE: Embed input features
        primal_node_features = self._embed_primal_features(
            primal_data.x, primal_data.pos, backbone_features, primal_data.batch
        )
        dual_node_features = self._embed_dual_features(
            dual_data.x, dual_data.pos, backbone_features, dual_data.batch
        )
        
        # Embed edge features
        primal_edge_features = self.primal_edge_embedding(primal_data.edge_attr)
        dual_edge_features = self.dual_edge_embedding(dual_data.edge_attr)
        
        # NEW CODE: Cross-stream processing with modulation
        # In each iteration, features from one stream modulate the other
        for layer_idx in range(self.num_layers):
            
            # Store previous features for residual connections
            primal_prev = primal_node_features
            dual_prev = dual_node_features
            
            # Process primal stream with dual edge modulation
            primal_node_features = self.primal_stream.gat_layers[layer_idx](
                primal_node_features,
                primal_data.edge_index,
                edge_attr=primal_edge_features,
                dual_edge_features=dual_edge_features  # Cross-stream modulation
            )
            
            # Process dual stream with primal edge modulation  
            dual_node_features = self.dual_stream.gat_layers[layer_idx](
                dual_node_features,
                dual_data.edge_index,
                edge_attr=dual_edge_features,
                dual_edge_features=primal_edge_features  # Cross-stream modulation
            )
            
            # Apply layer normalization if available
            if hasattr(self.primal_stream, 'layer_norms'):
                primal_node_features = self.primal_stream.layer_norms[layer_idx](primal_node_features)
            if hasattr(self.dual_stream, 'layer_norms'):
                dual_node_features = self.dual_stream.layer_norms[layer_idx](dual_node_features)
            
            # Update edge features based on updated node features
            primal_edge_features = self._update_edge_features(
                primal_node_features, primal_data.edge_index, primal_edge_features, 
                self.primal_edge_update
            )
            dual_edge_features = self._update_edge_features(
                dual_node_features, dual_data.edge_index, dual_edge_features,
                self.dual_edge_update
            )
        
        # NEW CODE: Generate predictions
        # Boundary classification on primal edges
        boundary_logits = None
        if primal_edge_features.size(0) > 0:
            boundary_logits = self.boundary_classifier(primal_edge_features)
        
        # Room classification on dual nodes
        room_logits = None
        if dual_node_features.size(0) > 0:
            room_logits = self.room_classifier(dual_node_features)
        
        results = {
            'boundary_logits': boundary_logits,
            'room_logits': room_logits,
            'primal_features': primal_node_features,
            'dual_features': dual_node_features,
            'primal_edge_features': primal_edge_features,
            'dual_edge_features': dual_edge_features
        }
        
        if return_attention:
            # Get attention weights from final layer
            with torch.no_grad():
                _, primal_attention = self.primal_stream.gat_layers[-1](
                    primal_node_features, primal_data.edge_index,
                    edge_attr=primal_edge_features, dual_edge_features=dual_edge_features,
                    return_attention_weights=True
                )
                _, dual_attention = self.dual_stream.gat_layers[-1](
                    dual_node_features, dual_data.edge_index,
                    edge_attr=dual_edge_features, dual_edge_features=primal_edge_features,
                    return_attention_weights=True
                )
            
            results.update({
                'primal_attention': primal_attention,
                'dual_attention': dual_attention
            })
        
        return results
    
    def _embed_primal_features(
        self, 
        node_features: torch.Tensor,
        node_positions: torch.Tensor,
        backbone_features: Optional[torch.Tensor],
        batch: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Embed primal node features with optional backbone features."""
        
        # Combine geometric and backbone features
        if backbone_features is not None and self.use_backbone:
            # Sample backbone features at node positions
            backbone_node_features = self._sample_backbone_features(
                backbone_features, node_positions, batch
            )
            combined_features = torch.cat([node_features, backbone_node_features], dim=-1)
        else:
            combined_features = node_features
        
        return self.primal_input_embedding(combined_features)
    
    def _embed_dual_features(
        self, 
        node_features: torch.Tensor,
        node_positions: torch.Tensor,
        backbone_features: Optional[torch.Tensor],
        batch: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Embed dual node features with optional backbone features."""
        
        # Combine geometric and backbone features  
        if backbone_features is not None and self.use_backbone:
            # Sample backbone features at node positions
            backbone_node_features = self._sample_backbone_features(
                backbone_features, node_positions, batch
            )
            combined_features = torch.cat([node_features, backbone_node_features], dim=-1)
        else:
            combined_features = node_features
        
        return self.dual_input_embedding(combined_features)
    
    def _sample_backbone_features(
        self,
        backbone_features: torch.Tensor,  # [batch_size, C, H, W]
        positions: torch.Tensor,          # [num_nodes, 2]
        batch: Optional[torch.Tensor]     # [num_nodes]
    ) -> torch.Tensor:
        """Sample backbone features at given positions."""
        
        if batch is None:
            batch = torch.zeros(positions.size(0), dtype=torch.long, device=positions.device)
        
        batch_size, C, H, W = backbone_features.shape
        num_nodes = positions.size(0)
        
        # Normalize positions to [-1, 1] for grid_sample
        normalized_positions = positions.clone()
        normalized_positions[:, 0] = 2.0 * positions[:, 0] / 255.0 - 1.0  # x
        normalized_positions[:, 1] = 2.0 * positions[:, 1] / 255.0 - 1.0  # y
        
        # Sample features for each node
        sampled_features = torch.zeros(num_nodes, C, device=positions.device)
        
        for batch_idx in range(batch_size):
            # Find nodes in this batch
            node_mask = (batch == batch_idx)
            if not node_mask.any():
                continue
            
            batch_positions = normalized_positions[node_mask]  # [num_batch_nodes, 2]
            
            # Reshape for grid_sample: [1, 1, num_batch_nodes, 2]
            grid = batch_positions.unsqueeze(0).unsqueeze(0)
            
            # Sample features: [1, C, 1, num_batch_nodes]
            batch_backbone = backbone_features[batch_idx:batch_idx+1]  # [1, C, H, W]
            sampled = F.grid_sample(
                batch_backbone, grid, 
                mode='bilinear', padding_mode='border', align_corners=False
            )
            
            # Extract features: [C, num_batch_nodes] -> [num_batch_nodes, C]
            sampled_features[node_mask] = sampled.squeeze(0).squeeze(1).t()
        
        return sampled_features
    
    def _update_edge_features(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        update_network: nn.Module
    ) -> torch.Tensor:
        """Update edge features based on incident node features."""
        
        if edge_index.size(1) == 0:
            return edge_features
        
        row, col = edge_index
        
        # Combine source node, target node, and current edge features
        edge_input = torch.cat([
            node_features[row],      # Source node features
            node_features[col],      # Target node features  
            edge_features            # Current edge features
        ], dim=-1)
        
        # Update edge features
        updated_features = update_network(edge_input)
        
        # Residual connection
        return edge_features + updated_features

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        boundary_labels: Optional[torch.Tensor] = None,
        room_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss."""
        
        losses = {}
        total_loss = 0.0
        
        # Boundary classification loss
        if predictions['boundary_logits'] is not None and boundary_labels is not None:
            boundary_loss = F.cross_entropy(
                predictions['boundary_logits'],
                boundary_labels,
                ignore_index=-1
            )
            losses['boundary_loss'] = boundary_loss
            total_loss += self.boundary_loss_weight * boundary_loss
        
        # Room classification loss (focal loss for class imbalance)
        if predictions['room_logits'] is not None and room_labels is not None:
            room_loss = self._focal_loss(
                predictions['room_logits'],
                room_labels,
                alpha=0.25,
                gamma=2.0
            )
            losses['room_loss'] = room_loss
            total_loss += room_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def _focal_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0
    ) -> torch.Tensor:
        """Focal loss for handling class imbalance."""
        
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        
        return focal_loss.mean()
```

## 4.3 Backbone Networks

```python
# File: src/models/backbone.py
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from typing import Optional

def get_backbone_model(
    backbone_name: str = "resnet101",
    pretrained: bool = True,
    feature_dim: int = 64,
    pretrained_path: Optional[str] = None
) -> nn.Module:
    """
    Get backbone CNN model for extracting image features.
    
    Args:
        backbone_name: Name of backbone ('resnet50', 'resnet101', 'vgg16')
        pretrained: Whether to use pretrained weights
        feature_dim: Output feature dimension
        pretrained_path: Path to custom pretrained weights
    """
    
    if backbone_name.startswith('resnet'):
        return ResNetBackbone(backbone_name, pretrained, feature_dim, pretrained_path)
    elif backbone_name.startswith('vgg'):
        return VGGBackbone(backbone_name, pretrained, feature_dim, pretrained_path)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

class ResNetBackbone(nn.Module):
    """ResNet backbone for image feature extraction."""
    
    def __init__(
        self,
        backbone_name: str = "resnet101",
        pretrained: bool = True,
        feature_dim: int = 64,
        pretrained_path: Optional[str] = None
    ):
        super().__init__()
        
        # Get ResNet model
        if backbone_name == "resnet50":
            self.resnet = models.resnet50(pretrained=False)
            backbone_dim = 2048
        elif backbone_name == "resnet101":
            self.resnet = models.resnet101(pretrained=False)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet variant: {backbone_name}")
        
        # Load pretrained weights
        if pretrained:
            if pretrained_path:
                # Load custom pretrained weights
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                self.resnet.load_state_dict(checkpoint, strict=False)
            else:
                # Load ImageNet pretrained weights
                if backbone_name == "resnet50":
                    self.resnet = models.resnet50(pretrained=True)
                elif backbone_name == "resnet101":
                    self.resnet = models.resnet101(pretrained=True)
        
        # Remove final classification layers
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Conv2d(backbone_dim, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Feature pooling for final output
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet backbone.
        
        Args:
            x: Input image tensor [batch_size, 3, H, W]
            
        Returns:
            Feature maps [batch_size, feature_dim, H//32, W//32]
        """
        # Ensure input has 3 channels
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Extract features
        features = self.resnet(x)  # [batch_size, 2048, H//32, W//32]
        
        # Project to desired feature dimension
        features = self.feature_proj(features)  # [batch_size, feature_dim, H//32, W//32]
        
        return features

class VGGBackbone(nn.Module):
    """VGG backbone for image feature extraction."""
    
    def __init__(
        self,
        backbone_name: str = "vgg16",
        pretrained: bool = True,
        feature_dim: int = 64,
        pretrained_path: Optional[str] = None
    ):
        super().__init__()
        
        # Get VGG model
        if backbone_name == "vgg16":
            self.vgg = models.vgg16(pretrained=False)
            backbone_dim = 512
        else:
            raise ValueError(f"Unsupported VGG variant: {backbone_name}")
        
        # Load pretrained weights
        if pretrained:
            if pretrained_path:
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                self.vgg.load_state_dict(checkpoint, strict=False)
            else:
                if backbone_name == "vgg16":
                    self.vgg = models.vgg16(pretrained=True)
        
        # Use only feature extraction part
        self.features = self.vgg.features
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Conv2d(backbone_dim, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through VGG backbone."""
        
        # Ensure input has 3 channels
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Extract features
        features = self.features(x)  # [batch_size, 512, H//32, W//32]
        
        # Project to desired feature dimension
        features = self.feature_proj(features)  # [batch_size, feature_dim, H//32, W//32]
        
        return features
```

## 4.4 Model Factory and Utilities

```python
# File: src/models/model_factory.py
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

from .two_stream_gnn import TwoStreamGNN
from ..utils.config import ModelConfig

def create_vectorfloorseg_model(
    config: ModelConfig,
    device: torch.device = None
) -> TwoStreamGNN:
    """Create VectorFloorSeg model from configuration."""
    
    logger = logging.getLogger(__name__)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = TwoStreamGNN(
        primal_input_dim=config.primal_input_dim,
        dual_input_dim=config.dual_input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        num_room_classes=config.num_room_classes,
        backbone_name=config.backbone,
        use_backbone=True,
        backbone_pretrained=True
    )
    
    model = model.to(device)
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Created VectorFloorSeg model:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    logger.info(f"  Device: {device}")
    
    return model

def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device = None,
    strict: bool = True
) -> Dict[str, Any]:
    """Load model from checkpoint."""
    
    logger = logging.getLogger(__name__)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Loading model checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        logger.info(f"Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        
        return checkpoint
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

def save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_val_loss: float,
    checkpoint_path: str,
    additional_info: Dict[str, Any] = None
):
    """Save model checkpoint."""
    
    logger = logging.getLogger(__name__)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'model_config': {
            'class_name': model.__class__.__name__,
            'config': model.__dict__ if hasattr(model, '__dict__') else {}
        }
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if additional_info:
        checkpoint.update(additional_info)
    
    try:
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to: {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        raise

class ModelEMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        self.register()
    
    def register(self):
        """Register model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
```

## 4.5 Model Testing Script

```python
# File: test_model.py
"""Test script for VectorFloorSeg model components."""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.model_factory import create_vectorfloorseg_model
from src.models.two_stream_gnn import TwoStreamGNN
from src.utils.config import ModelConfig, ExperimentConfig
from src.data.svg_processor import SVGFloorplanProcessor
from torch_geometric.data import Data

def test_model_creation():
    """Test model creation and forward pass."""
    
    print("Testing VectorFloorSeg model creation...")
    
    # Create configuration
    config = ModelConfig(
        primal_input_dim=66,
        dual_input_dim=66,
        hidden_dim=128,  # Smaller for testing
        num_layers=2,    # Fewer layers for testing
        num_heads=4,
        dropout=0.1,
        num_room_classes=12,
        backbone="resnet50"  # Lighter backbone for testing
    )
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_vectorfloorseg_model(config, device)
    
    print(f"✓ Model created successfully on {device}")
    
    return model, device

def create_dummy_data(device):
    """Create dummy graph data for testing."""
    
    # Dummy primal graph (line segments)
    primal_x = torch.randn(10, 66).to(device)  # 10 nodes, 66 features
    primal_edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ], dtype=torch.long).to(device)
    primal_edge_attr = torch.randn(9, 128).to(device)  # 9 edges, 128 features
    primal_pos = torch.rand(10, 2).to(device) * 255  # Random positions
    
    primal_data = Data(
        x=primal_x,
        edge_index=primal_edge_index,
        edge_attr=primal_edge_attr,
        pos=primal_pos
    )
    
    # Dummy dual graph (regions)
    dual_x = torch.randn(5, 66).to(device)  # 5 regions, 66 features
    dual_edge_index = torch.tensor([
        [0, 1, 2, 3],
        [1, 2, 3, 4]
    ], dtype=torch.long).to(device)
    dual_edge_attr = torch.randn(4, 128).to(device)  # 4 edges, 128 features
    dual_pos = torch.rand(5, 2).to(device) * 255  # Random positions
    
    dual_data = Data(
        x=dual_x,
        edge_index=dual_edge_index,
        edge_attr=dual_edge_attr,
        pos=dual_pos
    )
    
    # Dummy image
    image = torch.randn(1, 3, 256, 256).to(device)
    
    return primal_data, dual_data, image

def test_forward_pass(model, device):
    """Test forward pass through the model."""
    
    print("\nTesting forward pass...")
    
    # Create dummy data
    primal_data, dual_data, image = create_dummy_data(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(primal_data, dual_data, image)
    
    print(f"✓ Forward pass successful")
    print(f"  Boundary logits shape: {outputs['boundary_logits'].shape if outputs['boundary_logits'] is not None else 'None'}")
    print(f"  Room logits shape: {outputs['room_logits'].shape if outputs['room_logits'] is not None else 'None'}")
    
    return outputs

def test_loss_computation(model, device):
    """Test loss computation."""
    
    print("\nTesting loss computation...")
    
    # Create dummy data and labels
    primal_data, dual_data, image = create_dummy_data(device)
    
    # Dummy labels
    boundary_labels = torch.randint(0, 2, (9,)).to(device)  # Binary labels for 9 edges
    room_labels = torch.randint(0, 12, (5,)).to(device)     # Room labels for 5 regions
    
    # Forward pass
    model.train()
    outputs = model(primal_data, dual_data, image)
    
    # Compute loss
    losses = model.compute_loss(outputs, boundary_labels, room_labels)
    
    print(f"✓ Loss computation successful")
    print(f"  Total loss: {losses['total_loss'].item():.4f}")
    if 'boundary_loss' in losses:
        print(f"  Boundary loss: {losses['boundary_loss'].item():.4f}")
    if 'room_loss' in losses:
        print(f"  Room loss: {losses['room_loss'].item():.4f}")
    
    return losses

def test_svg_to_model_pipeline():
    """Test the complete pipeline from SVG to model prediction."""
    
    print("\nTesting SVG to model pipeline...")
    
    # Create simple test SVG
    test_svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
    <svg width="256" height="256" xmlns="http://www.w3.org/2000/svg">
        <rect x="50" y="50" width="100" height="100" fill="none" stroke="black" stroke-width="2"/>
        <line x1="100" y1="50" x2="100" y2="150" stroke="black" stroke-width="2"/>
    </svg>'''
    
    test_svg_path = Path("test_pipeline.svg")
    with open(test_svg_path, 'w') as f:
        f.write(test_svg_content)
    
    try:
        # Process SVG
        processor = SVGFloorplanProcessor()
        result = processor.process_svg_to_graphs(str(test_svg_path))
        
        print(f"✓ SVG processed successfully")
        print(f"  Primal graph: {result['primal'].x.shape[0]} nodes, {result['primal'].edge_index.shape[1]} edges")
        print(f"  Dual graph: {result['dual'].x.shape[0]} nodes, {result['dual'].edge_index.shape[1]} edges")
        
        # Create model and test
        config = ModelConfig(hidden_dim=64, num_layers=1)  # Minimal for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_vectorfloorseg_model(config, device)
        
        # Move data to device
        primal_data = result['primal'].to(device)
        dual_data = result['dual'].to(device)
        
        # Create dummy image
        image = torch.randn(1, 3, 256, 256).to(device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(primal_data, dual_data, image)
        
        print(f"✓ End-to-end pipeline successful")
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        raise
    finally:
        # Clean up
        if test_svg_path.exists():
            test_svg_path.unlink()

def main():
    """Run all model tests."""
    
    print("=" * 50)
    print("VectorFloorSeg Model Testing")
    print("=" * 50)
    
    try:
        # Test model creation
        model, device = test_model_creation()
        
        # Test forward pass
        outputs = test_forward_pass(model, device)
        
        # Test loss computation
        losses = test_loss_computation(model, device)
        
        # Test complete pipeline
        test_svg_to_model_pipeline()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed successfully!")
        print("✓ Model is ready for training")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        print("Please check the error and fix before proceeding to training.")
        raise

if __name__ == "__main__":
    main()
```

## Usage Instructions

### Test Model Implementation
```bash
cd VecFloorSeg
source vectorfloorseg_env/bin/activate

# Test all model components
python test_model.py

# Expected output: All tests should pass
```

### Create and Test Model
```python
from src.models.model_factory import create_vectorfloorseg_model
from src.utils.config import ModelConfig

# Create model configuration
config = ModelConfig(
    hidden_dim=256,
    num_layers=6,
    num_heads=8,
    backbone="resnet101"
)

# Create model
model = create_vectorfloorseg_model(config)
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
```

### Integration with Training
```python
# In training script
from src.models.model_factory import create_vectorfloorseg_model, save_model_checkpoint

# Create model
model = create_vectorfloorseg_model(config)

# Training loop
for epoch in range(num_epochs):
    # ... training code ...
    
    # Save checkpoint
    save_model_checkpoint(
        model, optimizer, scheduler, epoch, best_val_loss,
        f"checkpoints/model_epoch_{epoch}.pth"
    )
```

## Key Features Implemented

1. **Modulated GAT Layer**: Core innovation using dual edge features to modulate attention
2. **Two-Stream Architecture**: Separate processing of primal (lines) and dual (regions) graphs
3. **Cross-Stream Communication**: Information exchange between streams via edge correspondence
4. **Multi-Task Learning**: Joint boundary and room classification
5. **Backbone Integration**: CNN features from rasterized images
6. **Flexible Configuration**: Configurable architecture parameters
7. **Model Management**: Checkpoint saving/loading, EMA, factory functions

## Next Steps

Proceed to **Phase 5: Training Script** to implement the complete training pipeline.

## Troubleshooting

- **CUDA memory issues**: Reduce `hidden_dim` or `batch_size`
- **Attention computation errors**: Check edge correspondence mapping
- **Backbone loading errors**: Verify pretrained model paths
- **Graph construction issues**: Ensure valid edge indices and features
