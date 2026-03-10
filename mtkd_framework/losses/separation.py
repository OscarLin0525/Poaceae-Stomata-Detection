"""
Separation Loss for enforcing spatial gaps between adjacent stomata features.

Encourages valleys (low-similarity regions) between GT stomata centers in the feature space,
preventing high-response regions from merging together.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ValleySeparationLoss(nn.Module):
    """
    Encourages spatial separation between GT stomata by enforcing valleys
    (high contrast values) along paths connecting adjacent centers.
    
    For each pair of GT stomata centers:
      1. Sample points along the connecting line
      2. Compute contrast values (distance to nearest GT feature)
      3. Penalize if midpoint contrast is as low as endpoints
         (i.e., no valley exists)
    
    Args:
        sample_points: Number of points to sample along each connecting line
        valley_margin: Margin factor for valley depth (midpoint should be higher by this factor)
    """
    
    def __init__(self, sample_points: int = 5, valley_margin: float = 0.2):
        super().__init__()
        self.sample_points = sample_points
        self.valley_margin = valley_margin
    
    def forward(
        self,
        features: torch.Tensor,
        gt_centers_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            features: [B, H, W, C] extracted layer features
            gt_centers_list: List of [N, 2] tensors containing (row, col) for each image
                            Each tensor contains N GT stomata centers for that image
        
        Returns:
            loss: Scalar separation loss (higher when stomata are not separated)
        """
        B, H, W, C = features.shape
        device = features.device
        
        total_loss = 0.0
        total_pairs = 0
        
        for b in range(B):
            feat = features[b]  # [H, W, C]
            gt_centers = gt_centers_list[b]  # [N, 2]
            
            if len(gt_centers) < 2:
                continue  # Need at least 2 stomata to measure separation
            
            # Compute contrast map for this image
            # Contrast = L2 distance to nearest GT feature (lower = more similar)
            contrast_map = self._compute_contrast_map(feat, gt_centers)  # [H, W]
            
            # For each pair of GT centers
            N = len(gt_centers)
            for i in range(N):
                for j in range(i + 1, N):
                    r1, c1 = gt_centers[i]
                    r2, c2 = gt_centers[j]
                    
                    # Sample points along line connecting the two centers
                    line_coords = self._sample_line(r1, c1, r2, c2, self.sample_points)  # [P, 2]
                    
                    # Get contrast values at these points
                    line_values = contrast_map[line_coords[:, 0], line_coords[:, 1]]  # [P]
                    
                    # Endpoint values (should be low - similar to GT)
                    endpoint_contrast = contrast_map[[r1, r2], [c1, c2]]  # [2]
                    endpoint_avg = endpoint_contrast.mean()
                    
                    # Midpoint and intermediate values (should be high - valley)
                    mid_idx = self.sample_points // 2
                    mid_contrast = line_values[mid_idx]
                    
                    # Loss: penalize if midpoint is not sufficiently higher than endpoints
                    # We want: mid_contrast > endpoint_avg * (1 + valley_margin)
                    target_mid = endpoint_avg * (1.0 + self.valley_margin)
                    valley_loss = F.relu(target_mid - mid_contrast)
                    
                    total_loss += valley_loss
                    total_pairs += 1
        
        if total_pairs == 0:
            return torch.tensor(0.0, device=device)
        
        return total_loss / total_pairs
    
    def _compute_contrast_map(
        self,
        features: torch.Tensor,
        gt_centers: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrast map = min L2 distance to any GT center feature.
        
        Args:
            features: [H, W, C]
            gt_centers: [N, 2] (row, col)
        
        Returns:
            contrast_map: [H, W] where lower values = more similar to GT
        """
        H, W, C = features.shape
        N = len(gt_centers)
        
        # Extract GT features
        gt_features = features[gt_centers[:, 0], gt_centers[:, 1]]  # [N, C]
        
        # Compute distance from each spatial position to nearest GT
        features_flat = features.reshape(-1, C)  # [H*W, C]
        
        # Distance matrix: [H*W, N]
        dists = torch.cdist(features_flat, gt_features)  # [H*W, N]
        
        # Min distance to any GT
        min_dists = dists.min(dim=1).values  # [H*W]
        
        contrast_map = min_dists.reshape(H, W)
        
        return contrast_map
    
    def _sample_line(
        self,
        r1: torch.Tensor,
        c1: torch.Tensor,
        r2: torch.Tensor,
        c2: torch.Tensor,
        num_points: int
    ) -> torch.Tensor:
        """
        Sample points along line from (r1, c1) to (r2, c2).
        
        Args:
            r1, c1: Start point
            r2, c2: End point
            num_points: Number of points to sample (including endpoints)
        
        Returns:
            coords: [num_points, 2] tensor of (row, col) coordinates
        """
        # Linearly interpolate
        t = torch.linspace(0, 1, num_points, device=r1.device)
        
        r_interp = r1 + t * (r2 - r1)
        c_interp = c1 + t * (c2 - c1)
        
        # Round to integers
        r_interp = r_interp.round().long()
        c_interp = c_interp.round().long()
        
        coords = torch.stack([r_interp, c_interp], dim=1)  # [num_points, 2]
        
        return coords


class ComponentSeparationLoss(nn.Module):
    """
    Alternative: Directly penalize GT centers falling in the same connected component.
    
    Uses connected-component analysis on thresholded contrast map to measure
    whether GT stomata fall in disjoint high-response regions.
    
    Note: This requires scipy and is slower, mainly for debugging/validation.
    """
    
    def __init__(self, threshold_percentile: float = 0.80):
        super().__init__()
        self.threshold_percentile = threshold_percentile
    
    def forward(
        self,
        features: torch.Tensor,
        gt_centers_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            features: [B, H, W, C]
            gt_centers_list: List of [N, 2] tensors
        
        Returns:
            loss: Fraction of GT pairs that fall in same connected component
        """
        from scipy import ndimage
        import numpy as np
        
        B, H, W, C = features.shape
        device = features.device
        
        total_merged_pairs = 0
        total_pairs = 0
        
        for b in range(B):
            feat = features[b]  # [H, W, C]
            gt_centers = gt_centers_list[b]  # [N, 2]
            
            if len(gt_centers) < 2:
                continue
            
            # Compute contrast map
            contrast_map = self._compute_contrast_map(feat, gt_centers)  # [H, W]
            contrast_np = contrast_map.detach().cpu().numpy()
            
            # Threshold at percentile
            threshold = np.percentile(contrast_np, self.threshold_percentile * 100)
            mask = contrast_np >= threshold
            
            # Connected components (8-connectivity)
            struct = ndimage.generate_binary_structure(2, 2)
            labeled, num_components = ndimage.label(mask, structure=struct)
            
            # Check which component each GT center falls into
            gt_centers_np = gt_centers.cpu().numpy()
            component_ids = labeled[gt_centers_np[:, 0], gt_centers_np[:, 1]]
            
            # Count pairs in same component
            N = len(gt_centers)
            for i in range(N):
                for j in range(i + 1, N):
                    if component_ids[i] == component_ids[j] and component_ids[i] > 0:
                        total_merged_pairs += 1
                    total_pairs += 1
        
        if total_pairs == 0:
            return torch.tensor(0.0, device=device)
        
        merge_rate = total_merged_pairs / total_pairs
        return torch.tensor(merge_rate, device=device)
    
    def _compute_contrast_map(
        self,
        features: torch.Tensor,
        gt_centers: torch.Tensor
    ) -> torch.Tensor:
        """Same as ValleySeparationLoss._compute_contrast_map"""
        H, W, C = features.shape
        gt_features = features[gt_centers[:, 0], gt_centers[:, 1]]
        features_flat = features.reshape(-1, C)
        dists = torch.cdist(features_flat, gt_features)
        min_dists = dists.min(dim=1).values
        return min_dists.reshape(H, W)


def test_valley_loss():
    """Quick sanity test for ValleySeparationLoss"""
    import numpy as np
    
    # Create mock data
    B, H, W, C = 2, 14, 14, 768
    features = torch.randn(B, H, W, C)
    
    # Mock GT centers
    gt_centers_list = [
        torch.tensor([[3, 3], [10, 10]], dtype=torch.long),  # Image 0: 2 centers
        torch.tensor([[5, 5], [8, 8], [5, 10]], dtype=torch.long),  # Image 1: 3 centers
    ]
    
    loss_fn = ValleySeparationLoss(sample_points=5, valley_margin=0.2)
    loss = loss_fn(features, gt_centers_list)
    
    print(f"Valley Separation Loss: {loss.item():.4f}")
    print(f"Expected: positive value (features are random, unlikely to have valleys)")
    
    assert loss.item() >= 0, "Loss should be non-negative"
    print("✓ Test passed")


if __name__ == "__main__":
    test_valley_loss()
