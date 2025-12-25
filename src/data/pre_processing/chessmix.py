"""ChessMix augmentation for semantic segmentation."""

from __future__ import annotations

import random

import torch


class ChessMix:
    """ChessMix augmentation: https://arxiv.org/abs/2108.11535."""

    def __init__(
        self,
        prob: float = 0.5,
        grid_sizes: list[int] | None = None,
        ignore_index: int = 12,
        class_counts: list[float] | None = None,
        num_classes: int = 13,
    ) -> None:
        """Initialize ChessMix augmentation.

        Args:
            prob: Probability of applying augmentation per sample.
            grid_sizes: List of grid sizes to randomly choose from.
            ignore_index: Class index to use for unfilled regions.
            class_counts: Optional precomputed class counts for rare-class weighting.
            num_classes: Number of classes in the dataset.

        """
        self.prob = prob
        self.grid_sizes = grid_sizes or [4]
        self.ignore_index = ignore_index
        self.class_counts = torch.tensor(class_counts).float() if class_counts else None
        self.num_classes = num_classes

    def _calculate_patch_weights(
        self,
        patch_masks: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Calculate weights for each patch based on class rarity.

        Args:
            patch_masks: (N_patches, patch_area) flattened masks for each patch.
            device: Tensor device.

        Returns:
            weights: (N_patches,) weight for each patch.

        """
        if self.class_counts is not None:
            global_counts = self.class_counts.to(device)
            total_pixels = global_counts.sum()
            ci = global_counts / (total_pixels + 1e-6)
        else:
            flat_masks = patch_masks.flatten()
            class_counts = torch.bincount(flat_masks, minlength=self.num_classes).float()
            total_pixels = class_counts.sum()
            ci = class_counts / (total_pixels + 1e-6)

        cmax = ci.max()

        weights_i = torch.zeros_like(ci)
        valid_idx = ci > 0
        weights_i[valid_idx] = cmax / ci[valid_idx]

        weights_lookup = weights_i.to(device)
        pixel_weights = weights_lookup[patch_masks.long()]
        patch_weights = pixel_weights.sum(dim=1)  # (N,)

        return patch_weights

    def __call__(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply ChessMix with Sliding Window Patch Selection."""
        batch_size, c, h, w = images.shape
        if batch_size < 2:
            return images, masks

        grid_size = random.choice(self.grid_sizes)
        patch_h, patch_w = h // grid_size, w // grid_size

        valid_h = patch_h * grid_size
        valid_w = patch_w * grid_size

        stride_h = patch_h // 2
        stride_w = patch_w // 2

        masks_4d = masks.unsqueeze(1).float() if masks.ndim == 3 else masks.float()

        patches_mask_unfolded = torch.nn.functional.unfold(
            masks_4d,
            kernel_size=(patch_h, patch_w),
            stride=(stride_h, stride_w),
        )  # (B, Ph*Pw, L)

        num_patches = patches_mask_unfolded.size(2)
        pool_mask = patches_mask_unfolded.permute(0, 2, 1).reshape(-1, patch_h * patch_w)
        pool_mask_long = pool_mask.long()
        patch_weights = self._calculate_patch_weights(pool_mask_long, images.device)

        rows = torch.arange(grid_size, device=images.device).view(-1, 1)
        cols = torch.arange(grid_size, device=images.device).view(1, -1)
        grid_sum = rows + cols  # (G, G)

        phases = torch.randint(0, 2, (batch_size, 1, 1), device=images.device)
        active_mask = (grid_sum % 2) == phases  # (B, G, G)

        apply_probs = torch.rand(batch_size, device=images.device) < self.prob
        apply_mask = apply_probs.view(batch_size, 1, 1)  # (B, 1, 1)
        fill_mask = active_mask & apply_mask  # (B, G, G)

        num_fill = fill_mask.sum().item()

        target_img_grid = torch.zeros(
            (batch_size, grid_size, grid_size, c, patch_h, patch_w),
            device=images.device,
            dtype=images.dtype,
        )

        if masks.ndim == 3:
            target_mask_grid = torch.full(
                (batch_size, grid_size, grid_size, patch_h, patch_w),
                self.ignore_index,
                device=masks.device,
                dtype=masks.dtype,
            )
        else:
            target_mask_grid = torch.full(
                (batch_size, grid_size, grid_size, 1, patch_h, patch_w),
                self.ignore_index,
                device=masks.device,
                dtype=masks.dtype,
            )

        if num_fill > 0:
            selected_indices = torch.multinomial(patch_weights, num_fill, replacement=True)

            src_b = selected_indices // num_patches
            src_l = selected_indices % num_patches
            flat_patch_dim = c * patch_h * patch_w

            selected_patches_flat = torch.zeros(
                (num_fill, flat_patch_dim),
                device=images.device,
                dtype=images.dtype,
            )

            unique_batches = torch.unique(src_b)
            for b_int in unique_batches:
                b = b_int.item()
                mask_b = src_b == b
                dest_indices = torch.where(mask_b)[0]
                patches_l = src_l[mask_b]

                img_unfolded = torch.nn.functional.unfold(
                    images[b : b + 1],
                    kernel_size=(patch_h, patch_w),
                    stride=(stride_h, stride_w),
                )

                patches_data = img_unfolded[0, :, patches_l].permute(1, 0)
                selected_patches_flat[dest_indices] = patches_data

            selected_masks_flat = pool_mask[selected_indices]

            selected_patches_img = selected_patches_flat.view(num_fill, c, patch_h, patch_w)
            selected_patches_mask = selected_masks_flat.view(num_fill, patch_h, patch_w)
            b_idx, r_idx, c_idx = torch.where(fill_mask)

            target_img_grid[b_idx, r_idx, c_idx] = selected_patches_img

            if target_mask_grid.ndim == 5:
                target_mask_grid[b_idx, r_idx, c_idx] = selected_patches_mask.long()
            else:
                target_mask_grid[b_idx, r_idx, c_idx, 0] = selected_patches_mask.long()

        rec_images = target_img_grid.permute(0, 3, 1, 4, 2, 5).reshape(
            batch_size,
            c,
            valid_h,
            valid_w,
        )

        if masks.ndim == 3:
            rec_masks = target_mask_grid.permute(0, 1, 3, 2, 4).reshape(
                batch_size,
                valid_h,
                valid_w,
            )
        else:
            rec_masks = (
                target_mask_grid.permute(0, 3, 1, 4, 2, 5)
                .reshape(batch_size, 1, valid_h, valid_w)
                .squeeze(1)
            )

        final_images = torch.where(
            apply_probs.view(batch_size, 1, 1, 1),
            rec_images,
            images[:, :, :valid_h, :valid_w],
        )
        if masks.ndim == 3:
            final_masks = torch.where(
                apply_probs.view(batch_size, 1, 1),
                rec_masks,
                masks[:, :valid_h, :valid_w],
            )
        else:
            final_masks = torch.where(
                apply_probs.view(batch_size, 1, 1),
                rec_masks,
                masks[:, :, :valid_h, :valid_w],
            )

        if valid_h < h or valid_w < w:
            pad_h = h - valid_h
            pad_w = w - valid_w
            final_images = torch.nn.functional.pad(final_images, (0, pad_w, 0, pad_h))

            out_full_img = images.clone()
            out_full_mask = masks.clone()

            out_full_img[:, :, :valid_h, :valid_w] = final_images
            if masks.ndim == 3:
                out_full_mask[:, :valid_h, :valid_w] = final_masks
            else:
                out_full_mask[:, :, :valid_h, :valid_w] = final_masks

            return out_full_img, out_full_mask

        return final_images, final_masks
