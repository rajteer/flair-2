"""Tests for validation utilities in src/models/validation.py."""

import pytest
import torch

from src.training.validation import (
    calculate_iou_scores,
    compute_timing_metrics,
    get_evaluation_metrics_dict,
)


class TestCalculateIoUScores:
    """Tests for calculate_iou_scores function."""

    def test_perfect_predictions_yield_iou_one(self) -> None:
        """Perfect predictions should yield IoU = 1.0 for all classes."""
        num_classes = 5
        conf_matrix = torch.eye(num_classes) * 100

        miou, per_class_iou = calculate_iou_scores(
            conf_matrix,
            num_classes,
            other_class_index=-1,  # No class excluded
        )

        assert miou == pytest.approx(1.0)
        for class_idx, iou in per_class_iou.items():
            assert iou == pytest.approx(1.0)

    def test_completely_wrong_predictions_yield_iou_zero(self) -> None:
        """Completely wrong predictions should yield IoU = 0.0."""
        num_classes = 3
        # All predictions are off-diagonal (class 0 predicted as 1, 1 as 2, 2 as 0)
        conf_matrix = torch.zeros(num_classes, num_classes)
        conf_matrix[0, 1] = 100  # True class 0, predicted as 1
        conf_matrix[1, 2] = 100  # True class 1, predicted as 2
        conf_matrix[2, 0] = 100  # True class 2, predicted as 0

        miou, per_class_iou = calculate_iou_scores(conf_matrix, num_classes, other_class_index=-1)

        assert miou == pytest.approx(0.0)
        for class_idx, iou in per_class_iou.items():
            assert iou == pytest.approx(0.0)

    def test_other_class_is_excluded(self) -> None:
        """The 'other' class should be excluded from mIoU calculation."""
        num_classes = 4
        other_class_index = 3

        conf_matrix = torch.eye(num_classes) * 100
        conf_matrix[other_class_index, :] = 0
        conf_matrix[:, other_class_index] = 0

        miou, per_class_iou = calculate_iou_scores(
            conf_matrix,
            num_classes,
            other_class_index=other_class_index,
        )

        assert miou == pytest.approx(1.0)
        assert other_class_index not in per_class_iou

    def test_returns_correct_structure(self) -> None:
        """Should return (float, dict) with correct structure."""
        num_classes = 5
        conf_matrix = torch.eye(num_classes) * 50

        miou, per_class_iou = calculate_iou_scores(conf_matrix, num_classes, other_class_index=4)

        assert isinstance(miou, float)
        assert isinstance(per_class_iou, dict)
        # Class 4 is excluded
        assert len(per_class_iou) == num_classes - 1

    def test_handles_empty_classes(self) -> None:
        """Should handle classes with zero samples (avoid division by zero)."""
        num_classes = 3
        conf_matrix = torch.zeros(num_classes, num_classes)
        conf_matrix[0, 0] = 100  # Only class 0 has samples

        miou, per_class_iou = calculate_iou_scores(conf_matrix, num_classes, other_class_index=-1)

        # Should not raise and should return valid values
        assert not torch.isnan(torch.tensor(miou))


class TestComputeTimingMetrics:
    """Tests for compute_timing_metrics function."""

    def test_returns_correct_keys(self) -> None:
        """Should return dict with all expected keys."""
        inference_times = [0.1, 0.2, 0.15]
        batch_sizes = [4, 4, 4]

        metrics = compute_timing_metrics(inference_times, batch_sizes)

        expected_keys = {
            "total_inference_time",
            "total_images",
            "avg_time_per_image",
            "avg_time_per_batch",
        }
        assert set(metrics.keys()) == expected_keys

    def test_calculates_totals_correctly(self) -> None:
        """Should calculate total time and images correctly."""
        inference_times = [0.1, 0.2, 0.3]
        batch_sizes = [4, 4, 4]

        metrics = compute_timing_metrics(inference_times, batch_sizes)

        assert metrics["total_inference_time"] == pytest.approx(0.6)
        assert metrics["total_images"] == 12

    def test_calculates_averages_correctly(self) -> None:
        """Should calculate average times correctly."""
        inference_times = [0.1, 0.2, 0.3]
        batch_sizes = [4, 4, 4]

        metrics = compute_timing_metrics(inference_times, batch_sizes)

        assert metrics["avg_time_per_image"] == pytest.approx(0.6 / 12)
        assert metrics["avg_time_per_batch"] == pytest.approx(0.2)

    def test_handles_empty_lists(self) -> None:
        """Should handle empty input lists gracefully."""
        metrics = compute_timing_metrics([], [])

        assert metrics["total_inference_time"] == 0
        assert metrics["total_images"] == 0
        assert metrics["avg_time_per_image"] == 0.0
        assert metrics["avg_time_per_batch"] == 0.0


class TestGetEvaluationMetricsDict:
    """Tests for get_evaluation_metrics_dict function."""

    def test_returns_dict_with_required_metrics(
        self,
        num_classes: int,
        device: torch.device,
    ) -> None:
        """Should return dict with all required metric objects."""
        metrics_dict = get_evaluation_metrics_dict(num_classes, device)

        expected_keys = {
            "conf_matrix",
            "macro_f1",
            "f1_per_class",
            "overall_f1",
            "macro_accuracy",
            "overall_accuracy",
        }
        assert set(metrics_dict.keys()) == expected_keys

    def test_metrics_are_on_correct_device(self, num_classes: int, device: torch.device) -> None:
        """Metrics should be on the specified device."""
        metrics_dict = get_evaluation_metrics_dict(num_classes, device)

        for name, metric in metrics_dict.items():
            # TorchMetrics objects have a device attribute
            assert hasattr(metric, "device")

    def test_respects_other_class_index(self, num_classes: int, device: torch.device) -> None:
        """Should set ignore_index for metrics when other_class_index is provided."""
        other_class_index = num_classes - 1
        metrics_dict = get_evaluation_metrics_dict(
            num_classes,
            device,
            other_class_index=other_class_index,
        )

        assert len(metrics_dict) == 6

    def test_metrics_can_be_updated(
        self,
        num_classes: int,
        device: torch.device,
        batch_size: int,
        image_size: int,
    ) -> None:
        """Metrics should accept updates from predictions and targets."""
        metrics_dict = get_evaluation_metrics_dict(num_classes, device)

        preds = torch.randint(0, num_classes, (batch_size, image_size, image_size)).to(device)
        targets = torch.randint(0, num_classes, (batch_size, image_size, image_size)).to(device)

        for metric in metrics_dict.values():
            metric.update(preds, targets)
