import unittest

import torch
from torch import nn

from emboditta.core import (
    EMADomainShiftDetector,
    adapt_bn_decoupled,
    extract_bn_state,
    select_source_candidate,
)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
        self.classifier = nn.Linear(1, 2)

    def forward(self, images):
        features = self.bn(images).mean(dim=(2, 3))
        return self.classifier(features)


class EmbodiTTACoreTests(unittest.TestCase):
    def test_detector_triggers_and_suppresses_recovery_triggers(self):
        detector = EMADomainShiftDetector(
            reference_entropy=0.5,
            stable_ema_std=0.01,
            momentum=0.0,
            threshold_scale=2.0,
        )
        logits = torch.tensor([[10.0, 0.0]])
        self.assertEqual(detector.update(logits), [True])
        detector.begin_recovery(stable_samples=2)
        self.assertEqual(detector.update(logits), [False])
        self.assertEqual(detector.update(logits), [False])
        self.assertAlmostEqual(detector.reference_entropy, float((-torch.softmax(logits, 1) * torch.log_softmax(logits, 1)).sum()))

    def test_detector_keeps_high_momentum_after_recovery(self):
        detector = EMADomainShiftDetector(
            reference_entropy=0.5,
            stable_ema_std=0.01,
            momentum=0.995,
            threshold_scale=2.0,
        )
        detector.begin_recovery(stable_samples=101, initial_entropy=0.8)
        logits = torch.tensor([[10.0, 0.0]])
        detector.update(logits.repeat(101, 1))
        self.assertEqual(detector.momentum, 0.999)
        self.assertAlmostEqual(detector.reference_entropy, detector.ema_entropy)

    def test_rebaseline_uses_adaptation_entropy_and_checks_next_sample(self):
        detector = EMADomainShiftDetector(
            reference_entropy=0.5,
            stable_ema_std=0.01,
            momentum=0.0,
            threshold_scale=2.0,
            smoothing_window=10,
        )
        calibrated_threshold = detector.threshold
        confident_logits = torch.tensor([[10.0, 0.0]])
        adaptation_entropy = float(
            (-(confident_logits.softmax(dim=1) * confident_logits.log_softmax(dim=1)))
            .sum(dim=1)
            .mean()
        )

        detector.rebaseline(adaptation_entropy)

        self.assertAlmostEqual(detector.reference_entropy, adaptation_entropy)
        self.assertEqual(detector.threshold, calibrated_threshold)
        self.assertEqual(detector._recovery_remaining, 0)
        self.assertEqual(detector.update(torch.zeros(1, 2)), [True])

    def test_candidate_selection_loads_nearest_bn_state(self):
        model = TinyClassifier()
        base = extract_bn_state(model)
        near = {name: value.clone() for name, value in base.items()}
        far = {name: value.clone() for name, value in base.items()}
        near["bn.running_mean"].fill_(0.9)
        far["bn.running_mean"].fill_(8.0)
        images = torch.full((8, 1, 4, 4), 3.0)
        selected = select_source_candidate(
            model,
            images,
            [far, near],
            "cpu",
            feature_name="bn.running_mean",
            batch_size=4,
            stats_momentum=0.3,
        )
        self.assertEqual(selected, 1)
        self.assertTrue(torch.equal(model.bn.running_mean, near["bn.running_mean"]))

    def test_decoupled_adaptation_updates_bn_but_freezes_classifier(self):
        torch.manual_seed(3)
        model = TinyClassifier()
        images = torch.randn(4, 1, 4, 4) + 1.5
        with torch.no_grad():
            previous_logits = model(images).clone()
        classifier_before = model.classifier.weight.detach().clone()
        report = adapt_bn_decoupled(
            model,
            images,
            previous_logits,
            "cpu",
            stats_batch_size=2,
            parameter_batch_size=1,
            entropy_margin=10.0,
            consistency_weight=0.01,
        )
        self.assertGreater(report["updated_batches"], 0)
        self.assertTrue(torch.equal(model.classifier.weight, classifier_before))
        self.assertFalse(torch.equal(model.bn.weight, torch.ones_like(model.bn.weight)))

    def test_auto_stats_momentum_scales_with_window_and_batch(self):
        torch.manual_seed(3)
        model = TinyClassifier()
        images = torch.randn(8, 1, 4, 4)
        report = adapt_bn_decoupled(
            model,
            images,
            None,
            "cpu",
            stats_batch_size=2,
            parameter_batch_size=1,
            entropy_margin=10.0,
        )
        # 8 samples / stats batch 2 -> 4 stats updates -> momentum 1/4.
        self.assertAlmostEqual(report["stats_momentum"], 0.25)
        self.assertAlmostEqual(model.bn.momentum, 0.25)

    def test_consistency_requires_previous_logits_when_enabled(self):
        model = TinyClassifier()
        images = torch.randn(4, 1, 4, 4)
        with self.assertRaises(ValueError):
            adapt_bn_decoupled(
                model,
                images,
                None,
                "cpu",
                entropy_margin=10.0,
                consistency_weight=0.01,
            )


if __name__ == "__main__":
    unittest.main()