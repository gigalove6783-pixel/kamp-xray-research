import unittest
import numpy as np
from augmentation import fixture, extract_bank, augment
from evaluation import evaluate


class PipelineTests(unittest.TestCase):
    def test_insertion_is_local_and_repeatable(self):
        donor = fixture(1, "A", 2)
        clean = fixture(2, "A")
        banks = extract_bank([donor])
        a, b = [augment(clean, banks, 5) for _ in range(2)]
        np.testing.assert_array_equal(a.image, b.image)
        self.assertEqual(a.centers, b.centers)
        self.assertEqual(len(a.centers), 2)
        changed = a.image != clean.image
        self.assertTrue(changed.any())
        self.assertFalse(changed[~clean.mask].any())
        self.assertTrue(np.all(a.image <= clean.image))
        y, x = np.indices(a.image.shape)
        support = np.zeros_like(changed)
        for cx, cy in a.centers:
            support |= np.hypot(x-cx, y-cy) <= 2
        self.assertFalse(changed[~support].any())

    def test_missing_equipment_is_not_silently_mixed(self):
        banks = extract_bank([fixture(1, "A", 2)])
        with self.assertRaises(ValueError):
            augment(fixture(2, "B"), banks, 5)

    def test_duplicate_predictions_only_match_once(self):
        result = evaluate([([(10, 10)], [(10, 10, .9), (11, 10, .8)])])
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["precision"], .5)
        self.assertEqual(result["unmatched_per_image"], 1)

    def test_false_positive_before_hit_reduces_ap(self):
        result = evaluate([([(10, 10)], [(50, 50, .9), (10, 10, .8)])])
        self.assertEqual(result["point_ap_3px"], .5)

    def test_image_identity_and_negative_cases(self):
        result = evaluate([([(10, 10)], []), ([], [(10, 10, .9)])])
        self.assertEqual(result["true_positive"], 0)
        self.assertEqual(result["point_ap_3px"], 0)
        self.assertIsNone(evaluate([([], [])])["recall"])


if __name__ == "__main__":
    unittest.main()
