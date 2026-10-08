import tempfile
from pathlib import Path
import unittest
import numpy as np
from attentionforge import Adam, cross_entropy, clip_gradients
from attentionforge.recall import RecallModel, make_recall_batch
from test_multihead import numerical_objective_gradient


class LearningTests(unittest.TestCase):
    def test_cross_entropy_gradient_and_stability(self):
        logits = np.array([[1000., 999., 998.], [-1000., -1002., -999.]])
        targets = np.array([1, 2])
        loss, grad = cross_entropy(logits, targets)
        self.assertTrue(np.isfinite(loss))
        numeric = numerical_objective_gradient(logits, lambda: cross_entropy(logits, targets)[0])
        np.testing.assert_allclose(grad, numeric, atol=1e-7)

    def test_adam_first_step_and_validation(self):
        parameter = np.array([2.])
        adam = Adam({"x": parameter}, lr=0.1, eps=1e-8)
        adam.step({"x": np.array([1.])})
        self.assertAlmostEqual(parameter[0], 2-0.1/(1+1e-8), places=12)
        with self.assertRaises(ValueError):
            adam.step({"bad": np.array([1.])})
        self.assertEqual(adam.step_count, 1)

    def test_global_gradient_clipping(self):
        grads = {"a": np.array([3.]), "b": np.array([4.])}
        norm = clip_gradients(grads, 2)
        self.assertAlmostEqual(norm, 5)
        self.assertAlmostEqual(np.sqrt(sum((g*g).sum() for g in grads.values())), 2)

    def test_recall_parameter_gradients(self):
        model = RecallModel(d_model=8, num_heads=2, num_values=3, seed=6)
        q, m, t, _ = make_recall_batch(np.random.default_rng(3), 2, slots=2,
                                      num_keys=4, num_values=3, d_model=8)
        _, _, grads, _ = model.loss_and_grads(q, m, t)
        objective = lambda: cross_entropy(model.forward(q, m)[0], t)[0]
        for name, p in model.params.items():
            numeric = numerical_objective_gradient(p, objective)
            np.testing.assert_allclose(grads[name], numeric, atol=1e-8, rtol=2e-5, err_msg=name)

    def test_checkpoint_roundtrip(self):
        model = RecallModel(seed=4)
        q, m, _, _ = make_recall_batch(np.random.default_rng(4), 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"weights.npz"
            model.save(path)
            restored = RecallModel.load(path)
            np.testing.assert_array_equal(model.forward(q, m)[0], restored.forward(q, m)[0])

    def test_task_and_training_are_nontrivial(self):
        rng = np.random.default_rng(42)
        val = make_recall_batch(np.random.default_rng(1042), 512)
        q, memory, targets, info = val
        np.testing.assert_array_equal(targets, info["values"][np.arange(512), info["chosen"]])
        model = RecallModel(seed=42)
        initial, _ = cross_entropy(model.forward(q, memory)[0], targets)
        optimizer = Adam(model.params, lr=0.003)
        for _ in range(250):
            tq, tm, tt, _ = make_recall_batch(rng, 64)
            _, _, grads, _ = model.loss_and_grads(tq, tm, tt)
            clip_gradients(grads, 1)
            optimizer.step(grads)
        logits = model.forward(q, memory)[0]
        final, _ = cross_entropy(logits, targets)
        self.assertLess(final, initial*0.15)
        self.assertGreater(float(np.mean(logits.argmax(-1) == targets)), 0.95)


if __name__ == "__main__":
    unittest.main()
