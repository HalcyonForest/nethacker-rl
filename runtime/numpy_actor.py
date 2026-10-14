"""Portable FP32 actor. This module and features.py are the complete runtime."""
import json
from pathlib import Path
import numpy as np
from .features import FeatureBuilder, INVENTORY, TTY_ROWS


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -80, 80)))


def conv(x, weight, bias, stride=1, padding=0):
    """FP32 cross-correlation for one CHW/CL input, using NumPy only."""
    shape = weight.shape[2:]
    padded = np.pad(x, ((0, 0),) + ((padding, padding),) * len(shape))
    windows = np.lib.stride_tricks.sliding_window_view(padded, shape, axis=tuple(range(1, x.ndim)))
    windows = windows[(slice(None),) + (slice(None, None, stride),) * len(shape)]
    # Columns use the same C,kH,kW ordering as torch's convolution weights.
    axes = tuple(range(1, 1 + len(shape))) + (0,) + tuple(range(1 + len(shape), windows.ndim))
    columns = np.ascontiguousarray(windows.transpose(axes)).reshape(-1, int(np.prod(weight.shape[1:])))
    out = columns @ weight.reshape(weight.shape[0], -1).T + bias
    return out.T.reshape((weight.shape[0],) + windows.shape[1:1 + len(shape)])


def adaptive_pool(x, shape):
    output = np.empty((x.shape[0],) + tuple(shape), np.float32)
    for index in np.ndindex(*shape):
        spans = tuple(slice(i * n // m, ((i + 1) * n + m - 1) // m)
                      for i, n, m in zip(index, x.shape[1:], shape))
        output[(slice(None),) + index] = x[(slice(None),) + spans].mean(tuple(range(1, x.ndim)))
    return output


class NumpyActor:
    def __init__(self, archive):
        with np.load(archive, allow_pickle=False) as data:
            self.weights = {k: data[k] for k in data.files if k != "metadata"}
            self.meta = json.loads(str(data["metadata"]))
        self.features = FeatureBuilder(self.meta["actions"], self.meta["glyph_count"],
                                       cache=self.meta.get("feature_cache", False),
                                       full_cache=self.meta.get("full_feature_cache", False),
                                       architecture=self.meta.get("architecture", "pooled_v3"), forbid_quit=self.meta.get("forbid_quit", False))
        self.hidden = np.zeros(self.meta["hidden"], dtype=np.float32)

    def linear(self, name, x):
        return x @ self.weights[name + ".weight"].T + self.weights[name + ".bias"]

    def words(self, ids):
        valid = ids != 0
        embedded = self.weights["encoder.word.weight"][ids]
        return (embedded * valid[..., None]).sum(-2) / np.maximum(valid.sum(-1, keepdims=True), 1)

    def convolution(self, name, x, stride=1, padding=0):
        return np.maximum(conv(x, self.weights[name + ".weight"], self.weights[name + ".bias"], stride, padding), 0)

    def spatial(self, glyphs):
        x = self.weights["encoder.glyph.weight"][glyphs].transpose(2, 0, 1)
        return self.convolution("encoder.map_conv2", self.convolution("encoder.map_conv1", x, padding=1), stride=2, padding=1)

    def ordered_text(self, text):
        x = self.weights["encoder.char.weight"][text].T
        x = self.convolution("encoder.text_conv1", x, stride=2, padding=2)
        x = self.convolution("encoder.text_conv2", x, stride=2, padding=1)
        return np.tanh(self.linear("encoder.text_linear", adaptive_pool(x, (16,)).reshape(-1)))

    def logits(self, obs, hidden=None):
        w = self.weights
        hidden = self.hidden if hidden is None else hidden
        cnn_ui = self.meta.get("architecture", "pooled_v3") == "cnn_ui_v1"
        glyph = self.spatial(obs["glyphs"]) if cnn_ui else w["encoder.glyph.weight"][obs["glyphs"]].transpose(2, 0, 1)
        pooled = adaptive_pool(glyph, (4, 8))
        glob = np.tanh(self.linear("encoder.global_linear", pooled.reshape(-1)))
        local_input = self.spatial(obs["crop"]) if cnn_ui else w["encoder.glyph.weight"][obs["crop"]]
        local = np.tanh(self.linear("encoder.local_linear", local_input.reshape(-1)))
        inv_valid = (obs["inventory"] != 0).any(-1)
        menu_valid = (obs["menu"] != 0).any(-1)
        inv = np.tanh(self.linear("encoder.inventory_linear", np.concatenate((self.words(obs["inventory"]), w["encoder.glyph.weight"][obs["inv_glyphs"]], obs["inv_aux"]), -1))) * inv_valid[:, None]
        menu = np.tanh(self.linear("encoder.menu_linear", np.concatenate((self.words(obs["menu"]), obs["menu_aux"]), -1))) * menu_valid[:, None]
        encoded = np.concatenate((glob, local, self.words(obs["message"]),
                                  np.tanh(self.linear("encoder.stats_linear", obs["stats"])),
                                  inv.sum(0) / max(1, int(inv_valid.sum())), menu.sum(0) / max(1, int(menu_valid.sum())),
                                  w["encoder.mode_embedding.weight"][int(obs["mode"][0])],
                                  w["encoder.previous_embedding.weight"][int(obs["previous"][0])])).astype(np.float32)
        if cnn_ui:
            encoded = np.concatenate((encoded, self.ordered_text(obs["text_bytes"]), np.tanh(self.linear("encoder.ui_linear", obs["ui"])))).astype(np.float32)
        fused = np.tanh(self.linear("encoder.fuse", encoded))
        xi = fused @ w["core.core.weight_ih_l0"].T + w["core.core.bias_ih_l0"]
        hh = hidden @ w["core.core.weight_hh_l0"].T + w["core.core.bias_hh_l0"]
        ir, iz, inn = np.split(xi, 3)
        hr, hz, hn = np.split(hh, 3)
        r, z = sigmoid(ir + hr), sigmoid(iz + hz)
        n = np.tanh(inn + r * hn)
        state = ((1 - z) * n + z * hidden).astype(np.float32)
        logits = np.concatenate((self.linear("raw", state), inv @ self.linear("item_query", state) / 8,
                                 menu @ self.linear("menu_query", state) / 8)).astype(np.float32)
        logits[obs["action_mask"] == 0] = -1e9
        return logits, state

    def reset(self, initial_observation):
        self.hidden.fill(0)
        self.features.reset()

    def act(self, observation):
        features = self.features.prepare(observation)
        logits, self.hidden = self.logits(features)
        if self.meta.get("deterministic", False):
            action = int(np.argmax(logits))
        else:
            # Deploy the same masked categorical policy optimized by PPO.
            # Arena seeds NumPy with bot_seed before make_agent().
            probabilities = np.exp((logits - logits.max()).astype(np.float64))
            probabilities /= probabilities.sum()
            action = int(np.random.choice(len(logits), p=probabilities))
        return self.features.codec.decode(action, self.features.last_view)
