"""
scratch_xgboost.py
------------------
A small XGBoost-style classifier written FROM SCRATCH with only NumPy.
(Rubric item 3: "Implement at least one algorithm manually".)

How XGBoost works, in plain words
=================================
1. Start with the same guess for every user (the log-odds of the
   overall retention rate).
2. For every user compute
      gradient  g = p - y          (how wrong the current guess is)
      hessian   h = p * (1 - p)    (how confident the current guess is)
   where p = sigmoid(current score) and y is the true label.
3. Grow a small decision tree that groups users with similar gradients.
   A split is chosen by the XGBoost GAIN formula:

      Gain = 1/2 * [ GL^2/(HL+lambda) + GR^2/(HR+lambda)
                     - (GL+GR)^2/(HL+HR+lambda) ] - gamma

   (GL, HL = sum of g and h of users going LEFT; GR, HR = going RIGHT.)
4. Each leaf gets the weight   w = -G / (H + lambda)
5. Add  learning_rate * tree_output  to every user's score.
6. Repeat steps 2-5 for n_estimators trees.
7. Final probability = sigmoid(total score).

lambda (L2 regularisation) shrinks leaf weights, gamma is the minimum gain
needed to make a split - these are exactly XGBoost's `reg_lambda` and
`gamma` hyper-parameters.

To keep it fast we use XGBoost's own "histogram" trick: each feature is
cut into at most `n_bins` quantile buckets and only bucket borders are
tried as split points.
"""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "weight", "gain")

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.weight = 0.0
        self.gain = 0.0


class ScratchXGBClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 reg_lambda=1.0, gamma=0.0, min_child_weight=1.0, n_bins=32):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.n_bins = n_bins
        self.trees = []
        self.base_score = 0.0
        self.first_split_log = None      # explains the very first split
        self.train_loss_history = []

    # ---------- helpers ----------
    def _leaf_weight(self, G, H):
        return -G / (H + self.reg_lambda)

    def _score(self, G, H):
        return G * G / (H + self.reg_lambda)

    def _best_split(self, Xb, g, h, idx):
        """Find best (feature, bin) split for the users in idx."""
        G, H = g[idx].sum(), h[idx].sum()
        parent = self._score(G, H)
        best = (0.0, None, None, None)          # gain, feature, bin, stats
        for f in range(Xb.shape[1]):
            bins = Xb[idx, f]
            # sum gradients/hessians per bucket in one vectorised call
            g_hist = np.bincount(bins, weights=g[idx], minlength=self.n_bins)
            h_hist = np.bincount(bins, weights=h[idx], minlength=self.n_bins)
            GL = np.cumsum(g_hist)[:-1]
            HL = np.cumsum(h_hist)[:-1]
            GR, HR = G - GL, H - HL
            valid = (HL >= self.min_child_weight) & (HR >= self.min_child_weight)
            if not valid.any():
                continue
            gain = 0.5 * (self._score(GL, HL) + self._score(GR, HR)
                          - parent) - self.gamma
            gain[~valid] = -np.inf
            b = int(np.argmax(gain))
            if gain[b] > best[0]:
                best = (gain[b], f, b,
                        dict(GL=GL[b], HL=HL[b], GR=GR[b], HR=HR[b],
                             G=G, H=H))
        return best

    def _build(self, Xb, g, h, idx, depth):
        node = _Node()
        G, H = g[idx].sum(), h[idx].sum()
        node.weight = self._leaf_weight(G, H)
        if depth >= self.max_depth or len(idx) < 2:
            return node
        gain, f, b, stats = self._best_split(Xb, g, h, idx)
        if f is None or gain <= 0:
            return node
        if self.first_split_log is None:
            self.first_split_log = {"feature_index": f, "bin": b,
                                    "gain": float(gain),
                                    **{k: float(v) for k, v in stats.items()},
                                    "n_users": int(len(idx)),
                                    "reg_lambda": self.reg_lambda,
                                    "gamma": self.gamma}
        node.feature, node.threshold, node.gain = f, b, gain
        mask = Xb[idx, f] <= b
        node.left = self._build(Xb, g, h, idx[mask], depth + 1)
        node.right = self._build(Xb, g, h, idx[~mask], depth + 1)
        return node

    def _predict_tree(self, node, Xb):
        out = np.empty(Xb.shape[0])
        stack = [(node, np.arange(Xb.shape[0]))]
        while stack:
            nd, rows = stack.pop()
            if nd.feature is None:
                out[rows] = nd.weight
                continue
            m = Xb[rows, nd.feature] <= nd.threshold
            stack.append((nd.left, rows[m]))
            stack.append((nd.right, rows[~m]))
        return out

    def _binarize(self, X):
        Xb = np.empty(X.shape, dtype=np.int64)
        for f in range(X.shape[1]):
            Xb[:, f] = np.searchsorted(self.bin_edges_[f], X[:, f], side="right")
        return np.minimum(Xb, self.n_bins - 1)

    # ---------- public API (same style as scikit-learn) ----------
    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.bin_edges_ = [np.unique(np.quantile(X[:, f], qs))
                           for f in range(X.shape[1])]
        Xb = self._binarize(X)

        p0 = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.base_score = np.log(p0 / (1 - p0))           # log-odds
        F = np.full(len(y), self.base_score)
        all_idx = np.arange(len(y))
        self.trees, self.train_loss_history = [], []
        for _ in range(self.n_estimators):
            p = sigmoid(F)
            g, h = p - y, p * (1 - p)
            tree = self._build(Xb, g, h, all_idx, 0)
            F += self.learning_rate * self._predict_tree(tree, Xb)
            self.trees.append(tree)
            p = sigmoid(F)
            loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
            self.train_loss_history.append(float(loss))
        return self

    def decision_function(self, X):
        Xb = self._binarize(np.asarray(X, dtype=float))
        F = np.full(Xb.shape[0], self.base_score)
        for t in self.trees:
            F += self.learning_rate * self._predict_tree(t, Xb)
        return F

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def threshold_value(self, feature_index, bin_index):
        """Convert a bin index back to a real feature value."""
        edges = self.bin_edges_[feature_index]
        return float(edges[min(bin_index, len(edges) - 1)])
