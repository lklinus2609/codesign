"""Composite convergence gate for PPO inner loops.

Replaces the broken KL-based convergence check with multiple orthogonal
signals that are immune to the adaptive LR thermostat (which keeps KL
artificially elevated) and the AMP discriminator's continuous learning
signal (which prevents KL from decaying to zero).

All enabled signals must be satisfied simultaneously over the same
trailing window for convergence to be declared.
"""

import numpy as np
from collections import deque


class CompositeConvergenceGate:
    """Multi-signal convergence detector for PPO inner loops.

    Supported signal types:
        plateau  -- relative spread of recent values < threshold
        below    -- mean of recent values < threshold
        above    -- mean of recent values > threshold

    Usage:
        gate = CompositeConvergenceGate(window=10)
        gate.add_signal('critic_loss', 'plateau', threshold=0.05)
        gate.add_signal('adv_std', 'below', threshold=0.1)

        # In training loop:
        gate.update({'critic_loss': 0.02, 'adv_std': 0.08})
        if gate.check_converged():
            break
    """

    VALID_TYPES = ('plateau', 'below', 'above')

    def __init__(self, window=10):
        self.window = window
        self._signals = {}

    def add_signal(self, name, check_type, threshold):
        """Register a convergence signal.

        Args:
            name: Key used in update() metrics dict.
            check_type: One of 'plateau', 'below', 'above'.
            threshold: Float threshold for the check.
        """
        if check_type not in self.VALID_TYPES:
            raise ValueError(f"check_type must be one of {self.VALID_TYPES}")
        self._signals[name] = {
            'type': check_type,
            'threshold': threshold,
            'buffer': deque(maxlen=self.window),
        }

    def update(self, metrics):
        """Feed one observation of metrics.

        Args:
            metrics: dict mapping signal names to float values.
                     Unknown keys are silently ignored.
        """
        for name, sig in self._signals.items():
            val = metrics.get(name)
            if val is not None and np.isfinite(val):
                sig['buffer'].append(float(val))

    def check_converged(self):
        """Returns True if ALL enabled signals pass over the window."""
        if not self._signals:
            return False
        for sig in self._signals.values():
            if len(sig['buffer']) < self.window:
                return False
            recent = list(sig['buffer'])
            if sig['type'] == 'plateau':
                if not self._is_plateau(recent, sig['threshold']):
                    return False
            elif sig['type'] == 'below':
                if np.mean(recent) >= sig['threshold']:
                    return False
            elif sig['type'] == 'above':
                if np.mean(recent) <= sig['threshold']:
                    return False
        return True

    def get_diagnostics(self):
        """Returns per-signal status dict for logging."""
        diag = {}
        all_ok = True
        for name, sig in self._signals.items():
            buf = sig['buffer']
            if len(buf) < self.window:
                diag[f"gate/{name}_converged"] = False
                diag[f"gate/{name}_mean"] = float(np.mean(buf)) if buf else 0.0
                all_ok = False
            else:
                recent = list(buf)
                if sig['type'] == 'plateau':
                    ok = self._is_plateau(recent, sig['threshold'])
                elif sig['type'] == 'below':
                    ok = np.mean(recent) < sig['threshold']
                elif sig['type'] == 'above':
                    ok = np.mean(recent) > sig['threshold']
                diag[f"gate/{name}_converged"] = ok
                diag[f"gate/{name}_mean"] = float(np.mean(recent))
                if not ok:
                    all_ok = False
        diag["gate/all_converged"] = all_ok
        return diag

    def reset(self):
        """Clear all buffers (call at start of each outer iteration)."""
        for sig in self._signals.values():
            sig['buffer'].clear()

    @staticmethod
    def _is_plateau(values, threshold):
        """Check if values have plateaued (relative spread < threshold)."""
        mean_val = np.mean(values)
        spread = max(values) - min(values)
        if abs(mean_val) < 1.0:
            return spread < threshold
        return (spread / abs(mean_val)) < threshold
