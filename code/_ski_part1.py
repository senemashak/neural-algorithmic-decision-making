"""Ski configs 1-6 (first half). Uses latest sweep_weights.py with all fixes."""
import importlib
import sweep_weights
importlib.reload(sweep_weights)  # ensure latest code
import sys

sweep_weights.CONFIGS = [
    ("value_only",    1.0, 0.0, 0.0, "teacher_forcing"),
    ("action_only",   0.0, 1.0, 0.0, "teacher_forcing"),
    ("chain_only",    0.0, 0.0, 1.0, "teacher_forcing"),
    ("value+action",  1.0, 0.5, 0.0, "teacher_forcing"),
    ("value+chain",   0.5, 0.0, 0.5, "teacher_forcing"),
    ("action+chain",  0.0, 0.5, 0.5, "teacher_forcing"),
]

sys.argv = ['sw', '--device', 'cuda', '--only', 'ski', '--epochs', '30',
            '--out_dir', 'results/sweep_ski_part1']
sweep_weights.main()
