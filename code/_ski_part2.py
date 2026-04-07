"""Ski configs 7-11 (second half). Uses latest sweep_weights.py with all fixes."""
import importlib
import sweep_weights
importlib.reload(sweep_weights)  # ensure latest code
import sys

sweep_weights.CONFIGS = [
    ("equal_third",   1/3, 1/3, 1/3, "teacher_forcing"),
    ("emph_value",    0.5, 0.25, 0.25, "teacher_forcing"),
    ("emph_action",   0.25, 0.5, 0.25, "teacher_forcing"),
    ("emph_chain",    0.25, 0.25, 0.5, "teacher_forcing"),
    ("all_1_0.5_1",   1.0, 0.5, 1.0, "teacher_forcing"),
]

sys.argv = ['sw', '--device', 'cuda', '--only', 'ski', '--epochs', '30',
            '--out_dir', 'results/sweep_ski_part2']
sweep_weights.main()
