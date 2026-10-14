"""Select completed fine-tunes using full held-out prediction MSE.

SIGReg measures a batch distribution. Consecutive clips from the same
trajectory make its validation statistic incomparable with shuffled training
batches. Keep training unchanged and retain the original combined scores.
"""

import fcntl
import json
import math
import re
from pathlib import Path

from eval_ogbench_visual_cube import sha256, write_json


def validation_epochs(training):
    training = Path(training)
    config = json.loads((training / 'training-config.json').read_text())
    pattern = re.compile(r'^EPOCH (\d+) COMPLETE: validation=(\[[^\]]+\]), best=')
    rows = []
    for line in (training / 'train.log').read_text().splitlines():
        match = pattern.match(line)
        if not match:
            continue
        epoch = int(match[1])
        values = json.loads(match[2])
        assert len(values) == 3 and all(math.isfinite(v) and v >= 0 for v in values)
        checkpoint = training / 'checkpoints' / f'epoch_{epoch:02d}'
        assert (checkpoint / 'weights.pt').exists() and (checkpoint / 'config.json').exists()
        rows.append({'epoch': epoch,
                     'global_step': min(epoch * config['steps_per_epoch'], config['total_optimizer_steps']),
                     'val_prediction_mse': values[1], 'val_combined_loss': values[0],
                     'val_sigreg': values[2], 'validation_sequences': config['val_sequences'],
                     'checkpoint': str(checkpoint.resolve())})
    assert rows, 'No complete held-out validation results available'
    return rows


def evaluation_checkpoint(settings):
    """Opt in through a run-local request; all other evaluations are unchanged."""
    checkpoint = Path(settings['checkpoint']).resolve()
    if checkpoint.parent.name != 'checkpoints':
        return settings
    training = checkpoint.parent.parent
    root = training.parent.parent
    request = root / 'validation-selection-request.json'
    if not request.exists() or not json.loads(request.read_text()).get('enabled'):
        return settings
    with (training / 'prediction-selection.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        assert (training / 'train-complete.json').exists(), 'Do not select an unfinished fine-tune'
        audit_path = training / 'prediction-checkpoint-selection.json'
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
        else:
            rows = validation_epochs(training)
            complete = json.loads((training / 'train-complete.json').read_text())
            assert rows[-1]['global_step'] == complete['optimizer_steps'], 'Final validation is missing'
            winner = min(rows, key=lambda row: (row['val_prediction_mse'], row['epoch']))
            previous = json.loads((training / 'best-checkpoint.json').read_text())
            audit = {'completed': True, 'metric': 'minimum full held-out prediction MSE',
                     'reason': 'Consecutive-trajectory batching inflates validation SIGReg',
                     'original_combined_best': previous, 'candidates': rows, 'selected': winner,
                     'selected_sha256': sha256(Path(winner['checkpoint']) / 'weights.pt')}
            write_json(audit_path, audit)
        assert audit['completed']
        winner = audit['selected']
        assert sha256(Path(winner['checkpoint']) / 'weights.pt') == audit['selected_sha256']
        # Refer to an immutable epoch directory; preserve every trained weight.
        best = {**winner, 'val_loss': winner['val_prediction_mse'],
                'selection_metric': audit['metric'], 'sha256': audit['selected_sha256'],
                'original_combined_best': audit['original_combined_best']}
        write_json(training / 'best-checkpoint.json', best)
        selected = {**settings, 'checkpoint': winner['checkpoint'],
                    'checkpoint_selection_metric': audit['metric']}
        config_path = Path(settings['output']) / 'config.json'
        if config_path.exists():
            config = json.loads(config_path.read_text())
            already_selected = config.get('checkpoint_selection_metric') == audit['metric']
            if not already_selected:
                assert not list((config_path.parent / 'jobs').glob('*.json')), 'Do not mix evaluated checkpoints'
            else:
                assert config['checkpoint_sha256'] == audit['selected_sha256']
            config.update(checkpoint=winner['checkpoint'], checkpoint_sha256=audit['selected_sha256'],
                          checkpoint_selection_metric=audit['metric'],
                          checkpoint_selection_audit=str(audit_path.resolve()))
            write_json(config_path, config)
        return selected
