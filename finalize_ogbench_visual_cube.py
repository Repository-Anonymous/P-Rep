"""Validate a completed native-224 benchmark and append its results to notes."""

import argparse
import json
import shutil
import time
from pathlib import Path

from report_ogbench_visual_cube import ROOT, generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', type=Path)
    args = parser.parse_args()
    output = args.benchmark.resolve()
    while not (output / 'metrics.json').exists():
        time.sleep(10)
    metrics = json.loads((output / 'metrics.json').read_text())
    assert metrics['config']['total_episodes'] == 4000
    encoder_checks = [json.loads((output / f'gpu_{gpu}_encoder_input_check.json').read_text())
                      for gpu in range(8)]
    assert all(check['passed'] and check['actual_encoder_input_shape'][-3:] == [3, 224, 224]
               for check in encoder_checks)
    import imageio.v2 as imageio
    import imageio_ffmpeg
    videos = []
    for job in metrics['jobs']:
        if job['eval_seed'] != metrics['config']['seeds'][0]:
            continue
        path = output / 'videos' / f'{job["variant"]}_task{job["task_id"]}.mp4'
        reader = imageio.get_reader(path)
        metadata = reader.get_meta_data()
        reader.close()
        assert list(metadata['source_size']) == [224, 224]
        count, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
        assert count == job['records'][0]['steps'] + 1
        videos.append({'filename': path.name, 'variant': job['variant'], 'task_id': job['task_id'],
                       'size': list(metadata['source_size']), 'fps': metadata['fps'],
                       'frames': count, 'duration_seconds': seconds,
                       'video_episode_success': job['records'][0]['success'],
                       'included_in_benchmark_statistics': True})
    assert len(videos) == 20
    for path in (output / 'videos').glob('*_goal64.png'):
        path.rename(path.with_name(path.name.replace('_goal64.png', '_goal.png')))
    manifest = {'layout': 'native RGB, no border or overlay',
                'extra_demonstration_episodes': 0, 'encoder_input_checks': encoder_checks,
                'videos': videos}
    (output / 'video-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Clarify official action-space behavior without changing any recorded metric.
    config = json.loads((output / 'config.json').read_text())
    description = ('Simulator and NumPy seeded. Official action_space is a property creating '
                   'a new Box per access; goal-settling actions retain independent sampling.')
    config['reset_seeding'] = description
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    metrics = json.loads((output / 'metrics.json').read_text())
    metrics['config']['reset_seeding'] = description
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    generate(output / 'metrics.json', ROOT / 'zeyu-understand.md',
             ROOT / 'ogbench_table5_visual_cube_baselines.json')
    print(json.dumps(metrics['variants'], indent=2), flush=True)


if __name__ == '__main__':
    main()
