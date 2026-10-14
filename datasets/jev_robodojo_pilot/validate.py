"""Validate the pilot with Valen's real schema and check local media/provenance."""
import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(schema):
    if schema:
        module_spec = importlib.util.spec_from_file_location('upstream_valen_schema', schema)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        reader = module.read_jsonl
        data = schema.read_bytes()
        blob_sha = hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
        expected = json.loads((HERE/'sources.json').read_text())['format_reference']['schema_git_blob_sha']
        assert blob_sha == expected, (blob_sha, expected)
    else:
        from valen.data.schema import read_jsonl
        reader, blob_sha = read_jsonl, None
    records = reader(HERE/'samples.jsonl')
    seen, checked, clips = set(), set(), 0
    groups, decisions = set(), Counter()
    for record in records:
        meta = record['meta']
        assert meta['record_id'] not in seen
        seen.add(meta['record_id']); groups.add(record['group_id'])
        assert record['group_id'] == meta['source_episode_id']
        assert meta['sim_steps'] == sorted(set(meta['sim_steps']))
        assert abs(meta['sim_steps'][-1]*.04-meta['decision_sim_time_s']) < 1e-6
        messages = record['request']['state']['messages']
        referenced = [part['image_url']['url'] for message in messages for part in message['content'] if part['type']=='image_url']
        assert set(referenced) == {a['path'] for a in record['assets']}
        assert len(referenced) == len(meta['sim_steps'])*len(meta['cameras'])
        for asset in record['assets']:
            path = HERE/asset['path']
            assert path.resolve().is_relative_to(HERE)
            assert sha(path) == asset['sha256']
        for observation in meta['source_observations']:
            src = Path(observation['source_npz'])
            assert sha(src) == observation['source_npz_sha256']
            assert observation['sim_time_s'] <= meta['decision_sim_time_s']
            with np.load(src,allow_pickle=False) as raw:
                for camera in observation['cameras']:
                    path = HERE/'assets'/record['group_id']/f"{observation['step']:06d}_{camera}.png"
                    if path not in checked:
                        with Image.open(path) as img:
                            assert img.size == (640,480)
                            assert np.array_equal(np.asarray(img), raw[camera])
                        checked.add(path)
        review = record['targets'].get('request_review')
        decisions['unlabeled' if review is None else str(bool(review['probabilities']['true']))] += 1
        if review is None:
            assert record['targets']['execution_state']['probabilities']['insufficient_evidence'] == 1
            assert 'subgoal_complete' not in record['targets']
        video = meta['review_assets']['video']
        if video:
            p = HERE/video
            stream = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=nb_frames,width,height,r_frame_rate','-of','json',str(p)],text=True))['streams'][0]
            assert int(stream['nb_frames']) == meta['sim_steps'][-1]-meta['sim_steps'][0]+1
            assert (stream['width'],stream['height'],stream['r_frame_rate']) == (1920,360,'25/1')
            clips += 1
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key,value in attrs:
                if key in ('src','href') and value and not value.startswith('#'):
                    assert '://' not in value and (HERE/value).is_file(), value
    Links().feed((HERE/'review.html').read_text())
    result = dict(records=len(records), groups=len(groups), labelled_questions=sum(len(r['targets']) for r in records),
        review_labels=decisions, official_valen_read_jsonl='passed', schema_git_blob_sha_verified=blob_sha,
        image_sha256_and_native_pixel_equality_checked=len(checked), clips_with_exact_frame_counts=clips,
        html_local_links='passed', future_frames_in_model_inputs=0,
        model_processor_compile_tested=False, annotation_human_reviewed=False)
    (HERE/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--schema',type=Path,help='Optional fetched official valen/data/schema.py, with Git blob SHA verification.')
    main(parser.parse_args().schema)
