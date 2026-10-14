"""Build a local, reviewable Valen pilot from frozen annotations and existing rollouts.

Requires numpy, Pillow and ffmpeg/ffprobe. Does not call a model or modify sources.
"""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import subprocess
import shutil
import tempfile

import numpy as np
from PIL import Image, ImageDraw

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = 'ffmpeg'


HERE = Path(__file__).resolve().parent
CAMERAS = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
STATUS = {
    'progressing': 'The target action is still progressing; incomplete success alone is not failure.',
    'satisfied': 'The stated local success condition is visibly satisfied at the final observation.',
    'execution_failure': 'An attempted action visibly failed or lost its result, and remains unresolved at the final observation.',
    'repeated_no_effect': 'Multiple attempts fail to establish the required effect over the supplied history.',
    'insufficient_evidence': 'The supplied views or temporal history cannot establish the action outcome.',
}
POLICY = (
    'Judge only the supplied observation history, at its final timestamp, for the stated target and subgoal. '
    'Request a review when an attempted action visibly fails and remains unresolved, or repeated attempts '
    'do not establish the required effect. A review is an observation/planning check, not an emergency stop '
    'and not a claim that intervention would improve reward. Do not request review merely because an action '
    'has not finished. Do not repeat an old failure alarm if the supplied history shows recovery by the final '
    'timestamp. If evidence is insufficient, report insufficient_evidence instead of inventing an outcome. '
    'No previous review or cooldown is assumed. Current subgoal and success condition are supplied monitoring '
    'context; they are not a statement of what the robot has achieved.'
)
QUESTIONS = {
    'execution_state': {'type': 'choice', 'instructions': 'Choose the execution state at the final observation.', 'criteria': STATUS},
    'request_review': {'type': 'noul', 'instructions': 'Does the supplied evidence warrant a review now under the monitoring policy in the state?'},
    'subgoal_complete': {'type': 'noul', 'instructions': 'Is the supplied local success condition visibly satisfied at the final observation? This is not whole-task success.'},
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration',
        '-of', 'json', str(path)], text=True))['streams'][0]


def target(keys, selected):
    return {'probabilities': {k: float(k == selected) for k in keys},
            'label_source': 'codex_visual_draft_pending_human_review'}


def build(root):
    spec = json.loads((HERE / 'annotations.json').read_text())
    inventory = []
    for case in sorted(root.glob('[0-9][0-9]-*')):
        summary = case / 'sim/summary.json'
        if summary.exists():
            s = json.loads(summary.read_text())
            inventory.append(dict(case=case.name, episode_id=s['episode_id'], steps=s['step_id'],
                native_success=s['success'], video_present=(case/'sim/sensors.mp4').exists(),
                visual_overview_inspected=case.name[:2] in {'01', '02', '06', '07', '10'},
                selected=any(a['case'] == case.name for a in spec['records'])))
    dump(HERE/'inventory.json', inventory)
    cases, frames, records, cards = {}, {}, [], []
    for a in spec['records']:
        name = a['case']
        if name not in cases:
            case = root/name
            s = json.loads((case/'sim/summary.json').read_text())
            request = json.loads((case/'controller/request_000.json').read_text())
            video = case/'sim/sensors.mp4'
            info = probe(video)
            assert s['control_dt'] == .04 and info['r_frame_rate'] == '25/1'
            assert int(info['nb_frames']) == s['step_id'] + 1 == s['video_frames']
            cases[name] = dict(directory=str(case), episode_id=s['episode_id'],
                instruction=request['instruction'], native_success=s['success'],
                case_identity=s['evaluation_case'], source_video=str(video),
                source_video_sha256=digest(video), video=info, control_dt=s['control_dt'])
        c = cases[name]
        steps = [round(t / c['control_dt']) for t in a['times']]
        assert steps == sorted(set(steps))
        assert all(abs(t-step*.04) < 1e-6 for t,step in zip(a['times'],steps))
        cameras = a.get('cameras', CAMERAS)
        text = POLICY + '\nOriginal task: ' + c['instruction'] + '\nCurrent subgoal: ' + a['subgoal'] + '\nSuccess condition: ' + a['success']
        content = [{'type':'text', 'text':text}]
        assets, provenance, tiles = [], [], []
        for step,t in zip(steps, a['times']):
            key = (name, step)
            if key not in frames:
                src = Path(c['directory'])/'sim'/c['episode_id']/'observations'/f'{step:06d}.npz'
                frames[key] = {'source_path':str(src), 'source_sha256':digest(src), 'images':{}}
            f = frames[key]
            with np.load(f['source_path'], allow_pickle=False) as raw:
                for camera in cameras:
                    path = Path('assets') / c['episode_id'] / f'{step:06d}_{camera}.png'
                    dest = HERE/path
                    if camera not in f['images']:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        rgb = raw[camera]
                        assert rgb.shape == (480,640,3) and rgb.dtype == np.uint8
                        Image.fromarray(rgb).save(dest)
                        f['images'][camera] = {'path':path.as_posix(), 'sha256':digest(dest)}
                    asset = f['images'][camera]
                    assets.append(asset)
                    content.extend([{'type':'text','text':f'Simulation time {t:.2f} s; step {step}; camera {camera}.'},
                        {'type':'image_url','image_url':{'url':asset['path']}}])
                    if camera == cameras[0]:
                        tiles.append((Image.fromarray(raw[camera]).resize((320,240)), t, step))
            provenance.append({'sim_time_s':t,'step':step,'source_npz':f['source_path'],
                               'source_npz_sha256':f['source_sha256'],'cameras':cameras})
        targets = {'execution_state': target(STATUS, a['state'])}
        for qid, value in [('request_review',a['trigger']),('subgoal_complete',a['complete'])]:
            if value is not None:
                targets[qid] = target(['true','false'], str(value).lower())
        meta = dict(record_id=a['id'], domain='robodojo_manipulation', modality='image', language_bucket='en',
            annotation_status=spec['annotation_status'], confidence=a['confidence'], title_zh=a['title'],
            rationale_zh=a['reason'], source_run=spec['source_run'], source_case=name,
            source_episode_id=c['episode_id'], sim_start_s=a['times'][0], decision_sim_time_s=a['times'][-1],
            sim_steps=steps, cameras=cameras, observation_variant=a.get('observation_variant','native_three_camera_temporal'),
            context_source='offline_annotator_supplied_subgoal_and_condition_not_logged_policy_intent',
            context_is_oracle=True, event=a['event'], source_observations=provenance,
            split='unsplit_pilot_do_not_report_generalization', sample_interval_is_inclusive=True)
        event = a['event']
        if event:
            meta['event']['runtime_25_step_first_possible_endpoint_s'] = int(np.ceil(event['first_confirmed_s']))
            meta['event']['review_window_s'] = [event['first_confirmed_s'], a['times'][-1]]
            meta['event']['window_is_proposed_not_validated_deadline'] = True
        record = {'group_id':c['episode_id'],'request':{'state':{'messages':[{'role':'user','content':content}]},
                  'questions':QUESTIONS},'targets':targets,'assets':assets,'meta':meta}
        records.append(record)
        # Review-only labelled contact sheet; never part of request.state.
        preview = Path('previews')/f"{a['id']}.jpg"
        (HERE/preview).parent.mkdir(exist_ok=True)
        sheet = Image.new('RGB',(960,272*((len(tiles)+2)//3)), '#ffffff')
        draw = ImageDraw.Draw(sheet)
        for i,(img,t,step) in enumerate(tiles):
            x=i%3*320; y=i//3*272
            sheet.paste(img,(x,y));draw.text((x+6,y+246),f"{a['id']} | {t:.2f}s | step {step}",fill='black')
        sheet.save(HERE/preview,quality=92)
        clip = None
        if len(steps)>1:
            clip = Path('clips')/f"{a['id']}.mp4"
            (HERE/clip).parent.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='jev-pilot-encode-') as temporary:
                encoded = Path(temporary)/'clip.mp4'
                subprocess.run([FFMPEG,'-nostdin','-v','error','-y','-threads','1','-filter_threads','1','-i',c['source_video'],
                    '-vf',f'trim=start_frame={steps[0]}:end_frame={steps[-1]+1},setpts=PTS-STARTPTS',
                    '-an','-c:v','libx264','-threads','1','-preset','ultrafast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(encoded)],check=True,timeout=30)
                assert int(probe(encoded)['nb_frames']) == steps[-1]-steps[0]+1
                staged = (HERE/clip).with_suffix('.tmp.mp4')
                shutil.copyfile(encoded, staged)
                staged.replace(HERE/clip)
        # Isolated uncertain sample shows only its actual single-view input.
        label = '证据不足（不提供二元监督）' if a['trigger'] is None else ('请求检查' if a['trigger'] else '继续 / 子目标完成')
        video_html = f'<video controls preload="none" src="{clip.as_posix()}"></video>' if clip else ''
        input_links = ' '.join(f'<a href="{v["path"]}">{i+1}</a>' for i,v in enumerate(assets))
        cards.append(f'<article id="{a["id"]}" data-kind="{a["trigger"]}"><h2>{a["id"]} · {html.escape(a["title"])}</h2>'
            f'<p><b>{label}</b> · {html.escape(name)} · {a["times"][0]}–{a["times"][-1]} s</p>'
            f'<p>当前动作：{html.escape(a["subgoal"])}</p><p>成功条件：{html.escape(a["success"])}</p>'
            f'<a href="{preview}"><img src="{preview}" loading="lazy"></a>{video_html}'
            f'<p>{html.escape(a["reason"])}</p><p>状态：{a["state"]}；标注信心：{a["confidence"]}；待人工复核。</p>'
            f'<p>原始输入图片：{input_links}</p><details><summary>时间边界及来源</summary><pre>{html.escape(json.dumps(meta,ensure_ascii=False,indent=2))}</pre></details></article>')
        meta['review_assets'] = {'contact_sheet':str(preview),'video':str(clip) if clip else None,
            'video_note':'Review-only full-rate three-camera recording; model receives listed native PNG frames only.' if clip else 'Only the single input camera is shown.'}
        print(a['id'], a['state'], len(assets), 'images', flush=True)
    with (HERE/'samples.jsonl').open('w',encoding='utf-8') as f:
        for r in records: f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    dump(HERE/'sources.json', {'format_reference':spec['format_reference'], 'source_root':str(root),
        'camera_order_in_recorded_video':CAMERAS, 'time_mapping':'frame_index = simulation_step; simulation_seconds = step * 0.04; initial observation is frame 0',
        'cases':cases})
    counts = Counter(a['state'] for a in spec['records'])
    (HERE/'review.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jev / RoboDojo 标注样例</title><style>body{max-width:1120px;margin:36px auto;padding:0 20px;background:#f3f5f7;color:#182330;font:16px/1.65 system-ui}article{background:white;padding:24px;margin:24px 0;border-radius:12px}img,video{display:block;width:100%;height:auto;margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}a{color:#165da8}button{padding:8px 16px;margin:4px;border:1px solid #ccd3db;border-radius:6px;background:white}h1{font-size:30px}h2{font-size:21px}</style>
<h1>Jev / RoboDojo：12 个标注样例</h1>
<p>4 条纯 π₀.₅ 仿真轨迹 · 6 个请求检查 · 5 个正常进行或完成 · 1 个证据不足。模型视觉标注初稿，待人工复核。</p>
<p>动作与成功条件是离线补充的监测上下文。图集展示输入时间点的主视角；输入图片链接包含实际送入模型的全部视角。视频保留对应时间段的三视角连续记录，仅供复核。所有时间均为仿真秒。</p>
<p>判断针对片段最后时刻；请求检查不等于急停，也不证明介入一定有效。证据不足样例刻意仅保留单帧腕部近景，其二元标签缺省。</p>
<p><a href="samples.jsonl">Valen JSONL</a> · <a href="annotations.json">标注源文件</a> · <a href="README.md">标注说明</a></p>
<nav><button onclick="filter('all')">全部</button><button onclick="filter('True')">请求检查</button><button onclick="filter('False')">正常 / 恢复</button><button onclick="filter('None')">证据不足</button></nav>
''' + '\n'.join(cards) + '''<script>function filter(k){document.querySelectorAll('article').forEach(x=>x.hidden=k!=='all'&&x.dataset.kind!==k)}</script></html>''',encoding='utf-8')
    dump(HERE/'build_summary.json', {'records':len(records),'groups':len(cases),'status_counts':counts,
        'labelled_questions':sum(len(r['targets']) for r in records),'unique_input_assets':len({x['path'] for r in records for x in r['assets']}),
        'source_npz_files':len(frames),'train_validation_test_split_created':False,'model_called':False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root',type=Path,default=Path('/mnt/data_hdd/mhzhu/JevHarnessEnv/runs/robodojo-ten-pi05-only-20261013'))
    build(parser.parse_args().source_root)
