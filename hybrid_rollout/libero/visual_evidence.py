"""Bounded public-image evidence, shared by both local Astra planner paths."""
from pathlib import Path

VISUAL_GUIDANCE = (
    'VISUAL EVIDENCE REVIEW: Combine the current two camera images, historical keyframes, '
    'and task_context status board. First assess the current scene, then compare matching '
    'camera views across observation IDs and steps, then reconcile the board with those images. '
    'Current images establish the current visible state; historical images establish changes, '
    'not current facts. Distinguish directly visible evidence, measured execution records and '
    'your inferences; cite observation IDs for each material claim in the assessment text. '
    'Image manifest references identify the actual attached pixels. A board image path alone '
    'does not mean that image was attached or seen. Only cite attached observations as visual evidence. '
    'visual_change_before/after are selected by pixel difference, NOT a detected perturbation: '
    'camera motion, lighting and robot motion may explain the difference. Compare these candidates '
    'and eef_before/after with last_decision to reassess what changed and whether the current '
    'instruction still applies. Jev does not interpret these frames. Prior Astra assessments '
    'and Valen scores may be stale or wrong; never carry forward an old judgment without checking '
    'new evidence. Explicitly state contradictions, occlusion, missing views and uncertainty. '
    'Do not infer grasp, task success or achieved motion from commands or nominal previews. '
)


def pixel_change(before, after):
    """Mean thumbnail RGB difference; no objects, simulator labels or semantics."""
    import numpy as np
    from PIL import Image
    previous={i['camera']:i['path'] for i in before.get('images',[])}
    values=[]
    for item in after.get('images',[]):
        if item['camera'] not in previous:
            continue
        arrays=[]
        for path in (previous[item['camera']],item['path']):
            with Image.open(path) as im:
                arrays.append(np.asarray(im.convert('RGB').resize((32,32)),dtype=np.float32))
        values.append(float(np.abs(arrays[1]-arrays[0]).mean()/255.))
    return max(values,default=0.)


def evidence_images(request, root):
    root=Path(root).resolve()
    current=request['current_scene']
    if len(current.get('images',[]))!=2:
        raise ValueError('LIBERO current_scene requires both actual cameras')
    board=request.get('task_context',{})
    frames=[('current_scene',current)]
    keyframes=board.get('keyframes',{})
    # Preserve event pairs and the prior decision before ordinary temporal sampling.
    for role in ('visual_change_before','visual_change_after','last_decision','last_decision_after','eef_before','eef_after'):
        if role in keyframes:
            frames.append((role,keyframes[role]))
    trigger_frames=request.get('failure',{}).get('evidence',[])
    if trigger_frames:
        for i in sorted({0,len(trigger_frames)//2,len(trigger_frames)-1}):
            frames.append((f'trigger_window_{i}',trigger_frames[i]))
    history=request.get('evidence_frames',[])
    if request.get('review_kind')=='progress':
        history=[f for f in history if f['step_id']>=request['failure']['first_step']]
        history=[history[i] for i in sorted({0,len(history)//2,max(0,len(history)-2)})] if history else []
    else:
        history=history[-2:]
    frames += [(f'history_{i}_step_{f["step_id"]}',f) for i,f in enumerate(history)]
    frames += [(f'episode_context_step_{f["step_id"]}',f) for f in board.get('evidence_frames',[])[:3]]
    result=[];seen={}
    for role,frame in frames:
        for item in frame.get('images',[]):
            path=Path(item['path']).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError('Evidence image must be a file inside evidence-root')
            ref={k:frame.get(k) for k in ('observation_id','step_id','captured_at','plan_revision')}
            ref.update(role=role,camera=item['camera'])
            if path in seen:
                if ref not in seen[path]['references']:
                    seen[path]['references'].append(ref)
                continue
            row=dict(label=role,camera=item['camera'],path=str(path),**{k:ref[k] for k in ('observation_id','step_id','captured_at','plan_revision')},references=[ref])
            seen[path]=row;result.append(row)
    return result
