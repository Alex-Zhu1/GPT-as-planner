"""Side-by-side original live recordings; ended episodes explicitly freeze."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root.resolve()
summary=json.loads((root/'summary.json').read_text());details=json.loads((root/'case-details.json').read_text())
out=root/'representative-videos';out.mkdir(exist_ok=True)
index=[]
for pair in summary['representative_pairs']:
    modes=['none','fixed','jev'];ids=[pair+'-'+m for m in modes]
    readers=[imageio.get_reader(root/'run/episodes'/eid/'execution.mp4') for eid in ids]
    counts=[details[eid]['video']['frames'] for eid in ids]
    last=[None]*3;max_count=max(counts);fps=20
    destination=out/(pair+'.mp4')
    with imageio.get_writer(destination,fps=fps,codec='libx264',macro_block_size=1,
            ffmpeg_params=['-crf','20','-pix_fmt','yuv420p','-movflags','+faststart']) as writer:
        for i in range(max_count):
            canvas=Image.new('RGB',(1536,360),'#101826');draw=ImageDraw.Draw(canvas)
            for j,eid in enumerate(ids):
                if i<counts[j]:last[j]=readers[j].get_next_data()
                frame=Image.fromarray(last[j]).resize((512,306))
                canvas.paste(frame,(j*512,54));r=details[eid]['row']
                text=f"{modes[j].upper()} | {'SUCCESS' if r['success'] else 'FAIL'} | {r['action_steps']} controls | {r['astra_calls']} Astra calls"
                draw.text((j*512+8,7),text,fill='white')
                draw.text((j*512+8,28),f"{pair} | control {min(i,counts[j]-1)} | "+('ENDED: final frame held' if i>=counts[j] else 'LIVE RECORDING'),fill='#90cfff')
            writer.append_data(np.asarray(canvas))
        canvas.save(out/(pair+'-final.jpg'))
    for reader in readers:reader.close()
    index.append(dict(pair_id=pair,path=str(destination),frames=max_count,fps=fps,
        note='Three original live recordings at simulation speed. Finished arms hold their last frame with ENDED label.',
        arms=[details[eid]['row'] for eid in ids]))
(out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps(index,indent=2))
