"""Run OneVL or Alpamayo inference in its model environment."""
import argparse
import json
import math
import re
import time
import traceback
from pathlib import Path
from multiprocessing.connection import Connection

LATENT_PREFIX='<|start-latent-vis|>'+'<|latent-vis|>'*4+'<|end-latent-vis|><|start-latent|>'+'<|latent|>'*2+'<|end-latent|><answer>'


def parse_plan(answer):
    number=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    missing=bool(re.match(r'\s*'+number+r'\s*,\s*'+number+r'\s*\]',answer))
    text='['+answer.lstrip() if missing else answer
    points=[[float(a),float(b)]for a,b in re.findall(r'\[\s*('+number+r')\s*,\s*('+number+r')\s*\]',text)]
    return (points if len(points)==10 and all(math.isfinite(x)for p in points for x in p)else None),missing


def sampling_seed(seed,request):
    return seed+request['request_id']*1000003+request['inference_index']


class Planner:
    def __init__(self,cfg):
        import torch,transformers
        self.torch=torch;self.cfg=cfg;self.kind=cfg['model_kind'];self.generation={}
        torch.manual_seed(cfg.get('seed',7));torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if self.kind=='onevl':
            from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
            self.model,loading=Qwen3VLForConditionalGeneration.from_pretrained(cfg['model_path'],dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
            unexpected=[k for k in loading.get('unexpected_keys',[])if not k.startswith('_latent_cot_')]
            self.processor=AutoProcessor.from_pretrained(cfg['model_path'],local_files_only=True)
            self.processor.image_processor.max_pixels=1792*1792
            self.processor.image_processor.size['longest_edge']=1792*1792
            self.model.generation_config.do_sample=False
            self.model.generation_config.temperature=self.model.generation_config.top_p=self.model.generation_config.top_k=None
        else:
            from alpamayo1_5 import helper
            from alpamayo1_5.config import Alpamayo1_5Config
            from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
            self.helper=helper;root=Path(cfg['model_path'])
            raw=json.loads((root/'model/config.json').read_text());raw.update(vlm_name_or_path=str(root/'base-config'),attn_implementation='sdpa')
            self.model,loading=Alpamayo1_5.from_pretrained(root/'model',config=Alpamayo1_5Config(**raw),dtype=torch.bfloat16,local_files_only=True,output_loading_info=True)
            unexpected=loading.get('unexpected_keys',[])
            helper.BASE_PROCESSOR_NAME=str(root/'processor');self.processor=helper.get_processor(self.model.tokenizer)
            generate=self.model.vlm.generate
            def capture(*args,**kwargs):
                result=generate(*args,**kwargs);tokens=result.sequences[:,kwargs['input_ids'].shape[-1]:]
                marker=self.model.tokenizer.convert_tokens_to_ids('<|traj_future_start|>')
                self.generation=dict(output_tokens=int(tokens.shape[-1]),trajectory_marker_present=bool((tokens==marker).any().item()))
                return result
            self.model.vlm.generate=capture
        if unexpected or any(loading.get(k)for k in ['missing_keys','mismatched_keys','error_msgs']):raise RuntimeError('Checkpoint mismatch: '+str(loading))
        self.model=self.model.to('cuda').eval();self.model.requires_grad_(False)
        self.description=dict(torch=torch.__version__,transformers=transformers.__version__,kind=self.kind,loading=loading,precision='bfloat16',attention='sdpa',seed=cfg.get('seed',7))

    def infer(self,request):
        import numpy as np
        from PIL import Image
        from benchmark.policies.vlm_inputs import onevl_motion_text
        torch=self.torch;started=time.monotonic()
        images=[Image.fromarray(np.frombuffer(x['data'],np.uint8).reshape(x['shape']))for x in request['images']]
        if self.kind=='onevl':
            prefix=('You are an autonomous driving agent. You have access to multi-view camera images of a vehicle: (1) front view (which you should focus on with the most attention), (2) front right view, and (3) front left view. '
                "Your task is to predict future waypoints following the assigned route. Provided are the previous ego vehicle statuses recorded over the last 1.5 seconds (at 0.5-second intervals). Positive x means forward direction while positive y means leftwards.\n")
            motion,encoding=onevl_motion_text(request['states'],self.cfg.get('onevl_motion_encoding','numeric_history'))
            prompt=prefix+motion+'\nAssigned route: '+request['navigation']+'\nPredict exactly 10 future [x, y] waypoints in metres in the current ego frame at t+0.5, t+1.0, t+1.5, t+2.0, t+2.5, t+3.0, t+3.5, t+4.0, t+4.5, and t+5.0 seconds. Do not include t=0.\n'
            content=[dict(type='image',image=x)for x in images]+[dict(type='text',text=prompt)]
            text=self.processor.apply_chat_template([dict(role='user',content=content)],tokenize=False,add_generation_prompt=True)+LATENT_PREFIX
            batch=self.processor(text=[text],images=images,return_tensors='pt',padding=True).to('cuda')
            with torch.inference_mode():output=self.model.generate(**batch,max_new_tokens=1024,do_sample=False)
            tokens=output[:,batch.input_ids.shape[1]:]
            answer=self.processor.batch_decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)[0]
            plan,normalized=parse_plan(answer);valid=plan is not None and tokens.shape[1]<1024
            result=dict(valid=valid,trajectory_xy_m=plan,dt_s=.5,text=answer,prompt=prompt,motion_encoding=encoding,format_normalization='leading_bracket' if normalized else 'none',output_tokens=int(tokens.shape[1]))
        else:
            helper=self.helper
            messages=helper.create_message(torch.zeros((len(images),3,1,1),dtype=torch.uint8),camera_indices=torch.tensor([0,1,2]),nav_text=request['navigation'])
            items=[x for m in messages for x in m['content']if x['type']=='image']
            for item,image in zip(items,images):item['image']=image
            batch=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,continue_final_message=True,return_dict=True,return_tensors='pt')
            data=helper.to_device(dict(tokenized_data=batch,ego_history_xyz=torch.tensor(request['ego_history_xyz'],dtype=torch.float32)[None,None],ego_history_rot=torch.tensor(request['ego_history_rot'],dtype=torch.float32)[None,None]),'cuda')
            # Seed each prediction from the run seed, request ID and inference index.
            seed=sampling_seed(self.cfg.get('seed',7),request)
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                xyz,rot,extra=self.model.sample_trajectories_from_data_with_vlm_rollout(data=data,top_p=.98,temperature=.6,num_traj_samples=1,max_generation_length=256,return_extra=True)
            trajectory=xyz[0,0,0].float().cpu().numpy()
            valid=trajectory.shape==(64,3) and bool(np.isfinite(trajectory).all()) and self.generation['trajectory_marker_present']
            result=dict(valid=valid,trajectory_xy_m=trajectory[:,:2].tolist() if np.isfinite(trajectory).all() else None,dt_s=.1,
                native_trajectory_xyz_m=trajectory.tolist() if np.isfinite(trajectory).all() else None,text={k:np.asarray(v).tolist()for k,v in extra.items()},sampling_seed=seed,**self.generation)
        torch.cuda.synchronize()
        return dict(result,inference_wall_s=time.monotonic()-started,input_tokens=int(batch['input_ids'].shape[-1]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--fd',type=int,required=True);args=p.parse_args()
    conn=Connection(args.fd)
    try:
        planner=Planner(conn.recv());conn.send(dict(ready=True,description=planner.description))
        while True:
            request=conn.recv()
            if request is None:break
            conn.send(planner.infer(request))
    except EOFError:pass
    except Exception:
        detail=traceback.format_exc();print(detail,flush=True)
        conn.send(dict(error=detail))
    finally:conn.close()

if __name__=='__main__':main()
