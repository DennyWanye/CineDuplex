#!/usr/bin/env python3
"""Summarize a completed NAS call. Objective measurements, never listening approval."""
import argparse,array,hashlib,json,math,statistics,wave
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--call',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--fixture',type=Path);a=p.parse_args()
 if a.output.exists():raise RuntimeError('preserve prior audit')
 folder=a.root/'calls'/a.call
 def rows(name):
  path=folder/name
  if path.stat().st_size>8*1024*1024:raise ValueError('bounded metadata budget exceeded')
  return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
 e=rows('events.jsonl');c=rows('client.jsonl');summary=json.loads((folder/'summary.json').read_text())
 m=[x for x in e if x['type']=='metrics'];s=[x for x in m if x['state'] in ('S','B')];audio=[x for x in e if x['type']=='audio'];starts={x['event_id']:x['at'] for x in e if x['type']=='audio_start'}
 replies=[]
 for key,start in starts.items():
  chunks=[x for x in audio if x['event_id']==key];schedule=[x for x in c if x['type']=='audio_schedule' and x['event_id']==key]
  cancelled_starts={x['scheduled_start'] for x in c if x['type']=='audio_cancel' and x.get('event_id')==key and x.get('reason')=='complete_before_playback'}
  schedule=[x for x in schedule if x['scheduled_start'] not in cancelled_starts]
  text=''.join(x['text'] for x in e if x['type']=='text_delta' and x.get('event_id')==key)
  replies.append({'id':key,'text':text,'server_first_audio_after_model_start_s':round(chunks[0]['at']-start,3) if chunks else None,'scheduled_first_audio_after_model_start_s':round(chunks[0]['at']-start+schedule[0]['ahead_seconds']+(schedule[0]['at']-next(x['at'] for x in c if x['type']=='audio_receive' and x['event_id']==key)),3) if schedule and chunks else None,'output_seconds':sum(x['samples']/24000 for x in chunks),'chunks':len(chunks),'scheduling_clock_caveat':'server duration plus client buffer duration; network transit and physical sound not measured'})
 pcm=(folder/'output.pcm').read_bytes()
 if len(pcm)>48*1024*1024:raise ValueError('bounded output budget')
 values=array.array('h');values.frombytes(pcm)
 rms=math.sqrt(sum(x*x for x in values)/max(1,len(values)))/32768
 report={'call':a.call,'summary':summary,'errors':[x for x in e if x['type']=='error'],'client_errors':[x for x in c if x['type']=='client_error'],'rebuffer':[x for x in c if x['type']=='rebuffer'],'stop':[x for x in c if x['type']=='call_stop'][-1:],'responses':replies,'speaking_mean_ms':statistics.mean(x['compute_ms'] for x in s) if s else None,'speaking_max_ms':max((x['compute_ms'] for x in s),default=None),'dispatch_mean_ms':statistics.mean(x.get('tp_dispatch_ms',0) for x in m) if m else None,'dispatch_max_ms':max((x.get('tp_dispatch_ms',0) for x in m),default=None),'pending_peak_seconds':max((x['pending_seconds'] for x in m),default=0),'last_gpu_memory_mb':m[-1].get('cuda_allocated_mb') if m else None,'context_used':m[-1].get('context_used') if m else None,'codec_mean_ms':statistics.mean(x['compute_ms'] for x in e if x['type']=='codec_metrics'),'audio_objective':{'pcm_sha256':hashlib.sha256(pcm).hexdigest(),'seconds':len(pcm)/48000,'rms_dbfs':20*math.log10(max(1e-9,rms)),'peak':max((abs(x)/32768 for x in values),default=0),'clipped_samples':sum(abs(x)>=32735 for x in values),'max_chunk_boundary_jump':max((x.get('boundary_jump') or 0 for x in audio),default=0)},'subjective_listening_pass':False,'real_human_speech_test':False}
 report['slow_windows']=[{key:x.get(key) for key in ('input_seq','compute_ms','tp_dispatch_ms','state','context_used','pending_seconds','stages')} for x in sorted(m,key=lambda x:x['compute_ms']+x.get('tp_dispatch_ms',0),reverse=True)[:8]]
 report['gc_pause_max_ms']=max((x.get('stages',{}).get('host_gc',{}).get('pause_ms',0) for x in m),default=0)
 if a.fixture:
  fixture=json.loads(a.fixture.read_text());turns=fixture['turns']
  # Derive server receipt of the first replay packet from its measured input
  # age. These are source-timeline estimates, not physical human/microphone
  # end-to-end latency, and are labelled separately from model-start latency.
  first=next(x for x in m if x.get('input_seq')==0)
  replay_start=first['at']-first['input_age_ms']/1000
  for i,reply in enumerate(replies):
   if i>=len(turns):break
   turn=turns[i];end=turn.get('at',turn.get('start_seconds'))+turn.get('seconds',turn.get('duration_seconds'))
   reply['fixture_prompt']=turn['text']
   reply['recorded_prompt_end_to_server_audio_s']=round(starts[reply['id']]+reply['server_first_audio_after_model_start_s']-replay_start-end,3) if reply['server_first_audio_after_model_start_s'] is not None else None
   reply['recorded_prompt_end_to_scheduled_audio_estimate_s']=round(starts[reply['id']]+reply['scheduled_first_audio_after_model_start_s']-replay_start-end,3) if reply['scheduled_first_audio_after_model_start_s'] is not None else None
  report['fixture_latency_caveat']='real-time prerecorded source timeline; omits network transit and physical speaker; not human end-to-end'
 a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
 wav=a.output.with_suffix('.wav')
 with wave.open(str(wav),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(24000);f.writeframes(pcm)
 print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
