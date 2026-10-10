"""Independent LAN audio test app. Every result comes from the NAS model worker."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import ipaddress
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import wave
from urllib.parse import urlparse


def write_json(path,value):
    temp=path.with_suffix('.pending');temp.write_text(json.dumps(value,ensure_ascii=False));temp.replace(path)


def serve(root,run,host,port):
    def nas_online():
        if not Path('/srv/nas/.nas-online').is_file():return False
        if os.environ.get('CINEDUPLEX_CONTAINER_NAS')=='1':
            # A bind mount retains the real Btrfs filesystem. Refuse a root-disk
            # directory even when Docker recreates a missing source on restart.
            result=subprocess.run(['findmnt','-rn','-o','FSTYPE','-T',str(root)],capture_output=True,text=True,timeout=3)
            identity=json.loads((root/'.cineduplex-nas-identity.json').read_text())
            return result.returncode==0 and result.stdout.strip()=='btrfs' and identity['uuid']==os.environ.get('CINEDUPLEX_NAS_UUID')
        result=subprocess.run(['findmnt','-rn','-o','UUID','-T',str(root)],capture_output=True,text=True,timeout=3)
        return result.returncode==0 and result.stdout.strip()==os.environ.get('CINEDUPLEX_NAS_UUID')
    if not nas_online():raise RuntimeError('real NAS unavailable; no local fallback')
    work=root/run; web=work/'web';web.mkdir(exist_ok=True)
    for child in ('queue','jobs'): (web/child).mkdir(exist_ok=True)
    examples=[json.loads(x) for x in (work/'eval.jsonl').read_text().splitlines()[:8]]
    def ready():
        last={}
        for p in (web/'resident/SERVICE.json',web/'SERVICE.json'):
            try:
                v=json.loads(p.read_text());last=v
                if v.get('model_ready') is True and time.time()-v.get('heartbeat',0)<45:return True,v
            except (OSError,ValueError):pass
        return False,last
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def allowed(self):
            ip=ipaddress.ip_address(self.client_address[0])
            return ip.is_loopback or any(ip in ipaddress.ip_network(n) for n in ('192.168.12.0/24','192.168.11.0/24','192.168.9.0/24'))
        def send(self,status,data,kind='application/json'):
            if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if not self.allowed():return self.send(403,{'error':'仅限局域网'})
            path=urlparse(self.path).path
            if path=='/':return self.send(200,Path(__file__).with_name('index.html').read_bytes(),'text/html; charset=utf-8')
            if path=='/api/status':
                try:
                    s=json.loads((work/'status.json').read_text())
                    live=work/'attempt-01/training/LIVE.json'
                    if live.exists():
                        metrics=json.loads(live.read_text())
                        if s.get('phase')=='FAILED':metrics.pop('phase',None)
                        s.update(metrics)
                    if (work/'attempt-01/GPU-TRAINING-COMPLETE.json').exists():s['phase']='TRAINING_COMPLETE'
                    ok,service=ready();s.update(model_ready=ok,service=service,run_id=run.split('/')[-1])
                    return self.send(200,s)
                except OSError:return self.send(503,{'phase':'NAS_UNAVAILABLE','model_ready':False})
            if path=='/api/examples':return self.send(200,[{'id':i,'context_text':r.get('context_text',''),
                'target_text':r.get('target_text',''),'input_url':f'/api/example/{i}/context',
                'target_url':f'/api/example/{i}/target'} for i,r in enumerate(examples)])
            pieces=path.strip('/').split('/')
            if len(pieces)==4 and pieces[:2]==['api','example']:
                try:
                    r=examples[int(pieces[2])];key={'context':'user_audio_path','target':'target_audio_path'}[pieces[3]]
                    p=root/r[key].removeprefix('/Volumes/media/CineDuplex/')
                    if not p.resolve().is_relative_to(root.resolve()):raise ValueError('unsafe source')
                    return self.send(200,p.read_bytes(),'audio/wav')
                except (KeyError,IndexError,ValueError,OSError):return self.send(404,{'error':'样本不可用'})
            if len(pieces) in (3,4) and pieces[:2]==['api','jobs']:
                try:
                    jid=str(uuid.UUID(pieces[2]));folder=web/'jobs'/jid
                    if len(pieces)==3:return self.send(200,json.loads((folder/'status.json').read_text()))
                    if pieces[3] not in ('input','base','trained'):raise ValueError('unknown variant')
                    return self.send(200,(folder/(pieces[3]+'.wav')).read_bytes(),'audio/wav')
                except (ValueError,OSError):return self.send(404,{'error':'结果尚未生成'})
            self.send(404,{'error':'不存在'})
        def do_POST(self):
            if not self.allowed():return self.send(403,{'error':'仅限局域网'})
            if not nas_online():return self.send(503,{'error':'存储暂不可用'})
            origin=self.headers.get('Origin')
            if origin and urlparse(origin).netloc!=self.headers.get('Host'):return self.send(403,{'error':'来源不符'})
            if urlparse(self.path).path!='/api/jobs':return self.send(404,{'error':'不存在'})
            ok,service=ready()
            if not ok:return self.send(503,{'error':'模型尚未就绪，请等训练及加载完成'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 44<size<=400000:raise ValueError('音频最多12秒')
                blob=self.rfile.read(size)
                with wave.open(io.BytesIO(blob)) as f:
                    if (f.getframerate(),f.getnchannels(),f.getsampwidth())!=(16000,1,2):raise ValueError('请选择16kHz单声道PCM16 WAV')
                    seconds=f.getnframes()/16000
                    if not .3<=seconds<=12:raise ValueError('音频长度应为0.3到12秒')
                    if len(f.readframes(f.getnframes()))!=f.getnframes()*2:raise ValueError('音频文件不完整')
                queue=web/'resident/queue' if service.get('namespace')=='resident' else web/'queue'
                if len(list(queue.glob('*.json')))>=8:return self.send(429,{'error':'测试队列已满，请稍后'})
                jid=str(uuid.uuid4());folder=web/'jobs'/jid;folder.mkdir()
                (folder/'input.wav').write_bytes(blob)
                status={'id':jid,'state':'QUEUED','created_at':time.time(),'input_seconds':seconds,
                    'input_sha256':hashlib.sha256(blob).hexdigest(),'model':service.get('model'),
                    'input_url':f'/api/jobs/{jid}/input','quality_approved':False}
                write_json(folder/'status.json',status);write_json(queue/(jid+'.json'),{'id':jid})
                return self.send(202,status)
            except (ValueError,wave.Error,EOFError) as e:return self.send(400,{'error':str(e)})
    ThreadingHTTPServer((host,port),Handler).serve_forever()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',required=True)
    p.add_argument('--host',default='192.168.12.10');p.add_argument('--port',type=int,default=18086)
    a=p.parse_args();serve(a.root,a.run,a.host,a.port)
