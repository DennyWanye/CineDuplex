"""LAN-only realtime UI, with HTTPS and a localhost development entry point."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
import ssl
import socket
import sys
import threading
import uuid
import time
import urllib.request
import urllib.error
from urllib.parse import urlparse


def serve(root,cert,key,host,port,plain_port,upstream):
    context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.minimum_version=ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(cert,key)
    # Read the existing TLS key as root, then serve and write NAS logs as family.
    if os.getuid()==0:
        os.setgroups([]);os.setgid(990);os.setuid(990)
    log=(root.parent/(root.name+'.log')).open('a',buffering=1)
    sys.stdout=log;sys.stderr=log
    if not Path('/srv/nas/.nas-online').is_file():raise RuntimeError('NAS unavailable')
    diagnostic_lock=threading.Lock();diagnostic_seen={}
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def setup(self):
            super().setup();self.connection.settimeout(25)
            self.connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        def log_message(self,*a):pass
        def allowed(self):
            ip=ipaddress.ip_address(self.client_address[0])
            return ip.is_loopback or any(ip in ipaddress.ip_network(x) for x in ('192.168.12.0/24','192.168.11.0/24','192.168.9.0/24'))
        def send(self,code,body,kind='application/json'):
            if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False).encode()
            self.send_response(code);self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Permissions-Policy','microphone=(self), camera=()')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; media-src 'self' blob:; worker-src 'self'")
            self.end_headers();self.wfile.write(body)
        def diagnostics(self,call_id):
            folder=root.parent/'calls'/str(uuid.UUID(call_id))
            if not folder.is_dir():return self.send(404,{'error':'通话不存在'})
            if self.command=='GET':
                def rows(name):
                    path=folder/name
                    if not path.exists():return []
                    if path.stat().st_size>8*1024*1024:raise ValueError('log exceeds download budget')
                    return [json.loads(x) for x in path.read_text().splitlines(keepends=True) if x.endswith('\n') and x.strip()]
                events=rows('events.jsonl');client=rows('client.jsonl')
                summary=json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else {'active':True}
                return self.send(200,{'schema':2,'call_id':call_id,'summary':summary,
                    'server_events':events,'client_events':client,'audio_files':{
                        'input':'input.pcm / PCM16 mono 16000Hz','output':'output.pcm / PCM16 mono 24000Hz',
                        'raw_codec':'codec-raw.pcm / float32 mono 24000Hz'},
                    'meaning':'Browser scheduled/ended events establish transport, not listening quality.'})
            size=int(self.headers.get('Content-Length',0))
            if not 0<size<=32000:raise ValueError('diagnostics size')
            batch=json.loads(self.rfile.read(size));entries=batch.get('events',[])
            if not isinstance(entries,list) or len(entries)>64:raise ValueError('diagnostics batch')
            kinds={'client_config','audio_receive','audio_schedule','audio_played','audio_cancel','initial_buffer','rebuffer',
                'transport_sample','client_error','call_stop','rtc_state','mark_problem'}
            allowed={'type','client_seq','at','elapsed_ms','event_id','chunk_seq','server_seq','server_at',
                'samples','sample_rate','context_time','scheduled_start','duration','ahead_seconds','source_nodes',
                'pending_replies','send_queue','upload_ms','poll_wait_ms','diagnostics_dropped','rms','state','base_latency','output_latency',
                'echo_cancellation','noise_suppression','auto_gain_control','context_sample_rate','input_sample_rate',
                'precision','runtime_sha256','tensor_parallel_size','voice','language','reason','message','jitter','jitter_buffer_delay',
                'packets_lost','concealed_samples','total_samples_received','reference_connected','count','played_packets',
                'received_packets','sent_packets','muted','browser','output_device','lead_seconds','producer_ratio'}
            # Validate the whole batch before appending anything. Retries and
            # web service restarts must not duplicate diagnostic sequence IDs.
            for entry in entries:
                if not isinstance(entry,dict) or entry.get('type') not in kinds:raise ValueError('diagnostic type')
                if set(entry)-allowed or len(json.dumps(entry))>4000:raise ValueError('diagnostic fields')
                if type(entry.get('client_seq')) is not int or entry['client_seq']<0:raise ValueError('diagnostic sequence')
            with diagnostic_lock:
                path=folder/'client.jsonl';size=path.stat().st_size if path.exists() else 0
                if size>8*1024*1024:raise ValueError('diagnostic budget')
                last=diagnostic_seen.get(call_id)
                if last is None:
                    last=-1
                    if size:
                        with path.open('rb') as saved:
                            saved.seek(max(0,size-8192));tail=saved.read().splitlines()
                        if tail:last=json.loads(tail[-1])['client_seq']
                lines=[]
                for entry in entries:
                    seq=entry['client_seq']
                    if seq<=last:continue
                    lines.append(json.dumps({**entry,'server_received_at':time.time()},ensure_ascii=False)+'\n');last=seq
                if lines:
                    with path.open('a') as output:output.write(''.join(lines))
                diagnostic_seen[call_id]=last
                if len(diagnostic_seen)>1000:diagnostic_seen.pop(next(iter(diagnostic_seen)))
            return self.send(200,{'accepted':True,'last_client_seq':last})

        def proxy(self):
            try:
                if not Path('/srv/nas/.nas-online').is_file():return self.send(503,{'error':'存储不可用'})
                # A live HTTP process can retain weights after its NAS mount
                # disconnects. Only a fresh NAS heartbeat means usable storage.
                if self.path=='/api/status' or (self.command=='POST' and self.path=='/api/calls'):
                    try:
                        saved=json.loads((root.parent/'SERVICE.json').read_text())
                        fresh=time.time()-saved.get('heartbeat',0)<30
                    except (OSError,ValueError):fresh=False
                    if not fresh:
                        return self.send(200 if self.command=='GET' else 503,
                            {'ready':False,'phase':'storage_disconnected','model':'存储连接中断，暂时不能开始通话',
                             'error':'存储连接中断，请稍后重试'})
                if (root.parent/'STOP.json').exists():
                    if self.command=='GET' and self.path=='/api/status':
                        return self.send(200,{'ready':False,'phase':'updating','model':'step 3222 · 正在更新实时接口'})
                    return self.send(503,{'error':'实时接口正在更新，请稍后再连接'})
                parsed=urlparse(self.path);route=parsed.path.removeprefix('/api')
                parts=route.strip('/').split('/')
                if route!='/status' and route!='/calls':
                    import uuid
                    if len(parts)!=3 or parts[0]!='calls' or parts[2] not in ('audio','events','stop','diagnostics'):raise ValueError('route')
                    uuid.UUID(parts[1])
                    if parts[2]=='diagnostics':return self.diagnostics(parts[1])
                n=int(self.headers.get('Content-Length',0))
                if not 0<=n<=32000:raise ValueError('音频包过大')
                body=self.rfile.read(n) if self.command=='POST' else None
                if self.command=='POST' and route=='/calls':
                    spec=json.loads(body or b'{}')
                    if 'voice' in spec or 'language' in spec:
                        if spec.get('voice','female') not in ('male','female'):raise ValueError('无效声音预设')
                        if spec.get('language','auto') not in ('auto','zh','en'):raise ValueError('无效对话语言')
                        # Compatibility with the resident server's original
                        # single-field handler; no GPU process restart.
                        spec['variant']=spec.get('variant','trained')+':'+spec.pop('voice','female')+':'+spec.pop('language','auto')
                        body=json.dumps(spec).encode()
                headers={'Content-Type':self.headers.get('Content-Type','application/json')}
                if self.headers.get('X-Audio-Seq'):headers['X-Audio-Seq']=self.headers['X-Audio-Seq']
                req=urllib.request.Request(upstream+route+('?' + parsed.query if parsed.query else ''),
                                            data=body,headers=headers,method=self.command)
                with urllib.request.urlopen(req,timeout=20) as r:return self.send(r.status,r.read())
            except urllib.error.HTTPError as e:return self.send(e.code,e.read())
            except (urllib.error.URLError,TimeoutError,OSError):
                if self.command=='GET' and self.path=='/api/status':
                    try:
                        saved=json.loads((root.parent/'SERVICE.json').read_text())
                        if not saved.get('ready') and saved.get('phase')=='loading':
                            return self.send(200,{'ready':False,'phase':'loading','model':'step 3222 · 正在载入权重'})
                    except (OSError,ValueError):pass
                return self.send(503,{'error':'实时模型尚未连接，请稍后重试','ready':False})
            except ValueError as e:return self.send(400,{'error':str(e)})
        def do_GET(self):
            if not self.allowed():return self.send(403,{'error':'仅限局域网'})
            path=urlparse(self.path).path
            assets={'/':('realtime.html','text/html; charset=utf-8'),
                    '/lab/':('realtime-lab.html','text/html; charset=utf-8'),
                    '/voice.css':('voice.css','text/css; charset=utf-8'),
                    '/messages.css':('messages.css','text/css; charset=utf-8'),
                    '/messages.js':('messages.js','text/javascript; charset=utf-8'),
                    '/xiaocheng.png':('xiaocheng.png','image/png'),
                    '/realtime.js':('realtime.js','text/javascript; charset=utf-8'),
                    '/capture.worklet.js':('capture.worklet.js','text/javascript; charset=utf-8')}
            if path in assets:
                name,kind=assets[path];return self.send(200,(root/name).read_bytes(),kind)
            if path=='/household-root.crt':return self.send(200,Path('/run/cineduplex/root.crt').read_bytes(),'application/x-x509-ca-cert')
            if path.startswith('/api/'):return self.proxy()
            return self.send(404,{'error':'不存在'})
        def do_POST(self):
            if not self.allowed():return self.send(403,{'error':'仅限局域网'})
            origin=self.headers.get('Origin')
            if origin and urlparse(origin).netloc!=self.headers.get('Host'):return self.send(403,{'error':'来源不符'})
            if self.path.startswith('/api/'):return self.proxy()
            return self.send(404,{'error':'不存在'})
    class RealtimeHTTPServer(ThreadingHTTPServer):
        request_queue_size=64
        daemon_threads=True
    https=RealtimeHTTPServer((host,port),Handler)
    https.socket=context.wrap_socket(https.socket,server_side=True)
    local=RealtimeHTTPServer(('127.0.0.1',plain_port),Handler)
    threading.Thread(target=local.serve_forever,daemon=True).start();https.serve_forever()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cert',required=True);p.add_argument('--key',required=True)
    p.add_argument('--host',default='192.168.12.10');p.add_argument('--port',type=int,default=18088)
    p.add_argument('--plain-port',type=int,default=18089);p.add_argument('--upstream',default='http://127.0.0.1:18087')
    a=p.parse_args();serve(a.root,a.cert,a.key,a.host,a.port,a.plain_port,a.upstream)
