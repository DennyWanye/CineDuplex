"""Prepare opt-in normal-call native fixes; does not apply or publish them."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from cineduplex.train.nas_transport import NAS

BASE='2d9405253fa14781c83fd5db6868e403f8a9b7b8d10f3d9d167f3c2e5c7ecb6a'

def main():
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--prefix',required=True);p.add_argument('--version',type=int,required=True)
    p.add_argument('--native-linear-reference',action='store_true',help='Diagnostic: use the existing native BF16 prefill linear arithmetic for decode too')
    p.add_argument('--native-acoustic-reference',action='store_true',help='Keep native BF16 speech and merge projections; leave main/control acceleration unchanged')
    p.add_argument('--response-scoped-speech',action='store_true',help='Apply existing acoustic n-gram rule only within the current answer')
    p.add_argument('--full-native-reference',action='store_true',help='Diagnostic: restore full native BF16 projections and RMS in every branch')
    p.add_argument('--native-acoustic-rms-reference',action='store_true',help='Keep native RMS only for speech/merge; restore original qualified RMS elsewhere')
    p.add_argument('--drain-idle-grace',action='store_true',help='Keep the 15s idle bound but restart it after actual input processing')
    p.add_argument('--native-cuda-graphs',action='store_true',help='Capture only unchanged stateless native MLP/RMS arithmetic with bitwise qualification')
    p.add_argument('--natural-acoustic-repetition',action='store_true',help='Causal comparison: audit but do not apply text-style n-gram bans to acoustic codes; retain EOS, length bounds and all safety')
    p.add_argument('--aligned-listening',action='store_true',help='Match listening merge inputs to retained PAD channels and the official vLLM listening contract')
    p.add_argument('--native-row-partition',action='store_true',help='Qualified output-row native BF16 decode with full-matrix fallback')
    p.add_argument('--direct-answer-policy',action='store_true',help='Native initial instruction: answer direct planning requests without repeated optional clarifications')
    p.add_argument('--chronological-review',action='store_true',help='Sort cloud ASR slices by actual audio time; retain every original risk decision')
    p.add_argument('--audit-native-head',action='store_true',help='Read-only full vs restricted BF16 head comparison on actual hidden states')
    p.add_argument('--full-head-reference',action='store_true',help='Causal full native BF16 speech head comparison, with unchanged vocabulary and sampling')
    p.add_argument('--training-embedding-order',action='store_true',help='Causal input-addition order from training and official vLLM; no channel changes')
    p.add_argument('--audit-native-history',action='store_true',help='Observe cached token histories and completed response boundaries without changing generation')
    p.add_argument('--continuous-received-context',action='store_true',help='Bounded diagnostic: refresh received audio features and dependent KV during speaking too; no future audio')
    p.add_argument('--upstream-reference-prefix',action='store_true',help='Reference-only control: exact upstream Dataset/native system prefix; all release safety policies retained')
    p.add_argument('--skip-terminal-word-projection',action='store_true',help='Qualify and skip the full word head after native EOT when only TTS padding is legal')
    a=p.parse_args();n=NAS(json.loads(Path(a.config).read_text())['nas']);pre=a.prefix.rstrip('/')+'/'
    data=n.read(pre+'tp-runtime-v73.py',timeout=15)
    if hashlib.sha256(data).hexdigest()!=BASE:raise ValueError('normal baseline source changed')
    pins={}
    if a.native_cuda_graphs and not a.full_native_reference:
        raise ValueError('native graphs require complete native arithmetic')
    if a.natural_acoustic_repetition and not a.response_scoped_speech:
        raise ValueError('acoustic repetition comparison requires the existing scope audit')
    if a.native_row_partition and (not a.full_native_reference or a.native_cuda_graphs):
        raise ValueError('row partition requires full native arithmetic without graphs')
    if a.direct_answer_policy and not a.drain_idle_grace:
        raise ValueError('direct-answer policy requires the pinned normal call candidate')
    module_names=['native_audio_context','native_output_contract','native_quality_session']
    if a.native_cuda_graphs:module_names.append('native_cuda_graphs')
    if a.aligned_listening:module_names.append('native_listening_contract')
    if a.native_row_partition:module_names.append('native_row_parallel')
    if a.audit_native_head or a.full_head_reference:module_names.append('native_head_audit')
    if a.training_embedding_order:module_names.append('native_embedding_order')
    if a.audit_native_history:module_names.append('native_history_audit')
    if a.skip_terminal_word_projection:
        if a.audit_native_head or a.full_head_reference:
            raise ValueError('terminal shortcut cannot overlap a head diagnostic')
        module_names.append('native_terminal_projection')
    for name in module_names:
        source=(Path(__file__).resolve().parents[1]/'src/cineduplex/train'/f'{name}.py').read_bytes();ast.parse(source)
        pins[name]=n.put(pre+name.replace('_','-')+f'-v{a.version}.py',source)
    needle='        policy.install(native_prefix=False,speech_detector=model._cineduplex_vad_detector,turn_control=self.turn_control)\n'
    source=data.decode()
    call_pin=None
    if a.drain_idle_grace:
        call_source=n.read(pre+'call-runtime-v53.py',timeout=15)
        if hashlib.sha256(call_source).hexdigest()!='472812bc06a0aa59e7d485ef0a9f840e4b4f9ff0c6478b1f9e4ddc02fd5c0214':
            raise ValueError('original call source identity')
        call_source=call_source.decode()
        edits={
            'self.last_input=self.started;':'self.last_input=self.started; self.last_processed_at=self.started;',
            'time.time()-self.last_input>15 and self.input.empty()':'time.time()-max(self.last_input,self.last_processed_at)>15 and self.input.empty()',
            'self.processed+=len(audio)':'self.processed+=len(audio); self.last_processed_at=time.time()',
        }
        for old,new in edits.items():
            if call_source.count(old)!=1:raise ValueError('idle progress hook identity: '+old)
            call_source=call_source.replace(old,new)
        if a.direct_answer_policy:
            old=("'when asked to do something else. Give the direct answer in ONE short sentence, '\n"
                 "        'normally at most 30 Chinese characters or 20 English words. Expand only when '\n"
                 "        'explicitly asked. Do not repeat the question, add a follow-up offer, or narrate '\n"
                 "        'your voice, acting, reasoning or instructions. If unclear, ask one short question. '")
            new=("'when asked to do something else. Keep routine answers concise. For a requested '\n"
                 "        'plan or explanation, give a short but complete answer. Use details already '\n"
                 "        'provided. If the user asks you to answer directly, answer now without another '\n"
                 "        'clarifying question. For optional missing preferences, state a reasonable '\n"
                 "        'assumption briefly and proceed; do not invent factual details. Ask only about '\n"
                 "        'essential missing information, and never repeat a clarification already asked. '\n"
                 "        'Do not repeat the question, add a follow-up offer, or narrate your voice, '\n"
                 "        'acting, reasoning or instructions. '")
            if call_source.count(old)!=1:raise ValueError('native instruction policy identity')
            call_source=call_source.replace(old,new)
        ast.parse(call_source)
        if a.chronological_review:
            # Distinguish this Call binding from older identical-code hot
            # updates whose Application.start wrappers retain their bundle ID.
            call_source=f'# Chronological guardrail binding {a.version}\n'+call_source
        call_pin=n.put(pre+f'call-runtime-v{a.version}.py',call_source.encode())
        source=source.replace("CALL_RUNTIME_NAME='call-runtime-v53.py'",'CALL_RUNTIME_NAME='+repr(Path(call_pin['path']).name)).replace(
            "CALL_RUNTIME_SHA='472812bc06a0aa59e7d485ef0a9f840e4b4f9ff0c6478b1f9e4ddc02fd5c0214'",'CALL_RUNTIME_SHA='+repr(call_pin['sha256']))
    if source.count(needle)!=1:raise ValueError('native policy hook identity')
    hook=f'''        quality_pins={pins!r}
        quality_modules={{}}
        for name,pin in quality_pins.items():
            path=self.root.parents[1]/pin['path']
            payload=path.read_bytes()
            if hashlib.sha256(payload).hexdigest()!=pin['sha256']:raise ValueError('native quality source identity')
            mod=types.ModuleType('private_'+name)
            exec(compile(payload,str(path),'exec'),mod.__dict__)
            quality_modules[name]=mod
        from lychee_fd.runtime.hf_v9_realtime import HFRealtimeV9GenerationFramework
        factory=HFRealtimeV9GenerationFramework.create_incremental_stream_session
        if {a.response_scoped_speech!r}:
            native_factory=factory
            def factory(fw,*args,**kwargs):
                fw._cineduplex_response_speech_scope=True
                fw._cineduplex_repetition_receipts=[]
                fw._cineduplex_natural_acoustic_repetition={a.natural_acoustic_repetition!r}
                return native_factory(fw,*args,**kwargs)
        def quality_audit(receipt):
            from cineduplex.train.realtime_service import atomic
            path=self.root/'calls'/self.profile_id/('native-quality-rank'+str(self.rank)+'.json')
            atomic(path,receipt)
        HFRealtimeV9GenerationFramework.create_incremental_stream_session=quality_modules['native_quality_session'].wrap_factory(
            factory,quality_modules['native_audio_context'],quality_modules['native_output_contract'],model._cineduplex_vad_detector,audit=quality_audit,
            listening=quality_modules.get('native_listening_contract'),head_audit=quality_modules.get('native_head_audit'),full_head_reference={a.full_head_reference!r},embedding_order=quality_modules.get('native_embedding_order'),history_audit=quality_modules.get('native_history_audit'),continuous_received={a.continuous_received_context!r},upstream_reference_prefix={a.upstream_reference_prefix!r},terminal_projection=quality_modules.get('native_terminal_projection'))
'''
    source=source.replace(needle,needle+hook)
    if a.native_acoustic_rms_reference:
        if not a.native_acoustic_reference or a.full_native_reference:
            raise ValueError('acoustic RMS comparison requires acoustic projections')
        marker='        candidates=[]\n'
        if source.count(marker)!=1:raise ValueError('RMS restoration hook identity')
        rms='''        # Restore the exact admitted fused RMS recipe before this comparison.
        rms_path=self.root/'bf16-kernels-v2.py'
        if hashlib.sha256(rms_path.read_bytes()).hexdigest()!='3c086063d8fac9f45afafda3cc33b59b1ce525a64530f2dab220d1bdcfb123f6':
            raise ValueError('admitted RMS kernel identity')
        rms_spec=importlib.util.spec_from_file_location('cineduplex_bf16_restore',rms_path)
        rms_module=importlib.util.module_from_spec(rms_spec);sys.modules[rms_spec.name]=rms_module;rms_spec.loader.exec_module(rms_module)
        for name,layer in model.named_modules():
            if layer.__class__.__name__!='Qwen2RMSNorm' or not hasattr(layer,'_realtime_unfused_forward'):continue
            original=layer._realtime_unfused_forward
            def admitted_rms(m,x,base=original,kernels=rms_module):
                if x.is_cuda and x.dtype==torch.bfloat16 and x.numel()==m.weight.numel() and x.is_contiguous():
                    return kernels.rms(x,m.weight,m.variance_epsilon)
                return base(x)
            layer.forward=types.MethodType(admitted_rms,layer)
            if name.startswith(('stoken_model.','merge_model.')):
                layer._cineduplex_quality_previous_rms=layer.forward
                layer.forward=original
'''
        source=source.replace(marker,rms+marker)
    if a.native_linear_reference:
        predicate='if x.numel()>x.shape[-1]:'
        if source.count(predicate)!=2:raise ValueError('native BF16 linear reference hook identity')
        source=source.replace(predicate,'if True:  # diagnostic: identical native BF16 GEMM for decode and prefill')
    if a.native_acoustic_reference:
        if a.native_linear_reference:raise ValueError('select one linear comparison')
        local=(Path(__file__).resolve().parents[1]/'src/cineduplex/train/tensor_parallel.py').read_text()
        begin=local.index('        # Keep native full BF16 speech/merge arithmetic')
        end=local.index('        install_key=',begin)
        marker='        install_key=tuple((name,id(weight),id(bias),axis) for name,_,weight,bias,axis in candidates)'
        if source.count(marker)!=1:raise ValueError('acoustic reference hook identity')
        source=source.replace(marker,local[begin:end]+marker)
        source=source.replace("'all-branches-local'","'main-control-local/speech-merge-full-bf16'")
    if a.full_native_reference:
        if a.native_linear_reference or a.native_acoustic_reference:raise ValueError('select one computation reference')
        marker='        install_key=tuple((name,id(weight),id(bias),axis) for name,_,weight,bias,axis in candidates)'
        if source.count(marker)!=1:raise ValueError('full native reference hook identity')
        native='''        # Causal reference: same full module arithmetic as isolated native model.
        for _,layer,weight,bias,_ in candidates:
            for attr in ('_cineduplex_tp_operator','_cineduplex_tp_group','_cineduplex_tp_index'):
                if hasattr(layer,attr):delattr(layer,attr)
            layer.forward=types.MethodType(lambda m,x,w=weight,b=bias:torch.nn.functional.linear(x,w,b),layer)
        for name,layer in model.named_modules():
            if name.startswith(('model.','stoken_model.','control_model.','merge_model.')) and layer.__class__.__name__=='Qwen2RMSNorm':
                original=getattr(layer,'_realtime_unfused_forward',None)
                if original is not None:
                    layer._cineduplex_quality_previous_rms=layer.forward
                    layer.forward=original
        candidates=[]
'''
        source=source.replace(marker,native+marker)
        source=source.replace("'all-branches-local'","'all-branches-native-full-bf16'")
    if a.full_native_reference or a.native_acoustic_reference:
        # Call53 reads this class field when it reports the live layout.
        layout='native-full-bf16-replicated' if a.full_native_reference else 'main-control-tp4/speech-merge-native-bf16'
        source=source.replace('class Coordinator:\n','class Coordinator:\n    branch_layout='+repr(layout)+'\n',1)
    if a.native_cuda_graphs:
        reuse="        if getattr(self,'install_key',None)==install_key:\n"
        if source.count(reuse)!=1:raise ValueError('native graph per-call ownership hook identity')
        # Graph buffers belong to the call and are retired on close. Rebuild
        # them even when the immutable underlying weight objects are reused.
        source=source.replace(reuse,"        if False:  # native graphs are owned by this call\n")
        marker='        self.install_key=install_key;self.qualification=record\n'
        if source.count(marker)!=1:raise ValueError('native graph lifecycle hook identity')
        graph_code='''        graph_module=quality_modules['native_cuda_graphs']
        graph_error=None;graph_receipt=None
        try:graph_receipt=graph_module.install(model)
        except Exception as exc:graph_error=type(exc).__name__+': '+str(exc)[:240]
        graph_states=[None]*self.world
        dist.all_gather_object(graph_states,{'rank':self.rank,'error':graph_error,'receipt':graph_receipt},group=self.control)
        if any(item['error'] for item in graph_states):
            graph_module.close(model)
            raise ValueError('native graph qualification failed: '+json.dumps(graph_states))
        model._cineduplex_native_graph_close=graph_module.close
        record['native_stateless_graphs']=graph_states
'''
        source=source.replace(marker,graph_code+marker)
    if a.native_row_partition:
        source=source.replace("branch_layout='native-full-bf16-replicated'",
                              "branch_layout='native-bf16-output-row-tp4/down-full'")
        reuse="        if getattr(self,'install_key',None)==install_key:\n"
        if source.count(reuse)!=1:raise ValueError('native projection ownership hook')
        source=source.replace(reuse,"        if False:  # row partitions belong to this call\n")
        marker='        self.install_key=install_key;self.qualification=record\n'
        if source.count(marker)!=1:raise ValueError('native projection install hook')
        row_code='''        row_module=quality_modules['native_row_parallel']
        record['native_output_row_partition']=row_module.install(model,TP_GROUP)
        model._cineduplex_native_projection_close=row_module.close
'''
        source=source.replace(marker,row_code+marker)
    guard_pin=None
    if a.chronological_review:
        from prepare_chronological_guardrail import prepare
        source,guard_pin=prepare(n,pre,a.version,source)
    ast.parse(source)
    tp=n.put(pre+f'tp-runtime-v{a.version}.py',source.encode())
    result={'tp':tp,'modules':pins,'baseline':BASE,'call_runtime':'original Call53','weights_changed':False,
            'precision':'bf16','quantization':'none','safety_changed':False,'quality_accepted':False,'applied':False,
            'native_linear_reference':a.native_linear_reference,'native_acoustic_reference':a.native_acoustic_reference}
    result['response_scoped_speech']=a.response_scoped_speech
    result['full_native_reference']=a.full_native_reference
    result['native_acoustic_rms_reference']=a.native_acoustic_rms_reference
    result['idle_grace_after_processing']=a.drain_idle_grace
    result['native_cuda_graphs']=a.native_cuda_graphs
    result['natural_acoustic_repetition']=a.natural_acoustic_repetition
    result['aligned_listening']=a.aligned_listening
    result['audit_native_head']=a.audit_native_head
    result['full_head_reference']=a.full_head_reference
    result['training_embedding_order']=a.training_embedding_order
    result['audit_native_history']=a.audit_native_history
    result['continuous_received_context']=a.continuous_received_context
    result['upstream_reference_prefix']=a.upstream_reference_prefix
    result['generation_instruction_changed']=a.upstream_reference_prefix
    result['skip_terminal_word_projection']=a.skip_terminal_word_projection
    result['native_row_partition']=a.native_row_partition
    result['direct_answer_policy']=a.direct_answer_policy
    if guard_pin:
        result['guardrail_bundle']=guard_pin
        result['review_change']='chronological ASR assembly only; risk rules unchanged'
    if call_pin:
        result['call']=call_pin
        result['call_base_sha256']='472812bc06a0aa59e7d485ef0a9f840e4b4f9ff0c6478b1f9e4ddc02fd5c0214'
        result['call_runtime']='Call53 with bounded idle progress fix'
        if a.direct_answer_policy:
            result['call_runtime']='Call53 with idle fix and direct-answer instruction'
    print(n.put(pre+f'NATIVE-QUALITY-{a.version}-PINS.json',json.dumps(result,indent=2).encode()))
    print(json.dumps(result))

if __name__=='__main__':main()
