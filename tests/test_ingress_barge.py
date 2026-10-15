"""CPU contracts for ingress cancellation and synchronized native turn control."""
import ast
from pathlib import Path
import queue
import threading
import time
import types
import sys
import unittest
from unittest.mock import patch
import numpy as np
from cineduplex.train.native_turn_policy import install

class IngressTests(unittest.TestCase):
    def test_feed_cancels_without_consuming_backlog_and_preserves_pcm_and_review(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'src/cineduplex/train/realtime_service.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Call')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='feed')
        scope={'np':np,'time':time};exec(compile(ast.Module(body=[fn],type_ignores=[]),'feed','exec'),scope)
        saved=[];reviewed=[];events=[]
        call=types.SimpleNamespace(input_lock=threading.Lock(),closed=threading.Event(),last_seq=-1,
            input=queue.Queue(maxsize=75),archive=types.SimpleNamespace(write=lambda k,b:saved.append(b)),
            guardrail=types.SimpleNamespace(input_audio=lambda seq,b:reviewed.append((seq,b))),
            ingress_speech=types.SimpleNamespace(feed=lambda a:True,evidence=types.SimpleNamespace(active=False)),ingress_revision=0,epoch=7,
            emit=events.append,received=32000,processed=0)
        pcm=np.ones(6400,dtype='<i2').tobytes()
        scope['feed'](call,0,pcm)
        self.assertEqual(call.epoch,8);self.assertEqual(call.processed,0)
        self.assertEqual(call.input.get()[1],pcm);self.assertEqual(saved,[pcm]);self.assertEqual(reviewed,[(0,pcm)])
        self.assertEqual(events[0]['reason'],'ingress_confirmed_speech')
        scope['feed'](call,0,pcm)
        self.assertEqual(call.epoch,8);self.assertEqual(len(saved),1)

    def test_all_ranks_close_old_tail_and_catch_up_before_answering(self):
        class Session:
            listening_state='s';next_chunk_pos=100
            generated_input_ids=np.array([[1,2,3]])
            generated_stoken_ids=np.array([[4,5,6]])
            generated_control_ids=np.array([[7,8,9]])
            def _set_last_ss_pos(self,x):self.last_ss_pos=x
            def advance_stream(self,*a,**kw):yield {'type':'observed','state':self.listening_state}
        class Processor:
            ss_token_id=1;kl_token_id=0;bc_token_id=2
            def __call__(self,ids,scores):return scores
        class Framework:
            model=types.SimpleNamespace();tts_end_id=900;eos_token_id=800;tts_pad_id=801;sl_token_id=10
            def init_speaking_processor(self, factor):return [], []
            def create_incremental_stream_session(self,**kw):return Session()
            def _apply_s2l_tail_fallback(self,t,s):return self._set_last_token_safe(t,800),self._set_last_token_safe(s,900)
            def _set_last_token_safe(self,t,v):r=t.copy();r[0,-1]=v;return r
        evidence=types.SimpleNamespace(active=False,utterance=2,probability=0.,quiet_frames=50,
            can_reply=lambda n:True,reply_overdue=lambda n:False)
        activity=types.SimpleNamespace(evidence=evidence,feed=lambda a:False)
        mod=types.ModuleType('lychee_fd.runtime.hf_v9_realtime')
        mod.HFRealtimeV9GenerationFramework=Framework;mod.ListeningControlLogitsProcessor=Processor
        mod.SpeakingControlLogitsProcessor=type('Speaking',(),{'__call__':lambda *a:None})
        pkgs={n:types.ModuleType(n) for n in ('lychee_fd','lychee_fd.runtime')}
        control={'revision':1,'caught_up':False,'quiet':True}
        with patch.dict(sys.modules,{**pkgs,mod.__name__:mod}),patch('cineduplex.train.speech_activity.StreamingSpeech',return_value=activity):
            install(speech_detector=object(),turn_control=control)
            session=Framework().create_incremental_stream_session()
            events=list(session.advance_stream([0.]))
            self.assertEqual(events[0]['reason'],'ingress_barge_in');self.assertEqual(session.listening_state,'l')
            self.assertEqual(session.generated_input_ids.tolist(),[[1,2,800]])
            self.assertEqual(session.generated_stoken_ids.tolist(),[[4,5,900]])
            self.assertTrue(np.isneginf(Processor()(None,np.zeros((1,3)))[0,1]))
            control['caught_up']=True
            self.assertEqual(Processor()(None,np.zeros((1,3)))[0,1],0)
            control['quiet']=False
            self.assertTrue(np.isneginf(Processor()(None,np.zeros((1,3)))[0,1]))
            self.assertFalse(any(e.get('reason')=='ingress_barge_in' for e in session.advance_stream([0.])))
if __name__=='__main__':unittest.main()
