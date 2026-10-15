"""Bind native quality adapters to one session, with explicit cleanup."""
import types


def wrap_factory(original_create, context, output, detector, audit=None, listening=None, head_audit=None, full_head_reference=False, embedding_order=None, history_audit=None, continuous_received=False, upstream_reference_prefix=False, terminal_projection=None):
    def create(fw, *args, **kwargs):
        if getattr(fw, '_cineduplex_quality_session', None) is not None:
            raise RuntimeError('previous native quality session has not closed')
        previous_speaking = fw.init_speaking_processor
        previous_contract = getattr(fw, '_cineduplex_output_contract', False)
        if previous_contract:
            raise RuntimeError('unexpected persistent output adapter')
        session = None
        model = getattr(fw.model, '_model', fw.model)
        previous_history = getattr(model, '_cineduplex_merge_conditioning', None)
        receipt = {'previous_merge_positions':int(previous_history.shape[1]) if previous_history is not None else 0,
                   'new_call_merge_reset':True, 'events':[]}
        # Merge conditioning is conversation state, just like the four KV
        # caches. A new call must not inherit sampled words from another call.
        model._cineduplex_merge_conditioning = None

        def restore():
            if terminal_projection is not None:
                receipt['terminal_projection'] = terminal_projection.close(model)
            if embedding_order is not None:
                value = embedding_order.close(model)
                if value is not None:
                    receipt['embedding_order'] = value
            if head_audit is not None:
                receipt['head_projection'] = head_audit.close(model)
            graph_close = getattr(model, '_cineduplex_native_graph_close', None)
            projection_close = getattr(model, '_cineduplex_native_projection_close', None)
            try:
                if graph_close is not None:
                    receipt['native_graphs'] = graph_close(model)
                if projection_close is not None:
                    receipt['native_rows'] = projection_close(model)
            finally:
                if graph_close is not None:
                    del model._cineduplex_native_graph_close
                if projection_close is not None:
                    del model._cineduplex_native_projection_close
                for layer in model.modules() if hasattr(model, 'modules') else ():
                    if hasattr(layer, '_cineduplex_quality_previous_rms'):
                        layer.forward = layer._cineduplex_quality_previous_rms
                        del layer._cineduplex_quality_previous_rms
                fw.init_speaking_processor = previous_speaking
                fw._cineduplex_output_contract = False
                fw._cineduplex_quality_session = None
                model._cineduplex_merge_conditioning = None

        try:
            output_receipt = output.install(fw)
            if upstream_reference_prefix:
                # Reference-only causal control: use the exact Dataset / native
                # upstream system prefix, then restore the process global.
                # Input/output safety processors still own release decisions.
                from lychee_fd.runtime import vllm_generation
                previous_prefix = vllm_generation.SYSTEM_MESSAGE_PREFIX
                try:
                    vllm_generation.SYSTEM_MESSAGE_PREFIX = '<|BOT|>system\nYou are a helpful assistant.<|EOT|>'
                    session = original_create(fw, *args, **kwargs)
                    receipt['system_prefix_reference'] = 'upstream-dataset-native-default'
                finally:
                    vllm_generation.SYSTEM_MESSAGE_PREFIX = previous_prefix
            else:
                session = original_create(fw, *args, **kwargs)
            original_close = session.close
            if embedding_order is not None:
                receipt['embedding_order'] = embedding_order.install(model)
            if listening is not None:
                receipt['listening_contract'] = listening.install(session)
            if head_audit is not None:
                receipt['head_projection'] = head_audit.install(model, full_reference=full_head_reference)

            def close(current):
                try:
                    return original_close()
                finally:
                    restore()
                    if audit is not None:
                        receipt['audio_context'] = list(getattr(current, '_cineduplex_audio_context_receipts', []))
                        receipt['repetition_scope'] = list(getattr(fw, '_cineduplex_repetition_receipts', []))
                        audit(receipt)

            session.close = types.MethodType(close, session)
            fw._cineduplex_quality_session = session
            context_receipt = (context.install(session, detector, continuous_received=True)
                               if continuous_received else context.install(session, detector))
            if history_audit is not None:
                receipt['history_alignment'] = history_audit.install(session)
            if terminal_projection is not None:
                receipt['terminal_projection'] = terminal_projection.install(session)
            if audit is not None:
                advance = session.advance_stream
                def observed(current, *args, **kwargs):
                    for event in advance(*args, **kwargs):
                        if event.get('type') in ('speaking_token','state_change','speaking_done'):
                            if len(receipt['events']) >= 10000:
                                raise RuntimeError('native quality trace budget exceeded')
                            receipt['events'].append(dict(event))
                        yield event
                session.advance_stream = types.MethodType(observed, session)
            session._cineduplex_quality_receipt = {
                'input':context_receipt, 'output':output_receipt,
                'device_access':False, 'text_conditioning':False}
            return session
        except Exception:
            if session is not None:
                try:
                    session.close()
                finally:
                    restore()
            else:
                restore()
            raise
    return create
