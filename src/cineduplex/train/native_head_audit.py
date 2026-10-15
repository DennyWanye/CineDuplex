"""Bounded comparison of restricted and full native BF16 heads.

Observation mode returns the unmodified resident result. Optional reference
mode returns full-head speech logits with the same allowed vocabulary.
No RNG or transcript input. Owned by one recorded call.
"""
import types


def install(model, full_reference=False):
    import torch
    import torch.nn.functional as F
    head = model.lm_head
    if head.__class__.__name__ != 'RealtimeProjection':
        raise ValueError('restricted native head required')
    if getattr(head, '_native_head_audit', None) is not None:
        raise ValueError('head audit already owned')
    if head.weight.dtype != torch.bfloat16:
        raise ValueError('BF16 head required')
    previous = head.forward
    budget = {'speech': 320, 'control': 96}
    receipt = {'observation_only': not full_reference, 'returned_resident_result': not full_reference,
               'full_native_speech_reference': bool(full_reference),
               'budget': budget, 'branches': {}}
    for name in budget:
        index = getattr(head, name+'_index')
        copied = getattr(head, name+'_weight')
        same = torch.equal(head.weight.detach().index_select(0, index), copied)
        receipt['branches'][name] = {
            'copied_weights_equal': same, 'checked': 0, 'different': 0,
            'argmax_different': 0, 'max_abs': 0., 'examples': []}

    def observed(current, x):
        phase = current.phase
        active = not getattr(current, 'fixed_listening', False)
        actual = previous(x)
        name = 'control' if phase == 'control' else 'speech'
        if phase not in ('control', 'speech') or not active:
            return actual
        stats = receipt['branches'][name]
        inspect = stats['checked'] < budget[name]
        replace = full_reference and name == 'speech'
        if not inspect and not replace:
            return actual
        index = getattr(current, name+'_index')
        with torch.no_grad():
            full = F.linear(x[:, -1:], current.weight)
            expected = full.index_select(-1, index)
            result = actual
            if replace:
                # Preserve the exact permitted vocabulary. Only projection
                # arithmetic changes, never a processor or sampling rule.
                result = actual.clone()
                result[:, :, index] = expected
            if not inspect:
                return result
            selected = actual.index_select(-1, index)
            delta = (selected.float()-expected.float()).abs()
            maximum = float(delta.max().item())
            left = int(selected.argmax(-1).item())
            right = int(expected.argmax(-1).item())
            different = not torch.equal(selected, expected)
            stats['checked'] += 1
            stats['different'] += int(different)
            stats['argmax_different'] += int(left != right)
            stats['max_abs'] = max(stats['max_abs'], maximum)
            if different and len(stats['examples']) < 12:
                stats['examples'].append({
                    'check': stats['checked'], 'max_abs': maximum,
                    'resident_argmax': int(index[left].item()),
                    'full_argmax': int(index[right].item()),
                    'changed_values': int((selected != expected).sum().item())})
        return result

    head.forward = types.MethodType(observed, head)
    head._native_head_audit = (previous, receipt)
    return receipt


def close(model):
    head = model.lm_head
    state = getattr(head, '_native_head_audit', None)
    if state is None:
        return None
    previous, receipt = state
    head.forward = previous
    del head._native_head_audit
    return receipt
