"""Read the fixed codec YAML without HyperPyYAML/ruamel version coupling."""
import hashlib
import importlib
import yaml


def flow_from_config(data):
    if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!='8638dca97881751e50c132498acbfa41ecf74d6f':
        raise ValueError('codec YAML differs from pinned source')
    allowed={
        'cosyvoice2.flow.flow.CausalMaskedDiffWithXvec',
        'cosyvoice2.transformer.upsample_encoder_v2.UpsampleConformerEncoderV2',
        'cosyvoice2.flow.flow_matching.CausalConditionalCFM',
        'cosyvoice2.flow.decoder_dit.DiT',
    }
    class Loader(yaml.SafeLoader):pass
    def construct(loader,name,node):
        if name not in allowed:raise ValueError('unapproved codec constructor')
        module,cls=name.rsplit('.',1)
        return getattr(importlib.import_module(module),cls)(**loader.construct_mapping(node,deep=True))
    Loader.add_multi_constructor('!new:',construct)
    return yaml.load(data.decode(),Loader=Loader)['flow']
