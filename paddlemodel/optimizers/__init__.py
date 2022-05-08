import importlib
import sys

backend_name = 'paddle'

def _load_backend(mod_name):
    mod = importlib.import_module(".%s" % mod_name, __name__)
    thismod = sys.modules[__name__]
    for api, obj in mod.__dict__.items():
        setattr(thismod, api, obj)

_load_backend(backend_name.replace(".", "_"))
