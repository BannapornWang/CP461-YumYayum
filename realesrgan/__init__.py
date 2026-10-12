# flake8: noqa
import sys
from torchvision.transforms import functional as torchvision_functional
sys.modules.setdefault('torchvision.transforms.functional_tensor', torchvision_functional)

try:
    from .archs import *
except Exception:
    pass
try:
    from .data import *
except Exception:
    pass
try:
    from .models import *
except Exception:
    pass
try:
    from .utils import *
except Exception:
    pass
try:
    from .version import *
except Exception:
    pass
