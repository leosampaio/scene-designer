import os
import pkgutil
import importlib

from core.data import BaseDataLoader, BaseDataLoaderV2

models_by_name = {}
pkg_dir = os.path.dirname(__file__)
for (module_loader, name, ispkg) in pkgutil.iter_modules([pkg_dir]):
    importlib.import_module('.' + name, __package__)

all_subclasses = BaseDataLoader.__subclasses__() + [s for ss in [s.__subclasses__() for s in BaseDataLoader.__subclasses__()] for s in ss]
models_by_name = {cls.name: cls for cls in all_subclasses if hasattr(cls, 'name')}

models_by_name_v2 = {}
pkg_dir = os.path.dirname(__file__)
for (module_loader, name, ispkg) in pkgutil.iter_modules([pkg_dir]):
    importlib.import_module('.' + name, __package__)

all_subclasses_v2 = BaseDataLoaderV2.__subclasses__() + [s for ss in [s.__subclasses__() for s in BaseDataLoaderV2.__subclasses__()] for s in ss]
models_by_name_v2 = {cls.name: cls for cls in all_subclasses_v2 if hasattr(cls, 'name')}


def get_dataloader_by_name(dataloader_name):
    try:
        return models_by_name_v2[dataloader_name]
    except KeyError:
        return models_by_name[dataloader_name]