import os
import json
import inspect
import logging
import configparser
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CompoConfigurable(ABC):
    @classmethod
    def from_composer_config(cls, config):
        signature = inspect.signature(cls.from_composer)
        params = signature.parameters.values()
        all_param_names = set(param.name for param in params)
        explicit_kwargs = {
            param.name: getattr(config, param.name) for param in params
            if param.kind != inspect.Parameter.VAR_KEYWORD
        }
        explicit_attrs = set(explicit_kwargs.keys())
        yetpass_param_names = all_param_names - explicit_attrs
        if len(yetpass_param_names) == 0:
            kwargs = {}
        elif len(yetpass_param_names) == 1:
            leftover_attrs = CompoConfig.get_dict_attrs(config.dict()) - explicit_attrs
            kwargs = {
                attr: getattr(config, attr) for attr in leftover_attrs
            }
        else:
            raise ValueError
        kwargs.update(explicit_kwargs)
        return cls.from_composer(**kwargs)

    @classmethod
    @abstractmethod
    def from_composer(cls, **kwargs): ...


class CompoConfig(CompoConfigurable):
    _config_version = 'v1'
    _config_version_key = 'config.version'
    _save_json_file_name = 'compo.json'

    def __init__(self, config_dict: dict):
        self._configs = self._flatten_recur_keys(config_dict.copy())
        self.accessed_keys = set()
        self._configs[self._config_version_key] = self._config_version

    @staticmethod
    def _flatten_recur_keys(d):
        new_dict = dict()
        def recur(sub_dict, prefix=""):
            for key, value in sub_dict.items():
                new_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    recur(value, new_key)
                else:
                    new_dict[new_key] = value
            return new_dict
        return recur(d)

    def __getitem__(self, key):
        return self._configs[key]

    def __setitem__(self, key, value):
        self._configs[key] = value

    @classmethod
    def from_composer(cls, path:str, **kwargs):
        ini_config = configparser.ConfigParser(allow_no_value=True)
        assert os.path.exists(path), path
        ini_config.read(path)

        # parse INI "source-of-default" config file
        config_dict = dict()
        for section in ini_config.sections():
            items = ini_config[section]
            for item_key in items:
                key = f'{section}.{item_key}'
                val = eval(items.get(item_key))
                config_dict[key] = val

        if v := kwargs.get('save_json_file_name', None):
            cls._save_json_file_name = v
        return cls(config_dict)

    def composed(self, **inject_arguments):
        used_inject_keys = set()
        base_configs = self.dict()

        # pass#1: inject compositional keys
        base_compo_keys = list(filter(lambda k: '@' in k, base_configs.keys()))
        for base_compo_key in base_compo_keys:
            base_val = base_configs[base_compo_key]
            for inject_compo_key in filter(lambda k: '@' in k, inject_arguments.keys()):
                if inject_compo_key in base_compo_key:
                    base_key = base_compo_key.replace(inject_compo_key, '')
                    base_configs[base_key] = base_val
                    used_inject_keys.add(inject_compo_key)

        # pass#2: inject exact keys
        base_exact_configs = {k: v for k, v in base_configs.items() if '@' not in k}
        for base_exact_key in base_exact_configs.keys():
            if base_exact_key in inject_arguments:
                try:
                    injecting_val = eval(inject_arguments[base_exact_key]) # eager eval
                except:
                    injecting_val = inject_arguments[base_exact_key]
                base_exact_configs[base_exact_key] = injecting_val
                used_inject_keys.add(base_exact_key)

        unused_inject_keys = set(inject_arguments.keys()) - used_inject_keys
        assert len(unused_inject_keys) == 0, unused_inject_keys

        return CompoConfig(base_exact_configs)

    def __getattr__(self, key):
        if key in self._configs:
            return self._configs[key]
        else:
            return _CompoConfigProxy(self, key)

    def dict(self):
        return self._configs.copy()

    def pretty_json(self):
        return json.dumps(self._configs, indent=2, sort_keys=True)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.pretty_json()})"

    @staticmethod
    def resolve(x):
        if isinstance(x, _CompoConfigProxy) or isinstance(x, CompoConfig):
            if d := x.dict():
                return d
            else:
                return None
        else:
            return x

    @staticmethod
    def get_dict_attrs(d: dict):
        return set(k.split('.')[0] for k in d.keys())

    def save_json_file(self, directory, fname=None):
        fname = fname or self._save_json_file_name
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, fname), 'w') as fh:
            json.dump(self._configs, fh, indent=2, sort_keys=True)
            fh.write('\n')

    def load_json_file(self, path, warn_change_key_prefix='', ignore_keys=[]):
        with open(path, 'r') as fh:
            loaded_configs = json.load(fh)
        unexpected_changed_keys = []
        for key, load_val in loaded_configs.items():
            if key not in self._configs:
                curr_val = None
            else:
                curr_val = self._configs[key]
            if (curr_val != load_val and warn_change_key_prefix is not None
                and key.startswith(warn_change_key_prefix)
                and key not in ignore_keys):
                if key == self._config_version_key:
                    raise ValueError(f'Config version mismatch: {curr_val} -> {load_val}')
                else:
                    logger.warning(f'changed key [{key}]: {curr_val} -> {load_val}')
                unexpected_changed_keys.append(key)
            self._configs[key] = load_val
        return unexpected_changed_keys


class _CompoConfigProxy:
    """Proxy for nested attribute access (read/write)."""
    def __init__(self, config, prefix):
        object.__setattr__(self, "_root", config)
        object.__setattr__(self, "_prefix", prefix)

    def __getattr__(self, key):
        full_key = f"{self._prefix}.{key}"
        configs = self._root._configs
        if full_key in configs:
            self._root.accessed_keys.add(full_key)
            return configs[full_key]
        else:
            # allow long reach to a few non-existence fields
            return _CompoConfigProxy(self._root, full_key)

    def __setattr__(self, key, value):
        full_key = f"{self._prefix}.{key}"
        self._root._configs[full_key] = value

    def dict(self):
        configs = self._root._configs.copy()
        prefix = self._prefix + '.'
        d = {k.removeprefix(prefix): v for k, v in configs.items() if k.startswith(prefix)}
        return d.copy()

    def __bool__(self):
        return CompoConfig.resolve(self) is not None

    def __repr__(self):
        pretty_print = json.dumps(self.dict(), indent=2)
        return f"{self.__class__.__name__}(prefix={self._prefix},\n{pretty_print})"
