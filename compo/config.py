import os
import json
import logging
import configparser
logger = logging.getLogger(__name__)


class CompoConfig:
    def __init__(self, config_dict: dict):
        self._configs = config_dict.copy()
        self.accessed_keys = set()

    def __getitem__(self, key):
        return self._configs[key]

    def __setitem__(self, key, value):
        self._configs[key] = value

    @classmethod
    def from_config_file(cls, path: str):
        config = configparser.ConfigParser(allow_no_value=True)
        assert os.path.exists(path), path
        config.read(path)

        # parse config file
        config_dict = dict()
        for section in config.sections():
            items = config[section]
            for item_key in items:
                key = f'{section}.{item_key}'
                val = eval(items.get(item_key))
                config_dict[key] = val
        return cls(config_dict)

    def composed(self, **inject_arguments):
        used_inject_keys = set()
        base_configs = self._configs.copy()

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
        return _CompoConfigProxy(self, key)

    def pretty_json(self):
        return json.dumps(self._configs, indent=2, sort_keys=True)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.pretty_json()})"

    def save_json_file(self, directory, fname='compo.json'):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, fname), 'w') as fh:
            json.dump(self._configs, fh, indent=2, sort_keys=True)

    def load_json_file(self, path='compo.json', warn_change_key_prefix='', ignore_keys=[]):
        with open(path, 'r') as fh:
            loaded_configs = json.load(fh)
        unexpected_changed_keys = []
        for key, new_val in loaded_configs.items():
            if key not in self._configs:
                old_val = None
            else:
                old_val = self._configs[key]
            if (old_val != new_val and
                key.startswith(warn_change_key_prefix) and
                key not in ignore_keys):
                logger.warn(f'changed key [{key}]: {old_val} -> {new_val}')
                self._configs[key] = new_val
                unexpected_changed_keys.append(key)
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
            return _CompoConfigProxy(self._root, full_key)

    def __setattr__(self, key, value):
        full_key = f"{self._prefix}.{key}"
        self._root._configs[full_key] = value

    def dict(self):
        configs = self._root._configs.copy()
        prefix = self._prefix + '.'
        return {k.removeprefix(prefix): v for k, v in configs.items() if k.startswith(prefix)}

    def __repr__(self):
        pretty_print = json.dumps(self.dict(), indent=2)
        return f"{self.__class__.__name__}(prefix={self._prefix},\n{pretty_print})"
