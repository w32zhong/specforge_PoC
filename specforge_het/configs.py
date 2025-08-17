import os
import json
import configparser


class Configs():
    def __init__(self, configs):
        self._configs = configs

    def get_obj(self):
        return self._configs

    def set_obj(self, key, val):
        self._configs[key] = val

    def __getattr__(self, attr):
        return self.from_configs(self._configs, attr)

    def __repr__(self):
        return self.pretty_json()

    @classmethod
    def from_configs(cls, configs, prefix_key=''):
        new_configs = dict()
        for key in configs:
            if key.startswith(prefix_key + '.'):
                new_key = key[len(prefix_key) + 1:]
                new_configs[new_key] = configs[key]
            elif prefix_key in configs:
                return configs[prefix_key]
        return cls(new_configs)

    @classmethod
    def from_config_file(cls, config_file, **injects):
        return cls(cls.get_configs(config_file, **injects))

    @classmethod
    def get_configs(cls, config_file, **injects):
        config = configparser.ConfigParser(allow_no_value=True)
        assert os.path.exists(config_file), config_file
        config.read(config_file)

        # first pass (parse config file)
        configs = dict()
        for section in config.sections():
            items = config[section]
            for item_key in items:
                key = f'{section}.{item_key}'
                val = eval(items.get(item_key))
                configs[key] = val

        # second pass (inject @expanded configs)
        used_inj_keys = set()
        expand_keys = list(filter(lambda k: '@' in k, configs.keys()))
        for key in expand_keys:
            val = configs[key]
            for inj_key in filter(lambda k: '@' in k, injects.keys()):
                if inj_key in key:
                    new_key = key.replace(f'{inj_key}.', '')
                    configs[new_key] = val
                    used_inj_keys.add(inj_key)
        expanded_configs = {k: v for k, v in configs.items() if '@' not in k}

        # third pass (inject exact-path matches)
        for key in expanded_configs.keys():
            if key in injects:
                try:
                    new_val = eval(injects[key])
                except:
                    new_val = injects[key]
                expanded_configs[key] = new_val
                used_inj_keys.add(key)

        unused_inject_keys = set(injects.keys()) - used_inj_keys
        assert len(unused_inject_keys) == 0, unused_inject_keys

        return expanded_configs

    def pretty_json(self):
        return json.dumps(self._configs, indent=2)

    def save_json(self, directory, fname='configs.json'):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, fname), 'w') as fh:
            json.dump(self._configs, fh, indent=2, sort_keys=True)
