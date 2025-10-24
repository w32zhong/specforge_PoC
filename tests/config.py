import unittest
import tempfile
from draco.config import CompoConfig


class UnitTest(unittest.TestCase):

    def test1(self):
        cfg = CompoConfig.from_composer('./tests/configs.ini')
        kwargs = {
            "@llama2_7b": True,
            "training.max_length": 128_000
        }
        composed_cfg = cfg.composed(**kwargs)

        print(cfg)
        assert composed_cfg.modeling.tokenizer_path == "meta-llama/Llama-2-7b-chat-hf"
        assert composed_cfg.training.max_length == 128_000

        composed_cfg.dataset_generation.debug = True
        print('dataset_generation.debug changed to:', composed_cfg.dataset_generation.debug)
        print(composed_cfg)

        print('accessed_keys:', composed_cfg.accessed_keys)
        assert len(composed_cfg.accessed_keys) == 3

        with tempfile.TemporaryDirectory() as tempdir:
            composed_cfg.save_json_file(tempdir)
            unexpected_keys = cfg.load_json_file(f'{tempdir}/compo.json')
        assert len(unexpected_keys) == 5

        assert CompoConfig.resolve(composed_cfg.train) is None
        assert CompoConfig.resolve(composed_cfg.training.non_exists) is None
        assert 'max_length' in composed_cfg.training.dict()
        assert getattr(composed_cfg, 'training').tf32 is False

        composed_cfg.rock.paper = 'scissors'
        assert composed_cfg.rock.paper == 'scissors'

    @unittest.expectedFailure
    def test2(self):
        cfg = CompoConfig.from_config_file('./tests/configs.ini')
        kwargs = {
            "@non_exists": True, # <-- this is not an expected key
            "training.max_length": 128_000
        }
        composed_cfg = cfg.composed(**kwargs)

    def test3(self):
        tmp_cfg = CompoConfig({
            'path': './tests/configs.ini',
            'save_json_file_name': 'foo.json',
        })
        assert tmp_cfg.path == './tests/configs.ini'

        cfg1 = CompoConfig.from_composer('./tests/configs.ini')
        cfg2 = CompoConfig.from_composer_config(tmp_cfg)

        with tempfile.TemporaryDirectory() as tempdir:
            cfg2.save_json_file(tempdir)
            unexpected_keys = cfg1.load_json_file(f'{tempdir}/foo.json')
        assert len(unexpected_keys) == 0

    def test4(self):
        cfg = CompoConfig.from_composer_config(
            CompoConfig({
                'path': './tests/configs.ini',
                'save_json_file_name': 'foo.json',
            })
        )
        with tempfile.TemporaryDirectory() as tempdir:
            cfg['config.version'] = 'v2'
            cfg.save_json_file(tempdir)
            cfg['config.version'] = 'v1'
            try:
                cfg.load_json_file(f'{tempdir}/foo.json')
            except Exception as e:
                print(e)
                assert isinstance(e, ValueError)
            else:
                assert False

    def test5(self):
        cfg = CompoConfig.from_composer('./tests/configs.ini')
        setattr(cfg, 'foo.bar', 'baz')
        assert CompoConfig.resolve(cfg.foo.bar) is None

        cfg._configs['foo.bar'] = 'baz'
        assert CompoConfig.resolve(cfg.foo.bar) == 'baz'

    def test6(self):
        cfg = CompoConfig({
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            },
            "f": {
                "g": 4,
                "h": 5
            }
        })
        assert cfg.a == 1
        assert cfg.b.c == 2
        assert cfg.b.d.e == 3
        assert cfg.f.g == 4
        assert cfg.f.h == 5


if __name__ == '__main__':
    unittest.main()
