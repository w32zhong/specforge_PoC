import unittest
from compo.config import CompoConfig
import tempfile


class UnitTest(unittest.TestCase):

    def test1(self):
        cfg = CompoConfig.from_config_file('./tests/configs.ini')
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

        assert not composed_cfg.train.dict()
        assert not composed_cfg.training.non_exists.dict()
        assert 'max_length' in composed_cfg.training.dict()
        assert getattr(composed_cfg, 'training').tf32 is False

    @unittest.expectedFailure
    def test2(self):
        cfg = CompoConfig.from_config_file('./tests/configs.ini')
        kwargs = {
            "@non_exists": True, # <-- this is not an expected key
            "training.max_length": 128_000
        }
        composed_cfg = cfg.composed(**kwargs)


if __name__ == '__main__':
    unittest.main()
