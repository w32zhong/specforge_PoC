import unittest
import tempfile
import logging
from draco.config import CompoConfig


class UnitTest(unittest.TestCase):

    def test1(self):
        cfg = CompoConfig({})
        cfg.load_json_file('tests/heterogeneous_speculative_modeling.json', warn_change_key_prefix=None)
        cfg.algorithm_config.draft_depth = 2
        print(cfg)

        logger = logging.getLogger('draco.speculative_decoding.base_hf')
        logger.setLevel(logging.INFO)

        # compose draft and target models
        from draco.speculative_decoding.vanilla.modeling import VanillaSpeculativeDecodingModel
        model = VanillaSpeculativeDecodingModel.from_composer_config(cfg)
        assert len(model.draft_model.layers) == 1

        # test save and load
        with tempfile.TemporaryDirectory() as tempdir:
            model.save_pretrained(tempdir)
            model = VanillaSpeculativeDecodingModel.from_pretrained(tempdir)
        assert len(model.draft_model.layers) == 1
