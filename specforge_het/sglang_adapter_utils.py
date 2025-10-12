import json
from colorama import Fore, Style
from specforge_het.models import *


#################################
# SGLang speculative model mixin
#################################
class SGLangAdapterMixin:
    def bind_adapter(self, config):
        # move and store base model config
        base_model_config = json.loads(config._base_model_config)
        delattr(config, '_base_model_config')

        # bind the algorithm
        speculative_algorithm = base_model_config['speculative_decoding_algorithm']
        AlgoClass = eval(speculative_algorithm)
        AlgoClass.bind_model(self, load_device=None)

        return AlgoClass.__name__, base_model_config, config

    def warn_unmatch_weights(self, weights):
        model_params_keys = set([key for key, _ in self.named_parameters()])
        load_weights_keys = set([key for key, _ in weights])

        expect_extra_keys = model_params_keys - load_weights_keys
        loading_extra_keys = load_weights_keys - model_params_keys

        if expect_extra_keys or loading_extra_keys:
            print(
                f'{Fore.YELLOW}Loading weight keys mismatch: \n\n' +
                f'model_params_keys: {model_params_keys}\n\n' +
                f'expect_extra_keys: {expect_extra_keys}\n\n' +
                f'loading_extra_keys: {loading_extra_keys}\n\n' +
                Style.RESET_ALL
            )
            #import rpdb; rpdb.set_trace()

        return weights


############################
# SGLang frontend utilities
############################

from functools import partial
from transformers import AutoTokenizer

import sglang as sgl
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.lang.backend.base_backend import BaseBackend
from sglang.lang.chat_template import get_chat_template
from sglang.global_config import global_config
from sglang.lang.ir import SglGen


class CallbackBackend(BaseBackend):
    def __init__(self, callbk, model, sgl_chat_template=None):
        super().__init__()
        self.monkey_patch_model = model
        self.monkey_patch_callbk = callbk
        self.chat_template = get_chat_template(sgl_chat_template or 'default')

    def get_chat_template(self):
        return self.chat_template

    def generate(self, sgl_prompt, messages, sampling_params):
        model = self.monkey_patch_model
        new_text, meta_info = self.monkey_patch_callbk(
            model, sgl_prompt, messages, sampling_params.to_srt_kwargs(),
        )
        return new_text, meta_info


def monkey_patch_execute(self, origin_execute_fn, state, other):
    if isinstance(other, SglGen):
        new_text, meta_info = self.backend.generate(
            self.text_, self.messages_,
            sampling_params=self.default_sampling_para
        )
        name = other.name
        self.text_ += new_text
        self.variables[name] = new_text
        self.meta_info[name] = meta_info
        self.variable_event[name].set()

    else:
        return origin_execute_fn(other)


def money_patch_sgl_function_state(s):
    s.stream_executor._execute = partial(
        monkey_patch_execute, s.stream_executor, s.stream_executor._execute, s
    )


#################################
# Evaluation via SGLang frontend
#################################
@sgl.function
def answer_mt_bench(s, sys_prompt, question_1, question_2):
    money_patch_sgl_function_state(s)
    if sys_prompt:
        s += sgl.system(sys_prompt)
    # Note: Even if sgl.system() is not added, SGLang frontend will insert
    # a default_system_prompt (see `_execute_role_begin` at lang.interpreter)
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1"))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2"))


def run_mtbench(backend, model, data_source, sampling_params,
                sys_prompt=None, sgl_chat_template=None, num_threads=1):
    def load_mtbench(filename):
        questions = []
        with open(filename, "r") as fin:
            for line in fin:
                obj = json.loads(line)
                questions.append(obj)
        return questions
    questions_jsonl, *after = data_source.split(':')
    questions_num = 80 if len(after) == 0 else int(after[0])
    questions = load_mtbench(questions_jsonl)[: questions_num]

    global_config.enable_precache_with_tracing = False

    if isinstance(backend, RuntimeEndpoint):
        sgl.set_default_backend(backend)
    else:
        sgl.set_default_backend(CallbackBackend(backend, model, sgl_chat_template))

    question_turns = [
        {
            "sys_prompt": sys_prompt,
            "question_1": q["turns"][0],
            "question_2": q["turns"][1],
        }
        for q in questions
    ]

    return answer_mt_bench.run_batch(
        question_turns,
        **sampling_params,
        num_threads=num_threads,
        progress_bar=True
    )
