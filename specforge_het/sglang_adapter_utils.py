import json
from colorama import Fore, Style
from specforge_het.models import *


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


################################
# SGLang frontend utilities
################################

import sglang as sgl
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.lang.backend.base_backend import BaseBackend
from sglang.global_config import global_config
from sglang.lang.ir import SglGen


class MonkeyPatchLangBackend(BaseBackend):
    def __init__(self, llm):
        super().__init__()
        self.monkey_patch_llm = llm

        from sglang.lang.chat_template import get_chat_template_by_model_path
        self.chat_template = get_chat_template_by_model_path(
            #'meta-llama/Llama-2-7b-chat-hf'
            'Qwen/Qwen3-4B-Instruct-2507'
        )

    def get_chat_template(self):
        return self.chat_template

    def generate(self, prompt, sampling_params):
        llm = self.monkey_patch_llm
        tokenizer = llm.tokenizer_manager.tokenizer
        batch_texts, batch_meta_info = batch_generate(
            llm,
            tokenizer,
            [prompt],
            sampling_params.to_srt_kwargs(),
        )
        return batch_texts[0], batch_meta_info[0]


def monkey_patch_execute(self, origin_execute_fn, state, other):
    if isinstance(other, SglGen):
        messages = self.messages_
        llm = self.backend.monkey_patch_llm
        tokenizer = llm.tokenizer_manager.tokenizer
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text, meta_info = self.backend.generate(
            self.text_, sampling_params=self.default_sampling_para
        )

        name = other.name
        self.text_ += text
        self.variables[name] = text
        self.meta_info[name] = meta_info
        self.variable_event[name].set()

    else:
        return origin_execute_fn(other)


#################################
# Evaluation via SGLang frontend
#################################
@sgl.function
def answer_mt_bench(s, sys_prompt, question_1, question_2):
    s.stream_executor._execute = partial(
        monkey_patch_execute, s.stream_executor, s.stream_executor._execute, s
    )
    if sys_prompt:
        s += sgl.system(sys_prompt)
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1"))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2"))


def run_mtbench(backend, model, data_source, sampling_params,
                sys_prompt=None, num_threads=1):
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
        sgl.set_default_backend(MonkeyPatchLangBackend(model))

    question_turns = [
        {"question_1": q["turns"][0], "question_2": q["turns"][1]}
        for q in questions
    ]

    begin = time.perf_counter()
    res = answer_mt_bench.run_batch(
        question_turns,
        **sampling_params,
        num_threads=num_threads,
        progress_bar=True
    )
    time_cost = time.perf_counter() - begin

    meta_info = dict(completion_tokens=0, spec_verify_ct=0)
    for i, Q in enumerate(question_turns):
        print('=' * 80)
        print([Q['question_1']])
        print(res[i].get_var('answer_1'))
        mi = res[i].get_meta_info('answer_1')
        meta_info['completion_tokens'] += mi['completion_tokens']
        meta_info['spec_verify_ct'] += mi['spec_verify_ct']

        print('=' * 80)
        print([Q['question_2']])
        print(res[i].get_var('answer_2'))
        mi = res[i].get_meta_info('answer_2')
        meta_info['completion_tokens'] += mi['completion_tokens']
        meta_info['spec_verify_ct'] += mi['spec_verify_ct']
    print('-' * 80)

    return meta_info, time_cost
