"""Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374

The HumanEval dataset released by OpenAI includes 164 programming problems with a function signature,
docstring, body, and several unit tests.
They were handwritten to ensure not to be included in the training set of code generation models.

Homepage: https://github.com/openai/human-eval
"""

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval


def create_task(strip_prompt):
    class Spe(GeneralHumanEval):
        def __init__(self, **kwargs):
            super().__init__(strip_prompt, **kwargs)

    return Spe


class GeneralHumanEval(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "melihcatal/spe_benchmark"

    def __init__(self, strip_prompt, k=[1, 5, 10], num_workers=16, timeout=3.0):
        super().__init__(
            stop_words=[
                "\nclass",
                "\ndef",
                "\n#",
                "\n@",
                "\nprint",
                "\nif",
                "\n```",
                "<file_sep>",
            ],
            requires_execution=True,
        )
        self.strip_prompt = strip_prompt
        self.k = k
        self.num_workers = num_workers
        self.timeout = timeout

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        if self.strip_prompt:
            return doc["prompt"].strip()
        else:
            return doc["prompt"]

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        prompt = self.get_prompt(self.dataset["test"][idx])
        generation = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(generation, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        results, _ = compute_code_eval(
            references=references,
            predictions=generations,
            k=self.k,
            num_workers=self.num_workers,
            timeout=self.timeout,
        )
        return results
