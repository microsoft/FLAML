from functools import partial
import random
import numpy as np
from flaml.autogen.oai.selection_methods import RandomSelect

class ExemplarSelector:
    METHOD_MAPPING = {
        "random": RandomSelect,
        # You can add more methods here...
    }

    @classmethod
    def get_few_shot_template(cls, train_data, method=None, few_shot_template=None, method_params=None, template_params=None):
        if isinstance(method, str):
            method_class = cls.METHOD_MAPPING.get(method, None)
            if method_class is not None:
                method = method_class(train_data, **method_params).select
            else:
                raise ValueError(f"The specified method '{method}' is not recognized.")
        return partial(cls.construct_template, train_data=train_data, method=method,
                       few_shot_template=few_shot_template, method_params=method_params or {},
                       template_params=template_params or {})

    @staticmethod
    def construct_template(context, train_data, method=None, few_shot_template=None, method_params=None, template_params=None):
        
        if method is None:
            k = method_params.get('k', np.inf)
            exemplars = train_data[:k] if len(train_data) >= k else train_data
        else:
            exemplars = method(context)

        if few_shot_template is not None:
            return few_shot_template(context, exemplars=exemplars)
        else:
            key_order = template_params.get('key_order', None)
            return ExemplarSelector.default_template(context, exemplars, key_order)

    @staticmethod
    def default_template(context, exemplars, key_order):
        few_shot_prompt = ""
        for examplar in exemplars:
            few_shot_prompt += "\n".join(
                [
                    key + ": " + str(examplar[key]) for key in key_order
                ]
            ) + "\n"
        few_shot_prompt += "\n".join(
            [
                key + ": " + str(context[key]) for key in key_order[:-1]
            ]
        )
        few_shot_prompt += "\n" + key_order[-1] + ": " + "\n"
        return few_shot_prompt

