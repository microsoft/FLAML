import unittest
import datasets
from flaml import icl

class TestExemplarSelector(unittest.TestCase):
    '''
    template: default or user-defined, if default, key_order is specific
    method: existing (str) or user-defined (func) or None (return in the order of train_data)
    '''
    @classmethod
    def setUpClass(cls):
        seed = 41
        cls.data = datasets.load_dataset("piqa")
        cls.train_data = cls.data["train"].shuffle(seed=seed)
        cls.test_data = cls.data["test"].shuffle(seed=seed)
        cls.key_order = ["goal", "sol1", "sol2", "label"]
        cls.exemplars = list(cls.train_data)[:8]
        cls.context = list(cls.test_data)[0]

    def test_case_existing_method_default_template(self):
        # Most cases should use the default template and existing methods
        prompt_fn = icl.ExemplarSelector.get_few_shot_template(self.exemplars, method="random", 
                                                               method_params={"k": 3}, template_params={"key_order": self.key_order})
        output= prompt_fn(self.context)
        #print("Existing method + default template: prompt = ", output)
        self.assertIsInstance(output, str)
        self.assertIn(self.context[self.key_order[0]], output)
        
    def test_case_user_template_no_method(self):
        # User specify their own template + method is not specified, k is specific
        key_order = ["sol1", "sol2", "label"]
        def few_shot_template(context, exemplars=None):
            few_shot_prompt = "User template:\n"
            
            for exemplar in exemplars:
                few_shot_prompt += "\n".join(
                    [
                        key + ": " + str(exemplar[key]) for key in key_order
                    ]
                ) + "\n"
            few_shot_prompt += "\n".join(
                [
                    key + ": " + str(context[key]) for key in key_order[:-1]
                ]
            )
            few_shot_prompt += "\n" + key_order[-1] + ": " + "\n"
            return few_shot_prompt
        prompt_fn = icl.ExemplarSelector.get_few_shot_template(self.exemplars, 
                                                               few_shot_template=few_shot_template,
                                                               method_params={"k": 3})
        output= prompt_fn(self.context)
        #print("test_user_template_No_Method: prompt = ", output)
        self.assertIsInstance(output, str)
        self.assertIn(self.context[key_order[0]], output)
        self.assertIn(self.context[key_order[1]], output)
        # should pick first 3 exemplars 
        self.assertIn(self.exemplars[0][key_order[1]], output)
        self.assertIn(self.exemplars[1][key_order[1]], output)
        self.assertIn(self.exemplars[2][key_order[1]], output)
        self.assertNotIn(self.exemplars[3][key_order[1]], output)
        
        
    def test_case_user_method_no_k(self):
        # User specify their method is not specified, k is not specific
        def user_method(context):
            return self.exemplars[3:5]
        # key_order should be provided if we use the default template
        prompt_fn = icl.ExemplarSelector.get_few_shot_template(self.exemplars, 
                                                               method = user_method,
                                                               template_params={"key_order": self.key_order})
        output= prompt_fn(self.context)
        #print("test_user_method_no_k: prompt = ", output)
        self.assertIsInstance(output, str)
        self.assertIn(self.context[self.key_order[0]], output)
        self.assertIn(self.context[self.key_order[1]], output)
        self.assertIn(self.context[self.key_order[2]], output)
        # should pick the 3rd,4th exemplars 
        self.assertIn(self.exemplars[3][self.key_order[2]], output)
        self.assertIn(self.exemplars[4][self.key_order[2]], output)
        self.assertNotIn(self.exemplars[5][self.key_order[2]], output)
        self.assertNotIn(self.exemplars[2][self.key_order[2]], output)    
        
    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            icl.ExemplarSelector.get_few_shot_template(self.exemplars, method="nonexistent")
  



if __name__ == '__main__':
    unittest.main()


