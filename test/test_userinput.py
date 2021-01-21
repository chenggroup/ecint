import os

import pytest
from ecint.main import get_userinput

input_config_general = {
    "workflow": "CaoSingleWorkChain",
    # "webhook": "https://oapi.dingtalk.com/robot/send?access_token=xxx",
    "resdir": "results_json",
    "structure": "resources/ethane_1.xyz",
    "metadata": {
        "kind_section": {
            "C": {
                "BASIS_SET": "TZV2P-MOLOPT-GTH",
                "POTENTIAL": "GTH-PBE"
            },
            "O": {
                "BASIS_SET": "TZV2P-MOLOPT-GTH",
                "POTENTIAL": "GTH-PBE"
            },
            "Cu": {
                "BASIS_SET": "DZVP-MOLOPT-SR-GTH",
                "POTENTIAL": "GTH-PBE"
            }
        }
    }
}

input_config_combine = {
    "workflow": "ActiveLearning",
    "datadirs": ["datadirs"],
    "imd": [{"structures_folder": "resources", "TEMP": [330, 430]},
            {"structures_folder": "resources", "PRES": [-1]}],
    "subdata": {
        "exploration": {"max": 1}
    }
}


class TestUserInput:
    def test_general(self):
        userinput = get_userinput(input_config_general)
        inp = userinput.get_workflow_inp()
        assert inp['resdir'] == os.path.dirname(__file__)

    def test_combine(self):
        userinput = get_userinput(input_config_combine)
        inp = userinput.get_workflow_inp()
        assert inp['exploration']['max'] == 1
        assert inp['exploration']['resdir'] == os.path.dirname(__file__)
        assert inp['labeling']['resdir'] == os.path.dirname(__file__)
        assert inp['training']['resdir'] == os.path.dirname(__file__)
        # assert inp['imd']
