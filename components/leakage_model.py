from components import Component
from packages.sidechannel.leakagemodels.leakage_models import *
from components.decorators.database_decorators import leakage_model_database


class LeakageModel(Component):

    def __init__(self, settings):
        super().__init__(settings)
        self.leakage_model = LeakageModelEnum.Identity
        self.cipher = CipherEnum.AES128
        self.state = AesStateEnum.SBOX
        self.aes_round = 0
        self.byte = 2
        self.attack_direction = AttackDirectionEnum.INPUT
        self.cipher_direction = CipherDirectionEnum.ENCRYPTION
        self.classes = 256

    @leakage_model_database
    def add_leakage_model(self, leakage_model=None, cipher=None, state=None, aes_round=None, byte=None, attack_direction=None,
                          cipher_direction=None):
        self.leakage_model = leakage_model
        self.cipher = cipher
        self.state = state
        self.aes_round = aes_round
        self.byte = byte
        self.attack_direction = attack_direction
        self.cipher_direction = cipher_direction
        self.__set_classes()

    def __set_classes(self):
        self.classes = get_classes(self.leakage_model)

    def create_intermediates(self, data, key):
        if "aes" in self.cipher:
            leakage_model_obj = AesLeakageModel(self.cipher, self.state, self.aes_round, self.byte, self.attack_direction,
                                                self.cipher_direction)
            return leakage_model_obj.get_aes_intermediates(data, key)

    def create_labels_from_default_leakage_model(self, intermediates, default_leakage_model):
        if "aes" in self.cipher:
            leakage_model_obj = AesLeakageModel(self.cipher, self.state, self.aes_round, self.byte, self.attack_direction,
                                                self.cipher_direction)
            return leakage_model_obj.get_labels_from_default_leakage_model(intermediates, default_leakage_model)

    def create_labels(self, intermediates):
        if "aes" in self.cipher:
            leakage_model_obj = AesLeakageModel(self.cipher, self.state, self.aes_round, self.byte, self.attack_direction,
                                                self.cipher_direction)
            return leakage_model_obj.get_labels(intermediates, self.leakage_model)

    def create_labels_key_guesses(self, data):

        """ Generate array with labels for all key guesses. This function only supports 8-bit guesses maximum """

        if "aes" in self.cipher:
            leakage_model_obj = AesLeakageModel(self.cipher, self.state, self.aes_round, self.byte, self.attack_direction,
                                                self.cipher_direction)
            return leakage_model_obj.get_labels_key_guesses(data, self.leakage_model)
