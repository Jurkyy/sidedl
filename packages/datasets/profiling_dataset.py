import numpy as np


class CipherDataset:

    def __init__(self):
        """ side-channel traces """
        self.x_profiling = None
        self.x_validation = None
        self.x_attack = None

        """ categorical labels """
        self.y_profiling = None
        self.y_validation = None
        self.y_attack = None

        """ plaintexts """
        self.plaintexts_profiling = None
        self.plaintexts_validation = None
        self.plaintexts_attack = None

        """ ciphertexts """
        self.ciphertexts_profiling = None
        self.ciphertexts_validation = None
        self.ciphertexts_attack = None

        """ keys """
        self.keys_profiling = None
        self.keys_validation = None
        self.keys_attack = None

        """ main key """
        self.key_profiling = None
        self.key_validation = None
        self.key_attack = None

        """ number of traces to compute SCA metrics"""
        self.nt_sca_metrics_validation = None
        self.nt_sca_metrics_attack = None

    def set_main_keys(self):
        if self.keys_validation is not None:
            if np.all(self.keys_validation == self.keys_validation[0]):
                self.key_validation = self.keys_validation[0]
        if self.keys_attack is not None:
            if np.all(self.keys_attack == self.keys_attack[0]):
                self.key_attack = self.keys_attack[0]

    def get_validation_key_byte(self, byte=None):
        return self.key_validation[byte]

    def get_attack_key_byte(self, byte=None):
        return self.key_attack[byte]

    def set_number_of_traces_sca_metrics_attack(self, number_of_traces):
        self.nt_sca_metrics_attack = number_of_traces

    def set_number_of_traces_sca_metrics_validation(self, number_of_traces):
        self.nt_sca_metrics_validation = number_of_traces


class PublicKeySegmentsDataset:

    def __init__(self):
        """ side-channel traces """
        self.x_profiling_segments = None
        self.x_validation_segments = None
        self.x_attack_segments = None

        """ categorical labels """
        self.y_profiling_segments = None
        self.y_validation_segments = None
        self.y_attack_segments = None
