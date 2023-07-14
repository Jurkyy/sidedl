from packages.crypto.symmetric.aes_temp import *
import enum


class LeakageModelEnum(enum.Enum):
    Identity = "ID"
    Xor = "xor"
    HammingWeight = "HW"
    HammingDistance = "HD"
    Bit = "Bit"


class CipherEnum(enum.Enum):
    AES128 = "aes128"
    AES192 = "aes192"
    AES256 = "aes256"
    TripleDES = "3des"
    Present = "present"
    Speck = "speck"


class AesStateEnum(enum.Enum):
    ADDROUNDKEY = "add_round_key"
    SBOX = "sbox"
    INVSBOX = "inv_sbox"
    SHIFTROWS = "shift_rows"
    INVSHIFTROWS = "inv_shift_rows"
    MIXCOLUMNS = "mix_columns"
    INVMIXCOLUMNS = "inv_mix_columns"
    PLAINTEXT = "plaintext"
    CIPHERTEXT = "plaintext"


class AttackDirectionEnum(enum.Enum):
    INPUT = "input"
    OUTPUT = "output"


class CipherDirectionEnum(enum.Enum):
    ENCRYPTION = "encryption"
    DECRYPTION = "decryption"


class AesLeakageModelsEnum(enum.Enum):
    SBOX_OUTPUT_ROUND1 = "sbox_round_1"
    SBOX_INPUT_XOR_SBOX_OUTPUT_ROUND1 = "sbox_input_xor_sbox_output_round_1"
    INV_SBOX_INPUT_ROUND10 = "inv_sbox_input_round_10"
    INV_SBOX_INPUT_XOR_INV_SBOX_OUTPUT_ROUND10 = "inv_sbox_input_xor_inv_sbox_output_round_10"
    INV_SBOX_INPUT_XOR_CIPHERTEXT_ROUND10 = "inv_sbox_input_xor_ciphertext_round_10"


def get_classes(leakage_model):
    if leakage_model == LeakageModelEnum.HammingWeight or leakage_model == LeakageModelEnum.HammingDistance:
        return 9
    elif leakage_model == LeakageModelEnum.Identity or leakage_model == LeakageModelEnum.Xor:
        return 256
    elif leakage_model == LeakageModelEnum.Bit:
        return 2


class AesLeakageModel:

    def __init__(self, cipher, aes_state, aes_round, byte, attack_direction, cipher_direction, aes_state_first=None, aes_state_second=None,
                 aes_round_first=None, aes_round_second=None):
        self.cipher = cipher
        self.aes_state = aes_state
        self.aes_round = aes_round
        self.byte = byte
        self.attack_direction = attack_direction
        self.cipher_direction = cipher_direction

        self.aes_state_first = aes_state_first
        self.aes_state_second = aes_state_second

        self.aes_round_first = aes_round_first
        self.aes_round_second = aes_round_second

    def single_byte_intermediate(self, state_byte_index):
        pass

    def multi_byte_intermediate(self, state_byte_index_list):
        pass

    def get_aes_intermediates(self, data, keys):

        if self.cipher_direction is CipherDirectionEnum.ENCRYPTION and self.attack_direction is AttackDirectionEnum.INPUT:
            return get_encryption_intermediates(data, keys)
        if self.cipher_direction is CipherDirectionEnum.DECRYPTION and self.attack_direction is AttackDirectionEnum.INPUT:
            return get_decryption_intermediates(data, keys)
        if self.cipher_direction is CipherDirectionEnum.ENCRYPTION and self.attack_direction is AttackDirectionEnum.OUTPUT:
            return get_encryption_intermediates(data, keys)
        if self.cipher_direction is CipherDirectionEnum.DECRYPTION and self.attack_direction is AttackDirectionEnum.OUTPUT:
            return get_decryption_intermediates(data, keys)

    def get_labels_from_default_leakage_model(self, intermediates, aes_leakage_model):
        if aes_leakage_model == AesLeakageModelsEnum.SBOX_OUTPUT_ROUND1:
            return intermediates["sbox_1"][:, self.byte]
        if aes_leakage_model == AesLeakageModelsEnum.SBOX_INPUT_XOR_SBOX_OUTPUT_ROUND1:
            return intermediates["add_round_key_0"][:, self.byte], intermediates["sbox_1"][:, self.byte]
        if aes_leakage_model == AesLeakageModelsEnum.INV_SBOX_INPUT_ROUND10:
            return intermediates["inv_shift_rows_0"][:, self.byte]
        if aes_leakage_model == AesLeakageModelsEnum.INV_SBOX_INPUT_XOR_INV_SBOX_OUTPUT_ROUND10:
            return intermediates["inv_shift_rows_0"][:, self.byte], intermediates["inv_sbox_0"][:, self.byte]
        if aes_leakage_model == AesLeakageModelsEnum.INV_SBOX_INPUT_XOR_CIPHERTEXT_ROUND10:
            return intermediates["inv_shift_rows_0"][:, self.byte], intermediates["ciphertext"][:, shift_row_mask[self.byte]]

    def get_labels(self, intermediates, leakage_model):
        if leakage_model is LeakageModelEnum.Identity:
            return self.get_identity(intermediates)
        elif leakage_model is LeakageModelEnum.HammingWeight:
            return self.get_hamming_weight(intermediates)
        elif leakage_model is LeakageModelEnum.Xor:
            return self.get_xor(intermediates)
        elif leakage_model is LeakageModelEnum.HammingDistance:
            return self.get_hamming_distance(intermediates)

    def get_labels_key_guesses(self, data, leakage_model):
        labels_key_guess = np.zeros((256, len(data)))
        for key_guess in range(256):
            key = np.zeros(16)
            key[self.byte] = key_guess
            keys = np.full([len(data), 16], key)
            intermediates = self.get_aes_intermediates(data, keys)
            labels_key_guess[key_guess] = self.get_labels(intermediates, leakage_model)
        return labels_key_guess

    def get_identity(self, intermediates):
        return intermediates[f"{self.aes_state}_{self.aes_round}"][:, self.byte]

    def get_hamming_weight(self, intermediates):
        return [bin(iv).count("1") for iv in intermediates[f"{self.aes_state}_{self.aes_round}"][:, self.byte]]

    def get_xor(self, intermediates):
        intermediates_1, intermediates_2 = self.__get_two_intermediates(intermediates)
        return [int(iv1) ^ int(iv2) for iv1, iv2 in zip(intermediates_1, intermediates_2)]

    def get_hamming_distance(self, intermediates):
        intermediates_1, intermediates_2 = self.__get_two_intermediates(intermediates)
        return [bin(int(iv1) ^ int(iv2)).count("1") for iv1, iv2 in zip(intermediates_1, intermediates_2)]

    def __get_two_intermediates(self, intermediates):
        label_first, label_second = self.__get_intermediate_label_two_states()
        byte_first = self.__get_byte_after_shift_rows(self.byte)
        byte_second = self.byte
        intermediates_1 = intermediates[label_first][:, byte_first]
        intermediates_2 = intermediates[label_second][:, byte_second]
        return intermediates_1, intermediates_2

    def __get_intermediate_label_two_states(self):
        if self.aes_state_first in ["plaintext", "ciphertext"]:
            label_first = self.aes_state_first
        else:
            label_first = f"{self.aes_state_first}_{self.aes_round_first}"

        if self.aes_state_second in ["plaintext", "ciphertext"]:
            label_second = self.aes_state_second
        else:
            label_second = f"{self.aes_state_second}_{self.aes_round_second}"
        return label_first, label_second

    def __count_shift_rows(self):
        shift_row_counts = 0
        if self.cipher_direction is CipherDirectionEnum.ENCRYPTION:
            labels = encryption_intermediates_labels()
        else:
            labels = decryption_intermediates_labels()

        for label in labels:
            if "shift_row" in label:
                shift_row_counts += 1
            if label is self.aes_state_second:
                return shift_row_counts

    def __get_byte_after_shift_rows(self, byte):
        byte_tmp = byte.copy()
        shift_row_counts = self.__count_shift_rows()
        for sr in range(shift_row_counts):
            byte_tmp = shift_row_mask[byte_tmp]
        return byte_tmp
