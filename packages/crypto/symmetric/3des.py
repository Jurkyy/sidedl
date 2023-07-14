import numpy as np

# Permutation and translation tables for DES
pc1 = [56, 48, 40, 32, 24, 16, 8,
       0, 57, 49, 41, 33, 25, 17,
       9, 1, 58, 50, 42, 34, 26,
       18, 10, 2, 59, 51, 43, 35,
       62, 54, 46, 38, 30, 22, 14,
       6, 61, 53, 45, 37, 29, 21,
       13, 5, 60, 52, 44, 36, 28,
       20, 12, 4, 27, 19, 11, 3
       ]

# number left rotations of pc1
left_rotations = [
    1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1
]

# permuted choice key (table 2)
pc2 = [
    13, 16, 10, 23, 0, 4,
    2, 27, 14, 5, 20, 9,
    22, 18, 11, 3, 25, 7,
    15, 6, 26, 19, 12, 1,
    40, 51, 30, 36, 46, 54,
    29, 39, 50, 44, 32, 47,
    43, 48, 38, 55, 33, 52,
    45, 41, 49, 35, 28, 31
]

# initial permutation IP
ip = [57, 49, 41, 33, 25, 17, 9, 1,
      59, 51, 43, 35, 27, 19, 11, 3,
      61, 53, 45, 37, 29, 21, 13, 5,
      63, 55, 47, 39, 31, 23, 15, 7,
      56, 48, 40, 32, 24, 16, 8, 0,
      58, 50, 42, 34, 26, 18, 10, 2,
      60, 52, 44, 36, 28, 20, 12, 4,
      62, 54, 46, 38, 30, 22, 14, 6
      ]

expansion_table = [
    31, 0, 1, 2, 3, 4,
    3, 4, 5, 6, 7, 8,
    7, 8, 9, 10, 11, 12,
    11, 12, 13, 14, 15, 16,
    15, 16, 17, 18, 19, 20,
    19, 20, 21, 22, 23, 24,
    23, 24, 25, 26, 27, 28,
    27, 28, 29, 30, 31, 0
]

sbox = [
    # S1
    [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7,
     0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8,
     4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0,
     15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],

    # S2
    [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10,
     3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5,
     0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15,
     13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],

    # S3
    [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8,
     13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1,
     13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7,
     1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],

    # S4
    [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15,
     13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9,
     10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4,
     3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],

    # S5
    [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9,
     14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6,
     4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14,
     11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],

    # S6
    [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11,
     10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8,
     9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6,
     4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],

    # S7
    [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1,
     13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6,
     1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2,
     6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],

    # S8
    [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7,
     1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2,
     7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8,
     2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
]

# 32-bit permutation function P used on the output of the S-boxes
permut_sbox = [
    15, 6, 19, 20, 28, 11,
    27, 16, 0, 14, 22, 25,
    4, 17, 30, 9, 1, 7,
    23, 13, 31, 26, 2, 8,
    18, 12, 29, 5, 21, 10,
    3, 24
]

# final permutation IP^-1
fp = [
    39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30,
    37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28,
    35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26,
    33, 1, 41, 9, 49, 17, 57, 25,
    32, 0, 40, 8, 48, 16, 56, 24
]


class TripleDES:

    def __init__(self, key1, key2, key3):
        self.cipher1 = DES(key1)
        self.cipher2 = DES(key2)
        self.cipher3 = DES(key3)

    def encryption(self, plaintext):
        return self.cipher3.encryption(self.cipher2.decryption(self.cipher1.encryption(plaintext)))

    def decryption(self, ciphertext):
        return self.cipher1.decryption(self.cipher2.encryption(self.cipher3.decryption(ciphertext)))

    @staticmethod
    def get_intermediate_values(plaintexts, keys1, keys2, keys3, ciphertexts, leakage_model):
        """
        Function to compute the intermediate values
        :param plaintexts:  the plaintexts
        :param keys1: the keys
        :param keys2: the keys
        :param keys3: the keys
        :param ciphertexts: the ciphertexts
        :param leakage_model: the leakage model
        :return: list of intermediate values from leakage model.
        """
        if leakage_model['direction'] == 'Encryption':
            # tWhich one of the seperate DES parts to attack
            des_part = leakage_model['des_part']
            if des_part == 1:
                return DES.get_intermediate_values(plaintexts, keys1, ciphertexts, leakage_model)
            bit_keys = np.unpackbits(np.array(keys1, dtype=np.ubyte), axis=1)
            bit_plaintexts = np.unpackbits(np.array(plaintexts, dtype=np.ubyte), axis=1)
            new_ciphers = []
            des = DES(bit_keys[0])
            prev_key = bit_keys[0]
            for plaintext, key in zip(bit_plaintexts, bit_keys):
                if not np.array_equal(prev_key, key):
                    des.expand_key(key)
                    prev_key = key
                new_ciphers.append(des.encryption(plaintext))
            new_ciphers = np.array(new_ciphers, dtype=np.bool_)
            if des_part == 2:
                leakage_model['direction'] = 'Decryption'
                return DES.get_intermediate_values(plaintexts, keys2, np.packbits(new_ciphers, axis=1), leakage_model)
            new_plaintexts = []
            bit_keys = np.unpackbits(np.array(keys2, dtype=np.ubyte), axis=1)
            des = DES(bit_keys[0])
            prev_key = bit_keys[0]
            for ciphertext, key in zip(new_ciphers, bit_keys):
                if not np.array_equal(prev_key, key):
                    des.expand_key(key)
                    prev_key = key
                new_plaintexts.append(des.decryption(ciphertext))
            new_plaintexts = np.packbits(np.array(new_plaintexts, dtype=np.bool_))
            return DES.get_intermediate_values(new_plaintexts, keys3, ciphertexts, leakage_model)
        # Which one of the seperate DES parts to attack
        des_part = leakage_model['des_part']
        if des_part == 3:
            return DES.get_intermediate_values(plaintexts, keys1, ciphertexts, leakage_model)
        bit_keys = np.unpackbits(np.array(keys3, dtype=np.ubyte), axis=1)
        bit_ciphers = np.unpackbits(np.array(ciphertexts, dtype=np.ubyte), axis=1)
        new_plaintexts = []
        des = DES(bit_keys[0])
        prev_key = bit_keys[0]
        for ciphertext, key in zip(bit_ciphers, bit_keys):
            if not np.array_equal(prev_key, key):
                des.expand_key(key)
                prev_key = key
            new_plaintexts.append(des.decryption(ciphertext))
        new_plaintexts = np.array(new_plaintexts, dtype=np.bool_)
        if des_part == 2:
            leakage_model['direction'] = 'Encryption'
            return DES.get_intermediate_values(np.packbits(new_plaintexts, axis=1), keys2, ciphertexts, leakage_model)
        bit_keys = np.unpackbits(np.array(keys2, dtype=np.ubyte), axis=1)
        new_ciphers = []
        des = DES(bit_keys[0])
        prev_key = bit_keys[0]
        for plaintext, key in zip(new_plaintexts, bit_keys):
            if not np.array_equal(prev_key, key):
                des.expand_key(key)
                prev_key = key
            new_ciphers.append(des.encryption(plaintext))
        new_ciphers = np.packbits(np.array(new_plaintexts, dtype=np.bool_))
        return DES.get_intermediate_values(plaintexts, keys1, new_ciphers, leakage_model)

    @staticmethod
    def get_intermediate_values_multilabel(plaintexts, keys1, keys2, keys3, ciphertexts, leakage_models):
        """
        Function to compute the intermediate values
        :param plaintexts:  the plaintexts
        :param keys1: the keys
        :param keys2: the keys
        :param keys3: the keys
        :param ciphertexts: the ciphertexts
        :param leakage_models: the leakage models
        :return: list of intermediate values from leakage model of each leakage models
        """
        labels = []
        for leakage_model in leakage_models:
            labels.append(TripleDES.get_intermediate_values(plaintexts, keys1, keys2, keys3, ciphertexts, leakage_model))

        return labels


class DES:

    def __init__(self, key):
        """
        Initializes object with 56 bit key
        :param key: array with size 56 key array
        """
        self.L = np.zeros(32, dtype=np.bool_)
        self.R = np.zeros(32, dtype=np.bool_)
        self.KS = np.zeros((16, 48), dtype=np.bool_)
        self.expand_key(key)

    def expand_key(self, key):
        """"
        Expands key according to nist standard
        """
        permuted_key = key[pc1]
        c = permuted_key[:28]
        d = permuted_key[28:]
        for i in range(16):
            c = np.roll(c, -left_rotations[i])
            d = np.roll(d, -left_rotations[i])
            self.KS[i] = np.append(c, d)[pc2]

    def encryption(self, plaintext):
        """
        Performs encryption operation
        :param plaintext: array of 1/0's or booleans
        :return:
        """
        permuted_plaintext = np.array(plaintext[ip], dtype=np.bool_)
        self.L = permuted_plaintext[:32]
        self.R = permuted_plaintext[32:]
        for i in range(16):
            expanded_r = self.R[expansion_table]
            to_be_subbed = np.logical_xor(self.KS[i], expanded_r)
            output_sboxes = []
            for j in range(8):
                # We need to pad otherwise the 0's are in
                # I know this is a bit of a disaster, sorry
                sub_val = np.packbits(np.append(np.array([0, 0, 0, 0], dtype=np.bool_),
                                                to_be_subbed[j*6+1:j*6+5]))[0]
                if to_be_subbed[j*6]:
                    sub_val += 32
                if to_be_subbed[j*6 + 5]:
                    sub_val += 16
                output_val = sbox[j][sub_val]
                output_sboxes = np.append(output_sboxes, DES.convert_int_to_bool_arr(output_val))
            output_sboxes = np.array(output_sboxes, dtype=np.bool_)[permut_sbox]
            temp = self.R.copy()
            self.R = np.logical_xor(self.L, output_sboxes)
            self.L = temp
        # The order of L and R are swapped to form the pre-output block.
        output = np.append(self.R, self.L)[fp]
        return output

    def decryption(self, ciphertext):
        """
        Performs encryption operation
        :param ciphertext: array of 1/0's or booleans
        :return:
        """
        permuted_ciphertext = np.array(ciphertext, dtype=np.bool_)[ip]
        self.L = permuted_ciphertext[32:]
        self.R = permuted_ciphertext[:32]
        for i in range(16):
            expanded_l = self.L[expansion_table]
            to_be_subbed = np.logical_xor(self.KS[15 - i], expanded_l)
            output_sboxes = []
            for j in range(8):
                # We need to pad otherwise the 0's are in
                # I know this is a bit of a disaster, sorry
                sub_val = np.packbits(np.append(np.array([0, 0, 0, 0], dtype=np.bool_),
                                                to_be_subbed[j * 6 + 1:j * 6 + 5]))[0]
                if to_be_subbed[j * 6]:
                    sub_val += 32
                if to_be_subbed[j * 6 + 5]:
                    sub_val += 16
                output_val = sbox[j][sub_val]

                output_sboxes = np.append(output_sboxes, DES.convert_int_to_bool_arr(output_val))
            output_sboxes = np.array(output_sboxes, dtype=np.bool_)[permut_sbox]
            temp = self.L.copy()
            self.L = np.logical_xor(self.R, output_sboxes)
            self.R = temp
        output = np.append(self.L, self.R)[fp]
        return output

    def compute_iv_from_states(self, leakage_model, stored_states):
        """
        Utility function for getting Iv from states.
        :param leakage_model: Used to retrieve targetted byte
        :param stored_states:
        :return: the intermediate value
        """

        # Here I chose to use 4 or 6 bits for the return value if the states are shorter.
        # This was as this the Sboxes are 6 bits in and 4 bits out
        part = leakage_model["byte"]
        num_bits_part1 = len(stored_states[0]) // 8

        first_state = stored_states[0][num_bits_part1 * part: num_bits_part1 * part + num_bits_part1]
        if num_bits_part1 < 8:
            first_state = np.array(np.append(np.zeros(8 - num_bits_part1),
                                             first_state), dtype=np.bool_)
        first_state = np.packbits(first_state)[0]
        if not leakage_model["leakage_model"] == "HD":
            return first_state

        num_bits_part2 = len(stored_states[1]) // 8
        second_state = stored_states[1][num_bits_part2 * part: num_bits_part2 * part + num_bits_part2]
        if num_bits_part1 < 8:
            second_state = np.array(np.append(np.zeros(8 - num_bits_part2),
                                              second_state), dtype=np.bool_)
        second_state = np.packbits(second_state)[0]
        return first_state ^ second_state

    def encryption_iv_with_state(self, plaintext, leakage_model, state_index, state_index_first, state_index_second):
        """
        Function that returns the IV of the leakage_model.
        :param plaintext: plaintext array
        :param leakage_model: leakage model to be used
        :param state_index: index of state to retrieve
        :param state_index_first: in case of HD leakage model first state to retrieve
        :param state_index_second: in case of HD leakage model second state to retrieve
        :return: intermediate value of given leakage model
        """

        max_states_size = 2 if leakage_model["leakage_model"] == "HD" else 1
        state_indexes = [state_index, state_index_first, state_index_second]
        stored_states = []

        permuted_plaintext = np.array(plaintext[ip], dtype=np.bool_)
        self.L = permuted_plaintext[:32]
        self.R = permuted_plaintext[32:]
        if -1 in state_indexes:
            stored_states.append(np.copy(plaintext))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        if 0 in state_indexes:
            stored_states.append(np.copy(permuted_plaintext))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        for i in range(16):
            expanded_r = self.R[expansion_table]
            if i * 7 + 1 in state_indexes:
                stored_states.append(np.copy(expanded_r))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            to_be_subbed = np.logical_xor(self.KS[i], expanded_r)
            if i * 7 + 2 in state_indexes:
                stored_states.append(np.copy(to_be_subbed))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            output_sboxes = []
            for j in range(8):
                # We need to pad otherwise the 0's are in
                # I know this is a bit of a disaster, sorry
                sub_val = np.packbits(np.append(np.array([0, 0, 0, 0], dtype=np.bool_),
                                                to_be_subbed[j*6+1:j*6+5]))[0]
                if to_be_subbed[j*6]:
                    sub_val += 32
                if to_be_subbed[j*6 + 5]:
                    sub_val += 16
                output_val = sbox[j][sub_val]
                output_sboxes = np.append(output_sboxes, DES.convert_int_to_bool_arr(output_val))
                if i * + 3 in state_indexes:
                    stored_states.append(np.copy(output_sboxes))
                    if len(stored_states) >= max_states_size:
                        return self.compute_iv_from_states(leakage_model, stored_states)
            output_sboxes = np.array(output_sboxes, dtype=np.bool_)[permut_sbox]

            if i * 7 + 4 in state_indexes:
                stored_states.append(np.copy(output_sboxes))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            temp = self.R.copy()
            self.R = np.logical_xor(self.L, output_sboxes)
            if i * 7 + 5 in state_indexes:
                stored_states.append(np.copy(self.R))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)

            self.L = temp
            if i * 7 + 6 in state_indexes:
                stored_states.append(np.copy(self.L))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
        # The order of L and R are swapped to form the pre-output block.
        if 16 * 7 + 1 in state_indexes:
            stored_states.append(np.copy(np.append(self.R, self.L)))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        output = np.append(self.R, self.L)[fp]
        if 16 * 7 + 2 in state_indexes:
            stored_states.append(np.copy(output))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        # If the indices were not encountered something were wrong
        raise ValueError("State index/indices were not encountered during decryption, "
                         "there is probably something wrong with the leakage model")

    def decryption_iv_with_state(self, ciphertext, leakage_model, state_index, state_index_first, state_index_second):
        """
        Function that returns the IV of the leakage_model.
        :param ciphertext: plaintext array
        :param leakage_model: leakage model to be used
        :param state_index: index of state to retrieve
        :param state_index_first: in case of HD leakage model first state to retrieve
        :param state_index_second: in case of HD leakage model second state to retrieve
        :return: intermediate value of given leakage model
        """
        max_states_size = 2 if leakage_model["leakage_model"] == "HD" else 1
        state_indexes = [state_index, state_index_first, state_index_second]
        stored_states = []

        permuted_ciphertext = np.array(ciphertext, dtype=np.bool_)[ip]
        self.L = permuted_ciphertext[32:]
        self.R = permuted_ciphertext[:32]
        for i in range(16):
            expanded_l = self.L[expansion_table]
            if i * 7 + 1 in state_indexes:
                stored_states.append(np.copy(expanded_l))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            to_be_subbed = np.logical_xor(self.KS[15 - i], expanded_l)
            if i * 7 + 2 in state_indexes:
                stored_states.append(np.copy(to_be_subbed))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            output_sboxes = []
            for j in range(8):
                # We need to pad otherwise the 0's are in
                # I know this is a bit of a disaster, sorry
                sub_val = np.packbits(np.append(np.array([0, 0, 0, 0], dtype=np.bool_),
                                                to_be_subbed[j * 6 + 1:j * 6 + 5]))[0]
                if to_be_subbed[j * 6]:
                    sub_val += 32
                if to_be_subbed[j * 6 + 5]:
                    sub_val += 16
                output_val = sbox[j][sub_val]

                output_sboxes = np.append(output_sboxes, DES.convert_int_to_bool_arr(output_val))
                if i * 7 + 3 in state_indexes:
                    stored_states.append(np.copy(output_sboxes))
                    if len(stored_states) >= max_states_size:
                        return self.compute_iv_from_states(leakage_model, stored_states)
            output_sboxes = np.array(output_sboxes, dtype=np.bool_)[permut_sbox]
            if i * 7 + 4 in state_indexes:
                stored_states.append(np.copy(output_sboxes))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            temp = self.L.copy()
            self.L = np.logical_xor(self.R, output_sboxes)
            if i * 7 + 5 in state_indexes:
                stored_states.append(np.copy(self.L))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)

            self.R = temp
            if i * 7 + 6 in state_indexes:
                stored_states.append(np.copy(self.R))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)

        if 16 * 7 + 1 in state_indexes:
            stored_states.append(np.copy(np.append(self.L, self.R)))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)

        output = np.append(self.L, self.R)[fp]

        if 16 * 7 + 2 in state_indexes:
            stored_states.append(np.copy(output))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        raise ValueError("State index/indices were not encountered during decryption, "
                         "there is probably something wrong with the leakage model")

    @staticmethod
    def get_state_index_encryption(lm_round, operation):
        """"
        Returns the state index of certain leakage model.
        :param lm_round: the round to attack.
        :param operation: the operation to attack
        :return: the state index
        """
        if operation == 'Input':
            return -1
        if operation == 'Permuted_Input':
            return 0
        if operation == 'PrePermuted_Output':
            return 16 * 7 + 1
        if operation == 'Output':
            return 16 * 7 + 2
        if operation == 'Expand_R':
            return lm_round * 7 + 1
        if operation == 'AddRoundKey':
            return lm_round * 7 + 2
        if operation == 'Sbox':
            return lm_round * 7 + 3
        if operation == 'Permuted_SBox':
            return lm_round * 7 + 4
        if operation == 'R_out':
            return lm_round * 7 + 5
        if operation == 'L_out':
            return lm_round * 7 + 6

    @staticmethod
    def get_state_index_decryption(lm_round, operation):
        """"
        Returns the state index of certain leakage model.
        :param lm_round: the round to attack.
        :param operation: the operation to attack
        :return: the state index
        """
        if operation == 'Input':
            return -1
        if operation == 'Permuted_Input':
            return 0
        if operation == 'PrePermuted_Output':
            return 16 * 7 + 1
        if operation == 'Output':
            return 16 * 7 + 2
        if operation == 'Expand_L':
            return lm_round * 7 + 1
        if operation == 'AddRoundKey':
            return lm_round * 7 + 2
        if operation == 'Sbox':
            return lm_round * 7 + 3
        if operation == 'Permuted_SBox':
            return lm_round * 7 + 4
        if operation == 'L_out':
            return lm_round * 7 + 5
        if operation == 'R_out':
            return lm_round * 7 + 6

    @staticmethod
    def convert_int_to_bool_arr(val):
        """
        Converts value in range 0-16 into boolean array
        """
        output = np.zeros(4, dtype=np.bool_)
        for i in range(4):
            if val % 2 == 1:
                output[3 - i] = 1
            val = val // 2
        return output

    @staticmethod
    def get_intermediate_values(plaintexts, keys, ciphertexts, leakage_model):
        """
        Function to compute the intermediate values
        :param plaintexts:  the plaintexts
        :param keys: the keys
        :param ciphertexts: the ciphertexts
        :param leakage_model: the leakage model
        :return: list of intermediate values from leakage model.
        """

        intermediate_values = []
        get_state = DES.get_state_index_encryption if leakage_model["direction"] == "Encryption" \
            else DES.get_state_index_decryption

        if leakage_model["leakage_model"] == "HD":
            state_index = -1
            state_index_first = get_state(
                leakage_model['round_first'], leakage_model['target_state_first'])
            state_index_second = get_state(
                leakage_model['round_second'], leakage_model['target_state_second'])
        else:
            state_index = get_state(
                leakage_model['round'], leakage_model['target_state'])
            state_index_first = -1
            state_index_second = -1
        if leakage_model["direction"] == "Encryption" and \
                leakage_model["attack_direction"] == "input":
            bit_keys = np.unpackbits(np.array(keys, dtype=np.ubyte), axis=1)
            bit_plaintexts = np.unpackbits(np.array(plaintexts, dtype=np.ubyte), axis=1)
            des = DES(bit_keys[0])
            prev_key = bit_keys[0]
            for plaintext, key in zip(bit_plaintexts, bit_keys):
                if not np.array_equal(prev_key, key):
                    des.expand_key(key)
                    prev_key = key
                intermediate_values.append(
                    des.encryption_iv_with_state(plaintext, leakage_model, state_index,
                                                 state_index_first, state_index_second))
        else:
            bit_keys = np.unpackbits(np.array(keys, dtype=np.ubyte), axis=1)
            bit_ciphers = np.unpackbits(np.array(ciphertexts, dtype=np.ubyte), axis=1)
            des = DES(bit_keys[0])
            prev_key = bit_keys[0]
            for ciphertext, key in zip(bit_ciphers, bit_keys):
                if not np.array_equal(prev_key, key):
                    des.expand_key(key)
                    prev_key = key
                intermediate_values.append(
                    des.decryption_iv_with_state(ciphertext, leakage_model, state_index,
                                                 state_index_first, state_index_second))

        if leakage_model["leakage_model"] == "HW" or leakage_model["leakage_model"] == "HD":
            return [bin(iv).count("1") for iv in intermediate_values]
        elif leakage_model["leakage_model"] == "bit":
            return [int(bin(iv >> leakage_model["bit"])[len(bin(iv >> leakage_model["bit"])) - 1]) for iv in
                    intermediate_values]
        else:
            return intermediate_values

    @staticmethod
    def get_intermediate_values_multilabel(plaintexts, keys, ciphertexts, leakage_models):
        """
        Function to compute the intermediate values
        :param plaintexts:  the plaintexts
        :param keys: the keys
        :param ciphertexts: the ciphertexts
        :param leakage_models: the leakage models
        :return: list of intermediate values from leakage model of each leakage models
        """
        labels = []
        for leakage_model in leakage_models:
            labels.append(DES.get_intermediate_values(plaintexts, keys, ciphertexts, leakage_model))

        return labels
