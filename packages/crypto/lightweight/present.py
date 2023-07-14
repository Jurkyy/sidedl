import numpy as np


s_box = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]

inv_s_box = [0x5, 0xE, 0xF, 0x8, 0xC, 0x1, 0x2, 0xD, 0xB, 0x4, 0x6, 0x3, 0x0, 0x7, 0x9, 0xA]

p_layer_order = [0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21, 37, 53, 6,
                 22, 38, 54, 7, 23, 39, 55, 8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59, 12,
                 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63]

p_layer_order_inv = [p_layer_order.index(x) for x in range(64)]


def convert_int_to_bool_arr(val, length):
    """
    Converts value in range 0-2^length into boolean array
    """
    output = np.zeros(length, dtype=np.bool_)
    for i in range(length):
        if val % 2 == 1:
            output[length - i - 1] = 1
        val = val // 2
    return output


class Present:

    def __init__(self, key):
        """
        Initialization of present cipher with key.
        :param key: Key  for initialization
        """
        self.key_arr = self.expand_key(np.unpackbits(key))

    def expand_key(self, key):
        """
        Method for expanding the key
        :param key: key to expand
        :return: Array containing round keys
        """
        if len(key) == 128:
            return self.expand_key_128(key)
        round_keys = np.zeros((32, 64), dtype=np.bool_)
        key_register = np.array(key, dtype=np.bool_)
        for i in range(1, 33):
            round_keys[i-1] = np.array(key_register[:64], dtype=np.bool_)
            key_register = np.roll(key_register, -61)
            round_counter = i
            key_register[0:4] = self.sbox_word(key_register[0:4])
            key_register[-20:-15] = np.logical_xor(key_register[-20:-15],
                                                   convert_int_to_bool_arr(round_counter, 5))

        return round_keys

    def expand_key_128(self, key):
        """
        Method for expanding the key of length 128
        :param key: key to expand
        :return: Array containing round keys
        """
        round_keys = np.array((32, 64), dtype=np.bool_)
        key_register = key
        for i in range(1, 33):
            round_keys[i-1] = key_register[:64]
            key_register = np.roll(key_register, 66)
            round_counter = i
            key_register[:4] = self.sbox_word(key_register[:4])
            key_register[4:8] = self.sbox_word(key_register[4:8])
            key_register[-67:-62] = np.logical_xor(key_register[-67:-62],
                                                   convert_int_to_bool_arr(round_counter, 5))
        return round_keys

    def sbox_word(self, word):
        """
        Applies sbox to a 4-bit word
        :param word: boolean array of the word
        :return: boolean array of substituted word
        """
        temp_val = s_box[int(np.packbits(word))//16]
        return convert_int_to_bool_arr(temp_val, 4)

    def inv_sbox_word(self, word):
        """
        Applies inverted sbox to a 4-bit word
        :param word: boolean array of the word
        :return: boolean array of substituted word
        """
        temp_val = inv_s_box[int(np.packbits(word)//16)]
        return convert_int_to_bool_arr(temp_val, 4)

    def sbox_state(self, state):
        """
        Utility function for applying sbox
        """
        new_state = np.array(state, dtype=np.bool_)
        for j in range(16):
            new_state[j*4:(j+1)*4] = self.sbox_word(state[j*4:(j+1)*4])
        return new_state

    def inv_sbox_state(self, state):
        """
        Utility function for applying inverse sbox
        """
        new_state = np.array(state, dtype=np.bool_)
        for j in range(16):
            new_state[j*4:(j+1)*4] = self.inv_sbox_word(state[j*4:(j+1)*4])
        return new_state

    def player_state(self, state):
        """
        Utility function for applying permutation
        """
        new_state = np.array(state, dtype=np.bool_)
        for i in range(64):
            new_state[63 - p_layer_order[i]] = state[63-i]
        return new_state

    def inv_player_state(self, state):
        """
        Utility function for applying inverse permutation
        """
        new_state = np.array(state, dtype=np.bool_)
        for i in range(64):
            new_state[63 - p_layer_order_inv[i]] = state[63-i]
        return new_state

    def encryption(self, plaintext):
        """
        Encryption function
        :param plaintext: plaintext
        :return: encrypted result
        """
        state = np.unpackbits(plaintext)
        print(len(state))
        for i in range(31):
            # AddroundKey
            state = np.logical_xor(state, self.key_arr[i])
            # Sbox
            state = self.sbox_state(state)
            # permute
            state = self.player_state(state)
        # AddroundKey
        state = np.logical_xor(state, self.key_arr[31])
        return np.packbits(state)

    @staticmethod
    def get_state_index_encryption(lm_round, operation, num_rounds):
        """"
        Returns the state index of certain leakage model.
        :param lm_round: the round to attack.
        :param operation: the operation to attack
        :param num_rounds: the maximum number of rounds
        :return: the state index
        """
        if lm_round == 0 and operation == 'Input':
            return 0
        if operation == 'Output':
            return 32 * 3 + 1
        else:
            if operation == 'AddRoundKey':
                return ((lm_round-1) * 3) + 1
            if operation == 'Sbox':
                return ((lm_round-1) * 3) + 2
            if operation == 'Permutation':
                return ((lm_round-1) * 3) + 3
        raise ValueError("Leakage model is not compatible with cipher")

    @staticmethod
    def get_state_index_decryption(lm_round, operation, num_rounds=32):
        """"
        Returns the state index of certain leakage model.
        :param lm_round: the round to attack.
        :param operation: the operation to attack
        :param num_rounds: the maximum number of rounds
        :return: the state index
        """
        if lm_round == 0 and operation == 'Input':
            return 0
        if operation == 'Output':
            return 32 * 3 + 1
        else:
            if operation == 'AddRoundKey':
                return (abs(lm_round-32) * 3) + 1
            if operation == 'Sbox':
                return (abs(lm_round-32) * 3) + 3
            if operation == 'Permutation':
                return (abs(lm_round-32) * 3) + 2
        raise ValueError("Leakage model is not compatible with cipher")

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
        num_bits_part1 = 4

        first_state = stored_states[0][num_bits_part1 * part: num_bits_part1 * part + num_bits_part1]
        if num_bits_part1 < 8:
            first_state = np.array(np.append(np.zeros(8 - num_bits_part1),
                                             first_state), dtype=np.bool_)
        first_state = np.packbits(first_state)[0]
        if not leakage_model["leakage_model"] == "HD":
            return first_state

        num_bits_part2 = 4
        second_state = stored_states[1][num_bits_part2 * part: num_bits_part2 * part + num_bits_part2]
        if num_bits_part1 < 8:
            second_state = np.array(np.append(np.zeros(8 - num_bits_part2),
                                              second_state), dtype=np.bool_)
        second_state = np.packbits(second_state)[0]
        return first_state ^ second_state

    def encryption_iv_with_state(self, plain, leakage_model, state_index, state_index_first, state_index_second):
        """
        Function that returns the IV of the leakage_model.
        :param plain: plaintext
        :param leakage_model: leakage model to be used
        :param state_index: index of state to retrieve
        :param state_index_first: in case of HD leakage model first state to retrieve
        :param state_index_second: in case of HD leakage model second state to retrieve
        :return: intermediate value of given leakage model
        """

        max_states_size = 2 if leakage_model["leakage_model"] == "HD" else 1
        state_indexes = [state_index, state_index_first, state_index_second]
        stored_states = []
        state = np.unpackbits(plain)
        print(len(state))
        if 0 in state_indexes:
            stored_states.append(np.copy(state))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)

        for i in range(31):
            # AddroundKey
            state = np.logical_xor(state, self.key_arr[i])
            if i*3 + 1 in state_indexes:
                stored_states.append(np.copy(state))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            # Sbox
            state = self.sbox_state(state)
            if i * 3 + 1 in state_indexes:
                stored_states.append(np.copy(state))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            # permute
            state = self.player_state(state)
            if i * 3 + 1 in state_indexes:
                stored_states.append(np.copy(state))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
        # AddroundKey
        state = np.logical_xor(state, self.key_arr[31])
        if 32 * 3 + 1 in state_indexes:
            stored_states.append(np.copy(state))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        # If the indices were not encountered something were wrong
        raise ValueError("State index/indices were not encountered during decryption, "
                         "there is probably something wrong with the leakage model")


    def decryption_iv_with_state(self, cipher, leakage_model, state_index, state_index_first, state_index_second):
        """
        Function that returns the IV of the leakage_model.
        :param cipher: ciphertext
        :param leakage_model: leakage model to be used
        :param state_index: index of state to retrieve
        :param state_index_first: in case of HD leakage model first state to retrieve
        :param state_index_second: in case of HD leakage model second state to retrieve
        :return: intermediate value of given leakage model
        """

        max_states_size = 2 if leakage_model["leakage_model"] == "HD" else 1
        state_indexes = [state_index, state_index_first, state_index_second]
        stored_states = []
        state = np.unpackbits(cipher)
        print(len(state))
        if 0 in state_indexes:
            stored_states.append(np.copy(state))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)

        for i in range(31, 0, -1):
            # AddroundKey
            state = np.logical_xor(state, self.key_arr[i])
            if abs(32 - (i + 1)) * 3 + 1 in state_indexes:
                stored_states.append(np.copy(state))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            state = self.inv_player_state(state)
            if abs(32 - (i + 1)) * 3 + 2 in state_indexes:
                stored_states.append(np.copy(state))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            state = self.inv_sbox_state(state)
            if abs(32 - (i + 1)) * 3 + 3 in state_indexes:
                stored_states.append(np.copy(state))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
        # AddroundKey
        state = np.logical_xor(state, self.key_arr[0])
        if 32 * 3 + 1 in state_indexes:
            stored_states.append(np.copy(state))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        # If the indices were not encountered something were wrong
        raise ValueError("State index/indices were not encountered during decryption, "
                         "there is probably something wrong with the leakage model")


    def decryption(self, ciphertext):
        """
        Decryption function
        :param ciphertext: ciphertext
        :return: decrypted result
        """
        state = np.unpackbits(ciphertext)
        print(len(state))
        for i in range(31, 0, -1):
            # AddroundKey
            state = np.logical_xor(state, self.key_arr[i])
            # Sbox
            state = self.inv_player_state(state)
            # permute
            state = self.inv_sbox_state(state)
        # AddroundKey
        state = np.logical_xor(state, self.key_arr[0])
        return np.packbits(state)


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
        get_state = Present.get_state_index_encryption if leakage_model["direction"] == "Encryption" \
            else Present.get_state_index_decryption
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
        if leakage_model["attack_direction"] == "input":
            keys = np.array(keys, dtype=np.uint8)
            plaintexts = np.array(plaintexts, dtype=np.uint8)
            pres = Present(keys[0])
            prev_key = keys[0]
            for plaintext, key in zip(plaintexts, keys):
                if not np.array_equal(prev_key, key):
                    pres.expand_key(key)
                    prev_key = key
                intermediate_values.append(
                    pres.encryption_iv_with_state(list(plaintext), leakage_model, state_index,
                                                  state_index_first, state_index_second))
        else:
            keys = np.array(keys, dtype=np.uint8)
            ciphertexts = np.array(ciphertexts, dtype=np.uint8)
            pres = Present(keys[0])
            prev_key = keys[0]
            for ciphertext, key in zip(ciphertexts, keys):
                if not np.array_equal(prev_key, key):
                    pres.expand_key(key)
                    prev_key = key
                intermediate_values.append(
                    pres.decryption_iv_with_state(list(ciphertext), leakage_model, state_index,
                                                  state_index_first, state_index_second))

        if leakage_model["leakage_model"] == "HW" or leakage_model["leakage_model"] == "HD":
            return [bin(iv).count("1") for iv in intermediate_values]
        elif leakage_model["leakage_model"] == "bit":
            return [int(bin(iv >> (leakage_model["bit"]))[len(bin(iv >> (leakage_model["bit"]))) - 1]) for iv in
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
            labels.append(Present.get_intermediate_values(plaintexts, keys, ciphertexts, leakage_model))

        return labels
