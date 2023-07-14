import numpy as np


class Speck:

    def __init__(self, key):
        """
        Initializes Speck object with key
        :param key: array with size 128/192/256 key array
        """

        unpacked_key = np.unpackbits(key)
        self.round_keys = np.zeros((32, 64), np.bool_)
        self.expand_key(unpacked_key)

    @staticmethod
    def ror(x, r):
        """
        Rotates array x to the right by r
        """
        return np.roll(x, r)

    @staticmethod
    def rol(x, r):
        """
        Rotates array x to the left by r
        """
        return np.roll(x, -r)

    @staticmethod
    def r(x, y, k):
        x = Speck.ror(x, 8)
        x = Speck.boolean_array_add(x, y)
        x = np.logical_xor(x, k)
        y = Speck.rol(y, 3)
        y = np.logical_xor(x, y)
        return x, y

    @staticmethod
    def boolean_array_add(x, y):
        """
        Addition function for boolean array arithmetic
        :return: the resulting boolean array
        """
        temp = (Speck.boolean_array_to_int(x) + Speck.boolean_array_to_int(y)) & pow(2, len(x))-1
        return Speck.int_to_boolean_array(temp, len(x))

    @staticmethod
    def boolean_array_sub(x, y):
        """
        Substraction function for boolean array arithmetic
        :return: the resulting boolean array
        """

        temp = (Speck.boolean_array_to_int(x) - Speck.boolean_array_to_int(y)) & pow(2, len(x))-1
        return Speck.int_to_boolean_array(temp, len(x))

    @staticmethod
    def int_to_boolean_array(x, length):
        """
        Helper function that transforms an integer to a boolean array of length length
        :param x: integer
        :param length: the length of the array
        :return: the boolean array
        """
        temp = bin(x)
        result = np.zeros(length, dtype=np.bool_)
        for i in range(min(len(temp)-2, length)):
            result[length - i - 1] = temp[len(temp) - i - 1] == '1'

        return result

    @staticmethod
    def boolean_array_to_int(x):
        """
        Helper function that transforms boolean array to an integer
        :param x: the boolean array
        :return: the integer
        """
        result = 0
        for i in range(len(x)):
            if x[-(i+1)]:
                result += pow(2, i)
        return result

    def expand_key(self, key):
        """
        Key expansion for Speck 128
        :param key: base key
        :return:
        """
        if len(key) == 192:
            self.expand_key192(key)
            return
        if len(key) == 256:
            self.expand_key256(key)
            return
        a = key[:64]
        b = key[64:]
        self.round_keys[0] = a
        for i in range(31):

            b, a = Speck.r(b, a, Speck.int_to_boolean_array(i, len(b)))
            self.round_keys[i + 1] = a

    def expand_key192(self, key):
        """
        Key expansion for Speck 128/192
        :param key: base key
        :return:
        """
        a = key[:64]
        b = key[64:128]
        c = key[128:]
        self.round_keys = np.zeros((33, 64), np.bool_)
        self.round_keys[0] = a
        i = 1
        while i < 33:
            b, a = Speck.r(b, a, Speck.int_to_boolean_array(i-1, 64))
            self.round_keys[i] = a
            i += 1
            c, a = Speck.r(c, a, Speck.int_to_boolean_array(i-1, 64))
            self.round_keys[i] = a
            i += 1

    def expand_key256(self, key):
        """
        Key expansion for Speck 128/256
        :param key: base key
        :return:
        """
        a = key[:64]
        b = key[64:128]
        c = key[128:192]
        d = key[192:]
        self.round_keys = np.zeros((34, 64), np.bool_)
        self.round_keys[0] = a
        i = 1
        while i < 34:
            b, a = Speck.r(b, a, Speck.int_to_boolean_array(i-1, 64))
            self.round_keys[i] = a
            i += 1
            c, a = Speck.r(c, a, Speck.int_to_boolean_array(i-1, 64))
            self.round_keys[i] = a
            i += 1
            d, a = Speck.r(d, a, Speck.int_to_boolean_array(i-1, 64))
            self.round_keys[i] = a
            i += 1

    def decryption(self, plaintext):
        """
        Performs encryption operation
        :param plaintext: array of 1/0's or booleans
        :return:
        """
        y = np.unpackbits(plaintext)[64:]
        x = np.unpackbits(plaintext)[:64]

        y = np.logical_xor(x, y)
        y = Speck.ror(y, 3)
        x = np.logical_xor(x, self.round_keys[len(self.round_keys)-1])
        print(len(self.round_keys)-1)
        x = Speck.boolean_array_sub(x, y)
        x = Speck.rol(x, 8)
        for i in range(len(self.round_keys)-1):

            y = np.logical_xor(x, y)
            y = Speck.ror(y, 3)
            x = np.logical_xor(x, self.round_keys[len(self.round_keys)-2-i])

            x = Speck.boolean_array_sub(x, y)
            x = Speck.rol(x, 8)

        return np.packbits(np.append(x, y))

    def encryption(self, plaintext):
        """
            Performs encryption operation
            :param plaintext: array of 1/0's or booleans
            :return:
            """
        x = np.unpackbits(plaintext)[64:]
        y = np.unpackbits(plaintext)[:64]
        x = Speck.ror(x, 8)
        x = Speck.boolean_array_add(x, y)
        x = np.logical_xor(x, self.round_keys[0])
        y = Speck.rol(y, 3)
        y = np.logical_xor(x, y)

        for i in range(len(self.round_keys)-1):
            x = Speck.ror(x, 8)
            x = Speck.boolean_array_add(x, y)
            x = np.logical_xor(x, self.round_keys[i + 1])
            y = Speck.rol(y, 3)
            y = np.logical_xor(x, y)

        return np.packbits(np.append(x, y))

    def compute_iv_from_states(self, leakage_model, stored_states):
        """
        Utility function for getting Iv from states.
        :param leakage_model: Used to retrieve targetted byte
        :param stored_states:
        :return: the intermediate value
        """
        byte = leakage_model["byte"]
        if leakage_model["leakage_model"] == "HD":
            return Speck.boolean_array_to_int(stored_states[0][byte * 8: (byte+1)*8]) \
                   ^ Speck.boolean_array_to_int(stored_states[1][byte * 8: (byte+1)*8])
        return Speck.boolean_array_to_int(stored_states[0][byte * 8: (byte+1)*8])

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

        y = np.unpackbits(plaintext)[64:]
        x = np.unpackbits(plaintext)[:64]
        if -1 in state_indexes:
            stored_states.append(np.copy(np.append(x,  y)))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        x = Speck.ror(x, 8)
        if 0 in state_indexes:
            stored_states.append(np.copy(np.append(x,  y)))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        x = Speck.boolean_array_add(x, y)
        if 1 in state_indexes:
            stored_states.append(np.copy(np.append(x,  y)))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        x = np.logical_xor(x, self.round_keys[0])
        if 2 in state_indexes:
            stored_states.append(np.copy(np.append(x,  y)))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        y = Speck.rol(y, 3)
        if 3 in state_indexes:
            stored_states.append(np.copy(np.append(x,  y)))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        y = np.logical_xor(x, y)
        if 4 in state_indexes:
            stored_states.append(np.copy(np.append(x,  y)))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        for i in range(len(self.round_keys)-1):
            x = Speck.ror(x, 8)
            if (i+1) * 5 in state_indexes:
                stored_states.append(np.copy(np.append(x,  y)))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            x = Speck.boolean_array_add(x, y)
            if (i+1) * 5 + 1 in state_indexes:
                stored_states.append(np.copy(np.append(x,  y)))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            x = np.logical_xor(x, self.round_keys[i + 1])
            if (i + 1) * 5 + 2 in state_indexes:
                stored_states.append(np.copy(np.append(x,  y)))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            y = Speck.rol(y, 3)
            if (i + 1) * 5 + 3 in state_indexes:
                stored_states.append(np.copy(np.append(x,  y)))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            y = np.logical_xor(x, y)
            if (i+1) * 5 + 4 in state_indexes or -2 in state_indexes:
                stored_states.append(np.copy(np.append(x,  y)))
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
        y = np.unpackbits(ciphertext)[64:]
        x = np.unpackbits(ciphertext)[:64]
        if -1 in state_indexes:
            stored_states.append(np.copy(np.append(x, y)))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        y = np.logical_xor(x, y)
        if 0 in state_indexes:
            stored_states.append(np.copy(np.append(x, y)))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        y = Speck.ror(y, 3)
        if 1 in state_indexes:
            stored_states.append(np.copy(np.append(x, y)))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        x = np.logical_xor(x, self.round_keys[len(self.round_keys)-1])
        if 2 in state_indexes:
            stored_states.append(np.copy(np.append(x, y)))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        x = Speck.boolean_array_sub(x, y)
        if 3 in state_indexes:
            stored_states.append(np.copy(np.append(x, y)))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        x = Speck.rol(x, 8)
        if 4 in state_indexes:
            stored_states.append(np.copy(np.append(x, y)))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        for i in range(len(self.round_keys)-1):
            y = np.logical_xor(x, y)
            if (i+1) * 5 in state_indexes:
                stored_states.append(np.copy(np.append(x,  y)))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            y = Speck.ror(y, 3)
            if (i+1) * 5 + 1 in state_indexes:
                stored_states.append(np.copy(np.append(x,  y)))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            x = np.logical_xor(x, self.round_keys[len(self.round_keys)-2-i])
            if (i+1) * 5 + 2 in state_indexes:
                stored_states.append(np.copy(np.append(x,  y)))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            x = Speck.boolean_array_sub(x, y)
            if (i+1) * 5 + 3 in state_indexes:
                stored_states.append(np.copy(np.append(x,  y)))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            x = Speck.rol(x, 8)
            if (i+1) * 5 + 4 in state_indexes or -2 in state_indexes:
                stored_states.append(np.copy(np.append(x,  y)))
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
        if operation == 'RotateRight':
            return lm_round * 5 + 0
        if operation == 'AddY':
            return lm_round * 5 + 1
        if operation == 'AddRoundKey':
            return lm_round * 5 + 2
        if operation == 'RotateLeft':
            return lm_round * 5 + 3
        if operation == 'AddX':
            return lm_round * 5 + 4

        if operation == 'Output':
            return -2

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
        if operation == 'RotateRight':
            return lm_round * 5 + 4
        if operation == 'AddY':
            return lm_round * 5 + 3
        if operation == 'AddRoundKey':
            return lm_round * 5 + 2
        if operation == 'RotateLeft':
            return lm_round * 5 +1
        if operation == 'AddX':
            return lm_round * 5

        if operation == 'Output':
            return -2


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
        get_state = Speck.get_state_index_encryption if leakage_model["direction"] == "Encryption" \
            else Speck.get_state_index_decryption

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
            bit_keys = np.array(keys, dtype=np.ubyte)
            bit_plaintexts = np.array(plaintexts, dtype=np.ubyte)
            speck = Speck(bit_keys[0])
            prev_key = bit_keys[0]
            for plaintext, key in zip(bit_plaintexts, bit_keys):
                if not np.array_equal(prev_key, key):
                    speck.expand_key(key)
                    prev_key = key
                intermediate_values.append(
                    speck.encryption_iv_with_state(plaintext, leakage_model, state_index,
                                                 state_index_first, state_index_second))
        else:
            bit_keys = np.array(keys, dtype=np.ubyte)
            bit_ciphers = np.array(ciphertexts, dtype=np.ubyte)
            speck = Speck(bit_keys[0])
            prev_key = bit_keys[0]
            for ciphertext, key in zip(bit_ciphers, bit_keys):
                if not np.array_equal(prev_key, key):
                    speck.expand_key(key)
                    prev_key = key
                intermediate_values.append(
                    speck.decryption_iv_with_state(ciphertext, leakage_model, state_index,
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
            labels.append(Speck.get_intermediate_values(plaintexts, keys, ciphertexts, leakage_model))

        return labels

def runtest():

    keys = bytearray.fromhex(' 0706050403020100 0f0e0d0c0b0a0908')
    plaintexts= bytearray.fromhex(' 7469206564616d20 6c61766975716520')
    speck = Speck(keys)
    print(speck.encryption(plaintexts))
runtest()