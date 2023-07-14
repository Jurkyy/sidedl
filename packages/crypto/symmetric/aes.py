import numpy as np

s_box = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
])

inv_s_box = np.array([
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
])

r_con = (
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
    0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A,
    0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A,
    0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39,
)


def xtime(a: int):
    """ Code taken from https://github.com/bozhu/AES-Python/blob/master/aes.py, method used for mixcolumns.
    :param a: index to shift.
    :return: index to shift to.
    """
    return (((a << 1) ^ 0x1B) & 0xFF) if (a & 0x80) else (a << 1)


def mix_single_column(a):
    # please see Sec 4.1.2 in The Design of Rijndael
    t = a[0] ^ a[1] ^ a[2] ^ a[3]
    u = a[0]
    a[0] ^= t ^ xtime(a[0] ^ a[1])
    a[1] ^= t ^ xtime(a[1] ^ a[2])
    a[2] ^= t ^ xtime(a[2] ^ a[3])
    a[3] ^= t ^ xtime(a[3] ^ u)


class AES:

    def __init__(self, key: np.ndarray):
        """ Constructor for AES class.
        :param key: the key to use in encryption.
        """
        key_len = len(key)//4
        self.num_rounds = 10 if key_len == 4 else (12 if key_len == 6 else 14)
        print(self.num_rounds)
        self.keys = np.zeros((4 * self.num_rounds, 4), dtype=int)
        self.expand_key(key)
        self.state = np.zeros((4, 4), dtype=int)

    def expand_key(self, key: np.ndarray):
        """ Takes Key and expands it as described in section 3.6 of Design of Rijndael.
        :param key: the key to use in encryption.
        """

        key_len = len(key)//4
        num_rounds = 10 if key_len == 4 else (12 if key_len == 6 else 14)

        self.keys = np.zeros((4 * (num_rounds+1), 4), dtype=int)
        for i in range(key_len):
            for j in range(4):
                self.keys[i][j] = key[i * 4 + j]

        for i in range(key_len, (num_rounds+1) * 4):
            if i % key_len == 0:
                self.keys[i][0] = (self.keys[i - key_len][0] ^ s_box[self.keys[i - 1][1]]) ^ r_con[i // key_len]
                for j in range(1, 4):
                    self.keys[i][j] = self.keys[i - key_len][j] ^ s_box[self.keys[i - 1][(j + 1) % 4]]
            elif key_len > 6 and i % key_len == 4:
                for j in range(4):
                    self.keys[i][j] = self.keys[i - key_len][j] ^ s_box[self.keys[i - 1][j]]
            else:
                for j in range(4):
                    self.keys[i][j] = self.keys[i - key_len][j] ^ self.keys[i - 1][j]

    def decryption(self, cipher: np.ndarray):
        """ Decrypts given cipher text with the current key.
         Round is round for which to return result.
         Operation is operation on which to return result.
         Result byte is byte to return 0-15.
         If single byte is set to true returns single byte.
         :param cipher: cipher text to use.
         """

        self.state = np.zeros((4, 4), dtype=int)
        for i in range(4):
            for j in range(4):
                self.state[i][j] = cipher[i * 4 + j]

        self.add_round_key(self.num_rounds)
        self.inv_shift_rows()
        self.inv_subst()

        for i in range(self.num_rounds - 1, 0, -1):
            self.add_round_key(i)
            self.inv_mix_columns()
            self.inv_shift_rows()
            self.inv_subst()

        self.add_round_key(0)

        output = np.zeros(16, dtype=int)
        for i in range(4):
            for j in range(4):
                output[i * 4 + j] = self.state[i][j]
        return output

    def encryption(self, plain: np.ndarray):
        """ Encrypts given plaintext with the current key.
         :param plain: the plaintexts to use.
         :return: encrypted plaintext.
         """

        self.state = np.zeros((4, 4), dtype=int)

        for i in range(4):
            for j in range(4):
                self.state[i][j] = plain[i * 4 + j]

        self.add_round_key(0)
        for i in range(1, self.num_rounds):
            self.subst()
            self.shift_rows()
            self.mix_columns()
            self.add_round_key(i)

        self.subst()
        self.shift_rows()
        self.add_round_key(self.num_rounds)
        output = np.zeros(16, dtype=int)
        for i in range(4):
            for j in range(4):
                output[i * 4 + j] = self.state[i][j]
        return output

    def compute_iv_from_states(self, leakage_model, stored_states):
        """
        Utility function for getting Iv from states.
        :param leakage_model: Used to retrieve targetted byte
        :param stored_states:
        :return: the intermediate value
        """
        row = leakage_model["byte"] // 4
        column = leakage_model["byte"] % 4
        if leakage_model["leakage_model"] == "HD":
            return stored_states[0][row][column] ^ stored_states[1][row][column]
        return stored_states[0][row][column]

    def decryption_iv_with_state(self, cipher, leakage_model, state_index, state_index_first, state_index_second):
        """
        Function that returns the IV of the leakage_model.
        :param cipher: ciphertext
        :param leakage_model: leakage model to be used
        :param state_index: index of state to retrieve
        :param state_index_first: in case of HD leakage model first state to retrieve
        :param state_index_second: in case of HD leakage model second state to retrieve
        :return: intermediate value
        """
        self.state = np.zeros((4, 4), dtype=int)

        # If HD leakage we neeed to store 2 states
        max_states_size = 2 if leakage_model["leakage_model"] == "HD" else 1
        state_indexes = [state_index, state_index_first, state_index_second]
        stored_states = []
        self.state = np.zeros((4, 4), dtype=int)
        for i in range(4):
            for j in range(4):
                self.state[i][j] = cipher[i * 4 + j]

        # At every state check whether this state should be stored and whether we are done.
        if 0 in state_indexes:
            stored_states.append(np.copy(self.state))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        self.add_round_key(self.num_rounds)
        if 1 in state_indexes:
            stored_states.append(np.copy(self.state))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        self.inv_shift_rows()
        if 2 in state_indexes:
            stored_states.append(np.copy(self.state))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        self.inv_subst()
        if 3 in state_indexes:
            stored_states.append(np.copy(self.state))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)

        for i in range(self.num_rounds - 1, 0, -1):

            self.add_round_key(i)
            if (self.num_rounds - i - 1) * 4 + 4 in state_indexes:
                stored_states.append(np.copy(self.state))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            self.inv_mix_columns()
            if (self.num_rounds - i - 1) * 4 + 5 in state_indexes:
                stored_states.append(np.copy(self.state))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            self.inv_shift_rows()
            if (self.num_rounds - i - 1) * 4 + 6 in state_indexes:
                stored_states.append(np.copy(self.state))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            self.inv_subst()
            if (self.num_rounds - i - 1) * 4 + 7 in state_indexes:
                stored_states.append(np.copy(self.state))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)

        self.add_round_key(0)
        if (self.num_rounds - 1) * 4 + 4 in state_indexes:
            stored_states.append(np.copy(self.state))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)

        # If the indices were not encountered something were wrong
        raise ValueError("State index/indices were not encountered during decryption, "
                         "there is probably something wrong with the leakage model")

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

        self.state = np.zeros((4, 4), dtype=int)
        max_states_size = 2 if leakage_model["leakage_model"] == "HD" else 1
        state_indexes = [state_index, state_index_first, state_index_second]
        stored_states = []

        for i in range(4):
            for j in range(4):
                self.state[i][j] = plain[i * 4 + j]
        # At every state check whether this state should be stored and whether we are done.
        if 0 in state_indexes:
            stored_states.append(np.copy(self.state))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        self.add_round_key(0)
        if 1 in state_indexes:
            stored_states.append(np.copy(self.state))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        for i in range(1, self.num_rounds):
            self.subst()

            if 2 + (i-1)*4 in state_indexes:
                stored_states.append(np.copy(self.state))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            self.shift_rows()

            if 3 + (i-1)*4 in state_indexes:
                stored_states.append(np.copy(self.state))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            self.mix_columns()
            if 4 + (i-1)*4 in state_indexes:
                stored_states.append(np.copy(self.state))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)
            self.add_round_key(i)
            if 5 + (i-1)*4 in state_indexes:
                stored_states.append(np.copy(self.state))
                if len(stored_states) >= max_states_size:
                    return self.compute_iv_from_states(leakage_model, stored_states)

        self.subst()
        if 6 + (self.num_rounds-1) * 4 in state_indexes:
            stored_states.append(np.copy(self.state))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        self.shift_rows()
        if 7 + (self.num_rounds-1) * 4 in state_indexes:
            stored_states.append(np.copy(self.state))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)
        self.add_round_key(self.num_rounds)
        if 8 + (self.num_rounds-1) * 4 in state_indexes:
            stored_states.append(np.copy(self.state))
            if len(stored_states) >= max_states_size:
                return self.compute_iv_from_states(leakage_model, stored_states)

        # If the indices were not encountered something were wrong
        raise ValueError("State index/indices were not encountered during decryption, "
                         "there is probably something wrong with the leakage model")

    def add_round_key(self, r: int):
        """ Adds round key for round r to state array.
        :param r: round to add key to
        """

        for i in range(4):
            for j in range(4):
                self.state[i][j] = self.state[i][j] ^ self.keys[i + 4 * r][j]

    def subst(self):
        """ Does substitution phase of AES."""

        for i in range(4):
            for j in range(4):
                self.state[i][j] = s_box[self.state[i][j]]

    def inv_subst(self):
        """ Does inverse of substitution phase of AES."""

        for i in range(4):
            for j in range(4):
                self.state[i][j] = inv_s_box[self.state[i][j]]

    def shift_rows(self):
        self.state[0][1], self.state[1][1], self.state[2][1], self.state[3][1] = \
            self.state[1][1], self.state[2][1], self.state[3][1], self.state[0][1]
        self.state[0][2], self.state[1][2], self.state[2][2], self.state[3][2] = \
            self.state[2][2], self.state[3][2], self.state[0][2], self.state[1][2]
        self.state[0][3], self.state[1][3], self.state[2][3], self.state[3][3] \
            = self.state[3][3], self.state[0][3], self.state[1][3], self.state[2][3]

    def inv_shift_rows(self):
        self.state[0][1], self.state[1][1], self.state[2][1], self.state[3][1] = \
            self.state[3][1], self.state[0][1], self.state[1][1], self.state[2][1]
        self.state[0][2], self.state[1][2], self.state[2][2], self.state[3][2] = \
            self.state[2][2], self.state[3][2], self.state[0][2], self.state[1][2]
        self.state[0][3], self.state[1][3], self.state[2][3], self.state[3][3] = \
            self.state[1][3], self.state[2][3], self.state[3][3], self.state[0][3]

    def mix_columns(self):
        for i in range(4):
            mix_single_column(self.state[i])

    def inv_mix_columns(self):
        # see Sec 4.1.3 in The Design of Rijndael
        for i in range(4):
            u = xtime(xtime(self.state[i][0] ^ self.state[i][2]))
            v = xtime(xtime(self.state[i][1] ^ self.state[i][3]))
            self.state[i][0] ^= u
            self.state[i][1] ^= v
            self.state[i][2] ^= u
            self.state[i][3] ^= v

        self.mix_columns()

    @staticmethod
    def encrypt_block_leakage_model_from_input_fast(plaintexts, keys, leakage_model):
        """
        If AES leakage model is HW or ID of S-Box output in the first encryption round, the fast method is called.
        :param plaintexts: 2D array containing the plaintexts for all traces
        :param keys: 2D array containing the keys for all traces
        :param leakage_model: leakage model dictionary
        :return: intermediates
        """

        plaintext = [row[leakage_model["byte"]] for row in plaintexts]
        key = [row[leakage_model["byte"]] for row in keys]
        state = [int(p) ^ int(k) for p, k in zip(np.asarray(plaintext[:]), np.asarray(key[:]))]

        return s_box[state]

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
        if lm_round == 0 and operation == 'AddRoundKey':
            return 1
        if lm_round < num_rounds:
            if operation == 'Sbox':
                return (lm_round - 1)*4 + 2
            if operation == 'ShiftRows':
                return (lm_round - 1)*4 + 3
            if operation == 'MixColumns':
                return (lm_round - 1)*4 + 4
            if operation == 'AddRoundKey':
                return (lm_round - 1)*4 + 5
        if lm_round == num_rounds:
            if operation == 'Sbox':
                return (lm_round - 1)*4 + 2
            if operation == 'ShiftRows':
                return (lm_round - 1)*4 + 3
            if operation == 'AddRoundKey':
                return (lm_round - 1)*4 + 5
        raise ValueError("Leakage model is not compatible with cipher")

    @staticmethod
    def get_state_index_decryption(lm_round, operation, num_rounds):
        """"
        Returns the state index of certain leakage model.
        :param lm_round: the round to attack.
        :param operation: the operation to attack
        :param num_rounds: the maximum number of rounds
        :return: the state index
        """
        if lm_round == 1 and operation == 'Input':
            return 0
        if lm_round == 1 and operation == 'AddRoundKey':
            return 1
        if lm_round == 1 and operation == 'InvShiftRows':
            return 2
        if lm_round == 1 and operation == 'InvSbox':
            return 3
        if lm_round < num_rounds:
            if operation == 'AddRoundKey':
                return (lm_round - 2) * 4 + 4
            if operation == 'InvMixColumns':
                return (lm_round - 2) * 4 + 5
            if operation == 'InvShiftRows':
                return (lm_round - 2) * 4 + 6
            if operation == 'InvSbox':
                return (lm_round - 2) * 4 + 7
        if lm_round == num_rounds:
            if operation == 'AddRoundKey':
                return (lm_round - 2) * 4 + 4
        raise ValueError("Leakage model is not compatible with cipher")

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

        if leakage_model["attack_direction"] == "input" and leakage_model["direction"] == "Encryption" and \
                leakage_model["target_state"] == "Sbox" and leakage_model["round"] == 1  \
                and not leakage_model["leakage_model"] == "HD":
            intermediate_values = AES.encrypt_block_leakage_model_from_input_fast(plaintexts, keys, leakage_model)
        else:
            aes = AES(keys[0])
            intermediate_values = []
            get_state = AES.get_state_index_encryption if leakage_model["direction"] == "Encryption" \
                else AES.get_state_index_decryption
            if leakage_model["leakage_model"] == "HD":

                state_index = -1
                state_index_first = get_state(
                    leakage_model['round_first'], leakage_model['target_state_first'], aes.num_rounds)
                state_index_second = get_state(
                    leakage_model['round_second'], leakage_model['target_state_second'], aes.num_rounds)
            else:
                state_index = get_state(
                    leakage_model['round'], leakage_model['target_state'], aes.num_rounds)
                state_index_first = -1
                state_index_second = -1
            if leakage_model["attack_direction"] == "input":
                keys = np.array(keys, dtype=np.uint8)
                plaintexts = np.array(plaintexts, dtype=np.uint8)
                aes = AES(keys[0])
                prev_key = keys[0]
                for plaintext, key in zip(plaintexts, keys):
                    if not np.array_equal(prev_key, key):
                        aes.expand_key(key)
                        prev_key = key
                    intermediate_values.append(
                        aes.encryption_iv_with_state(list(plaintext), leakage_model, state_index,
                                                     state_index_first, state_index_second))
            else:
                keys = np.array(keys, dtype=np.uint8)
                ciphertexts = np.array(ciphertexts, dtype=np.uint8)
                aes = AES(keys[0])
                prev_key = keys[0]
                for ciphertext, key in zip(ciphertexts, keys):
                    if not np.array_equal(prev_key, key):
                        aes.expand_key(key)
                        prev_key = key
                    intermediate_values.append(
                        aes.decryption_iv_with_state(list(ciphertext), leakage_model, state_index,
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
            labels.append(AES.get_intermediate_values(plaintexts, keys, ciphertexts, leakage_model))

        return labels

