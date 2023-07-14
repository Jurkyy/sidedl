import numpy as np


s_box = np.array([0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2])

inv_s_box = np.array([0x5, 0xE, 0xF, 0x8, 0xC, 0x1, 0x2, 0xD, 0xB, 0x4, 0x6, 0x3, 0x0, 0x7, 0x9, 0xA])

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

def expand_key(key):
    """
    Method for expanding the key
    :param key: key to expand
    :return: Array containing round keys
    """
    if len(key) == 128:
        return expand_key_128(key)
    round_keys = np.zeros((32, 64), dtype=np.bool_)
    key_register = np.array(key, dtype=np.bool_)
    for i in range(1, 33):
        round_keys[i - 1] = np.array(key_register[:64], dtype=np.bool_)
        key_register = np.roll(key_register, -61)
        round_counter = i
        key_register[0:4] = sbox_word(key_register[0:4])
        key_register[-20:-15] = np.logical_xor(key_register[-20:-15],
                                               convert_int_to_bool_arr(round_counter, 5))

    return round_keys

def expand_key_128(key):
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
        key_register[:4] = sbox_word(key_register[:4])
        key_register[4:8] = sbox_word(key_register[4:8])
        key_register[-67:-62] = np.logical_xor(key_register[-67:-62],

                                               convert_int_to_bool_arr(round_counter, 5))
    return round_keys

def expand_key_multiple(key):
    """
    Method for expanding the key
    :param key: key to expand
    :return: Array containing round keys
    """
    key = np.unpackbits(key, axis=1)
    if len(key[0]) == 128:
        return expand_key_128_multiple(key)
    round_keys = np.zeros((key.shape[0], 32, 64), dtype=np.bool_)
    key_register = np.array(key, dtype=np.bool_)
    round_counter = np.zeros((key.shape[0], 1), dtype=np.ubyte)
    for i in range(1, 33):
        round_keys[:, i - 1] = np.array(key_register[:, :64], dtype=np.bool_)
        key_register = np.roll(key_register, -61)
        key_register[:, 0:4] = sbox_word_multpile(key_register[:, 0:4])
        round_counter = round_counter + 1
        key_register[:, -20:-15] = np.logical_xor(key_register[:, -20:-15],
                                                  np.unpackbits(round_counter, axis=1)[:, 3:])

    return round_keys

def expand_key_128_multiple(key):
    """
    Method for expanding the key of length 128
    :param key: key to expand
    :return: Array containing round keys
    """
    round_keys = np.zeros((key.shape[0], 32, 64), dtype=np.bool_)
    key_register = key
    round_counter = np.zeros((key.shape[0], 1), dtype=np.ubyte)
    for i in range(1, 33):
        round_keys[:, i-1] = key_register[:, :64]
        key_register = np.roll(key_register, 66)
        round_counter = round_counter + 1
        key_register[:, :4] = sbox_word_multpile(key_register[:, :4])
        key_register[:, 4:8] = sbox_word_multpile(key_register[:, 4:8])
        key_register[:, -67:-62] = np.logical_xor(key_register[:, -67:-62],
                                                  np.unpackbits(round_counter, axis=1)[:, 3:])
    return round_keys


def sbox_word(word):
    """
    Applies sbox to a 4-bit word
    :param word: boolean array of the word
    :return: boolean array of substituted word
    """
    temp_val = s_box[int(np.packbits(word))//16]
    return convert_int_to_bool_arr(temp_val, 4)

def inv_sbox_word(word):
    """
    Applies inverted sbox to a 4-bit word
    :param word: boolean array of the word
    :return: boolean array of substituted word
    """
    temp_val = inv_s_box[int(np.packbits(word)//16)]
    return convert_int_to_bool_arr(temp_val, 4)

def sbox_word_multpile(words):
    """
    Applies sbox to a 4-bit word
    :param words: boolean array of the word
    :return: boolean array of substituted word
    """

    temp_val = s_box[np.packbits(words, axis=1)//16]
    return np.unpackbits(np.array(temp_val, dtype=np.ubyte), axis=1)[:, 4:]


def inv_sbox_word_multiple(words):
    """
    Applies inverted sbox to a 4-bit word
    :param words: boolean array of the word
    :return: boolean array of substituted word
    """
    temp_val = inv_s_box[np.packbits(words, axis=1)//16]
    return np.unpackbits(np.array(temp_val, dtype=np.ubyte), axis=1)[:, 4:]


def sbox_state(state):
    """
    Utility function for applying sbox
    """
    new_state = np.array(state, dtype=np.bool_)
    for j in range(16):
        new_state[j*4:(j+1)*4] = sbox_word(state[j*4:(j+1)*4])
    return new_state


def inv_sbox_state(state):
    """
    Utility function for applying inverse sbox
    """
    new_state = np.array(state, dtype=np.bool_)
    for j in range(16):
        new_state[j*4:(j+1)*4] = inv_sbox_word(state[j*4:(j+1)*4])
    return new_state


def sbox_state_multiple(state):
    """
    Utility function for applying sbox
    """
    new_state = np.array(state, dtype=np.bool_)
    for j in range(16):
        new_state[:, j*4:(j+1)*4] = sbox_word_multpile(state[:, j*4:(j+1)*4])
    return new_state


def inv_sbox_state_multiple(state):
    """
    Utility function for applying inverse sbox
    """
    new_state = np.array(state, dtype=np.bool_)
    for j in range(16):
        new_state[:, j*4:(j+1)*4] = inv_sbox_word_multiple(state[:, j*4:(j+1)*4])
    return new_state

def player_state(state):
    """
    Utility function for applying permutation
    """
    new_state = np.array(state, dtype=np.bool_)
    for i in range(64):
        new_state[63 - p_layer_order[i]] = state[63-i]
    return new_state


def inv_player_state(state):
    """
    Utility function for applying inverse permutation
    """
    new_state = np.array(state, dtype=np.bool_)
    for i in range(64):
        new_state[63 - p_layer_order_inv[i]] = state[63-i]
    return new_state


def player_state_multiple(states):
    """
    Utility function for applying permutation
    """
    new_states = np.array(states, dtype=np.bool_)
    for i in range(64):
        new_states[:, 63 - p_layer_order[i]] = states[:, 63-i]
    return new_states


def inv_player_state_multiple(states):
    """
    Utility function for applying inverse permutation
    """
    new_states = np.array(states, dtype=np.bool_)
    for i in range(64):
        new_states[:, 63 - p_layer_order_inv[i]] = states[:, 63-i]
    return new_states


def encryption_multiple(plaintexts, keys):
    round_keys = expand_key_multiple(keys)
    state = np.unpackbits(plaintexts, axis=1)
    for i in range(31):
        # AddroundKey
        state = np.logical_xor(state, round_keys[:, i])
        # Sbox
        state = sbox_state_multiple(state)
        # permute
        state = player_state_multiple(state)
    # AddroundKey
    state = np.logical_xor(state, round_keys[:, 31])
    return np.packbits(state, axis=1)


def decryption_multiple(ciphertext, keys):
    """
    Decryption function
    :param ciphertext: ciphertext
    :return: decrypted result
    """
    round_keys = expand_key_multiple(keys)
    state = np.unpackbits(ciphertext, axis=1)
    for i in range(31, 0, -1):
        # AddroundKey
        state = np.logical_xor(state, round_keys[:, i])
        #InvPlayer
        state = inv_player_state_multiple(state)
        # permute
        state = inv_sbox_state_multiple(state)
    # AddroundKey
    state = np.logical_xor(state, round_keys[:, 0])
    return np.packbits(state, axis=1)


def get_encryption_intermediates(plaintexts, keys):
    round_keys = expand_key_multiple(keys)

    state = np.unpackbits(plaintexts, axis=1)
    intermediates = {"plaintext": state}
    for i in range(31):
        # AddroundKey
        state = np.logical_xor(state, round_keys[:, i])
        intermediates[f"add_round_key_{i}"] = state
        # Sbox
        state = sbox_state_multiple(state)
        intermediates[f"sbox_{i}"] = state
        # permute
        state = player_state_multiple(state)
        intermediates[f"permutation_{i}"] = state
    # AddroundKey
    state = np.logical_xor(state, round_keys[:, 31])
    intermediates["ciphertext"] = state
    return intermediates


def get_decryption_intermediates(ciphertext, keys):
    """
    Decryption function
    :param ciphertext: ciphertext
    :return: decrypted result
    """
    round_keys = expand_key_multiple(keys)
    state = np.unpackbits(ciphertext, axis=1)
    intermediates = {"ciphertext": state}
    for i in range(31, 0, -1):
        # AddroundKey
        state = np.logical_xor(state, round_keys[:, i])
        intermediates[f"add_round_key_{i}"] = state
        #InvPlayer

        state = inv_player_state_multiple(state)
        intermediates[f"inv_permutation_{i}"] = state
        # SBox
        state = inv_sbox_state_multiple(state)
        intermediates[f"inv_sbox_{i}"] = state
    # AddroundKey
    state = np.logical_xor(state, round_keys[:, 0])
    intermediates["plaintext"] = state
    return intermediates


def encryption_intermediates_labels():
    labels = ["plaintext", "ciphertext"]
    for i in range(31):
        labels.append(f"sbox_{i}")
        labels.append(f"permutation_{i}")
        labels.append(f"add_round_key_{i}")
    return labels


def decryption_intermediates_labels():
    labels = ["plaintext", "ciphertext"]
    for i in range(31, 0, -1):
        labels.append(f"inv_sbox_{i}")
        labels.append(f"inv_permutation_{i}")
        labels.append(f"inv_add_round_key_{i}")
    return labels
