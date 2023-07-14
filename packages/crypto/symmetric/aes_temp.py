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

shift_row_mask = np.array([0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11])
inv_shift_row_mask = np.array([0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3])


def xtime_multiple(a: np.array):
    temp = a & 0x80
    temp_0 = np.argwhere(temp == 0).T
    temp_1 = np.argwhere(temp == 128).T
    a_0 = a[temp_0[0]] << 1
    a_1 = ((a[temp_1[0]] << 1) ^ 0x1B) & 0xFF
    args = np.concatenate((temp_0[0], temp_1[0]), axis=0)
    result = np.concatenate((a_0, a_1), axis=0)
    args = np.argsort(args)
    return result[args]


def mix_single_column_multiple(a: np.array):
    t = a[:, 0] ^ a[:, 1] ^ a[:, 2] ^ a[:, 3]
    u = a[:, 0].copy()
    a[:, 0] ^= t ^ xtime_multiple(a[:, 0] ^ a[:, 1])
    a[:, 1] ^= t ^ xtime_multiple(a[:, 1] ^ a[:, 2])
    a[:, 2] ^= t ^ xtime_multiple(a[:, 2] ^ a[:, 3])
    a[:, 3] ^= t ^ xtime_multiple(a[:, 3] ^ u)
    return a


def mix_columns_multiple(s: np.array):
    for i in range(4):
        s[:, i * 4:i * 4 + 4] = mix_single_column_multiple(s[:, i * 4:i * 4 + 4])
    return s


def inv_mix_columns_multiple(s: np.array):
    for i in range(4):
        u = xtime_multiple(xtime_multiple(s[:, i * 4] ^ s[:, i * 4 + 2]))
        v = xtime_multiple(xtime_multiple(s[:, i * 4 + 1] ^ s[:, i * 4 + 3]))
        s[:, i * 4] ^= u
        s[:, i * 4 + 1] ^= v
        s[:, i * 4 + 2] ^= u
        s[:, i * 4 + 3] ^= v
    s = mix_columns_multiple(s)
    return s


def sub_bytes(s):
    return s_box[s]


def inv_sub_bytes(s):
    return inv_s_box[s]


def shift_rows(s):
    s = np.array(s)
    return s[shift_row_mask]


def inv_shift_rows(s):
    s = np.array(s)
    return s[inv_shift_row_mask]


def shift_rows_multiple(s):
    s = np.array(s)
    return s[:, shift_row_mask]


def inv_shift_rows_multiple(s):
    s = np.array(s)
    return s[:, inv_shift_row_mask]


def add_round_key(s, k):
    return s ^ k


xtime = lambda a: (((a << 1) ^ 0x1B) & 0xFF) if (a & 0x80) else (a << 1)


def mix_single_column(a):
    t = a[0] ^ a[1] ^ a[2] ^ a[3]
    u = a[0]
    a[0] ^= t ^ xtime(a[0] ^ a[1])
    a[1] ^= t ^ xtime(a[1] ^ a[2])
    a[2] ^= t ^ xtime(a[2] ^ a[3])
    a[3] ^= t ^ xtime(a[3] ^ u)
    return a


def mix_columns(s):
    for i in range(4):
        s[i * 4:i * 4 + 4] = mix_single_column(s[i * 4:i * 4 + 4])
    return s


def inv_mix_columns(s):
    for i in range(4):
        u = xtime(xtime(s[i * 4] ^ s[i * 4 + 2]))
        v = xtime(xtime(s[i * 4 + 1] ^ s[i * 4 + 3]))
        s[i * 4] ^= u
        s[i * 4 + 1] ^= v
        s[i * 4 + 2] ^= u
        s[i * 4 + 3] ^= v
    s = mix_columns(s)
    return s


r_con = (
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
    0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A,
    0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A,
    0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39,
)


def expand_key(master_key):
    iteration_count = 0
    for i in range(4, 44):
        word = list(master_key[len(master_key) - 4:])
        if i % 4 == 0:
            word.append(word.pop(0))
            word = s_box[word]
            word[0] ^= r_con[i // 4]

        word = np.array(word) ^ np.array(master_key[iteration_count * 4:iteration_count * 4 + 4])
        for w in word:
            master_key.append(w)

        iteration_count += 1

    return [master_key[16 * i: 16 * (i + 1)] for i in range(len(master_key) // 16)]


def expand_key_multiple(master_key: np.array):
    master_key = np.array(master_key, dtype=np.uint8)
    nb_bytes_key = len(master_key[0])
    expanded_key = np.zeros((len(master_key), nb_bytes_key * 11), dtype=np.uint8)

    iteration_count = 0
    word_count = 16

    expanded_key[:, :nb_bytes_key] = master_key

    for i in range(4, 44):
        """ take last 4 bytes of expanded key """
        word = expanded_key[:, word_count - 4:word_count]

        if i % 4 == 0:
            word = np.roll(word, 3)
            word = np.array(word, dtype=np.uint8)
            word = s_box[word]
            word[:, 0] ^= r_con[i // 4]

        word = word ^ expanded_key[:, iteration_count * 4:iteration_count * 4 + 4]

        expanded_key[:, word_count:word_count + 4] = word

        word_count += 4
        iteration_count += 1

    return np.array([expand_key_row.reshape(11, nb_bytes_key) for expand_key_row in expanded_key])


def encrypt_block(plaintext, key):
    """
    :argument:
        plaintext (numpy array)
        key (numpy array)
    :return:
        ciphertext (numpy array)
    """

    round_keys = expand_key(key)
    state = add_round_key(plaintext, round_keys[0])

    for i in range(1, 10):
        state = sub_bytes(state)
        state = shift_rows(state)
        state = mix_columns(state)
        state = add_round_key(state, round_keys[i])

    state = sub_bytes(state)
    state = shift_rows(state)
    state = add_round_key(state, round_keys[-1])
    return state


def encrypt_block_multiple(plaintexts: np.array, keys: np.array):
    """
    :argument:
        plaintext (numpy array)
        key (numpy array)
    :return:
        ciphertext (numpy array)
    """

    round_keys = expand_key_multiple(keys)
    state = add_round_key(plaintexts, round_keys[:, 0])

    for i in range(1, 10):
        state = sub_bytes(state)
        state = shift_rows_multiple(state)
        state = mix_columns_multiple(state)
        state = add_round_key(state, round_keys[:, i])

    state = sub_bytes(state)
    state = shift_rows_multiple(state)
    state = add_round_key(state, round_keys[:, -1])
    return state


def get_encryption_intermediates(plaintexts: np.array, keys: np.array):
    """
    :argument:
        plaintext (numpy array)
        key (numpy array)
    :return:
        ciphertext (numpy array)
    """

    intermediates = {"plaintext": plaintexts}
    round_keys = expand_key_multiple(keys)
    state = add_round_key(plaintexts, round_keys[:, 0])
    intermediates["add_round_key_0"] = state

    for i in range(1, 10):
        state = sub_bytes(state)
        intermediates[f"sbox_{i}"] = state
        state = shift_rows_multiple(state)
        intermediates[f"shift_rows_{i}"] = state
        state = mix_columns_multiple(state)
        intermediates[f"mix_columns_{i}"] = state
        state = add_round_key(state, round_keys[:, i])
        intermediates[f"add_round_key_{i}"] = state

    state = sub_bytes(state)
    intermediates["sbox_10"] = state
    state = shift_rows_multiple(state)
    intermediates["shift_rows_10"] = state
    state = add_round_key(state, round_keys[:, -1])
    intermediates["ciphertext"] = state
    return intermediates


def encryption_intermediates_labels():
    labels = ["plaintext", "add_round_key_0"]
    for i in range(1, 10):
        labels.append(f"sbox_{i}")
        labels.append(f"shift_rows_{i}")
        labels.append(f"mix_columns_{i}")
        labels.append(f"add_round_key_{i}")
    labels.append("sbox_10")
    labels.append("shift_rows_10")
    labels.append("ciphertext")
    return labels


def decrypt_block(ciphertext, key):
    """
    :argument:
        ciphertext (numpy array)
        key (numpy array)
    :return:
        plaintext (numpy array)
    """

    round_keys = expand_key(key)
    state = add_round_key(ciphertext, round_keys[10])
    state = inv_shift_rows(state)
    state = inv_sub_bytes(state)

    for i in range(9, 0, -1):
        state = add_round_key(state, round_keys[i])
        state = inv_mix_columns(state)
        state = inv_shift_rows(state)
        state = inv_sub_bytes(state)

    return add_round_key(state, round_keys[0])


def decrypt_block_multiple(ciphertexts, keys):
    """
    :argument:
        ciphertext (numpy array)
        key (numpy array)
    :return:
        plaintext (numpy array)
    """

    round_keys = expand_key_multiple(keys)
    state = add_round_key(ciphertexts, round_keys[:, 10])
    state = inv_shift_rows_multiple(state)
    state = inv_sub_bytes(state)

    for i in range(9, 0, -1):
        state = add_round_key(state, round_keys[:, i])
        state = inv_mix_columns_multiple(state)
        state = inv_shift_rows_multiple(state)
        state = inv_sub_bytes(state)

    return add_round_key(state, round_keys[:, 0])


def get_decryption_intermediates(ciphertexts, keys):
    """
    :argument:
        ciphertext (numpy array)
        key (numpy array)
    :return:
        plaintext (numpy array)
    """

    intermediates = {"ciphertext": ciphertexts}
    round_keys = expand_key_multiple(keys)
    state = add_round_key(ciphertexts, round_keys[:, 10])
    intermediates["add_round_key_10"] = state
    state = inv_shift_rows_multiple(state)
    intermediates["inv_shift_rows_10"] = state
    state = inv_sub_bytes(state)
    intermediates["inv_sbox_10"] = state

    for i in range(9, 0, -1):
        state = add_round_key(state, round_keys[:, i])
        intermediates[f"add_round_key_{i}"] = state
        state = inv_mix_columns_multiple(state)
        intermediates[f"inv_mix_columns_{i}"] = state
        state = inv_shift_rows_multiple(state)
        intermediates[f"inv_shift_rows_{i}"] = state
        state = inv_sub_bytes(state)
        intermediates[f"inv_sbox_{i}"] = state

    state = add_round_key(state, round_keys[:, 0])
    intermediates["plaintext"] = state
    return intermediates


def decryption_intermediates_labels():
    labels = ["ciphertext", "add_round_key_10", "inv_shift_rows_10", "inv_sbox_10"]
    for i in range(9, 0, -1):
        labels.append(f"add_round_key_{i}")
        labels.append(f"inv_mix_columns_{i}")
        labels.append(f"inv_shift_rows_{i}")
        labels.append(f"inv_sbox_{i}")
    labels.append("plaintext")
    return labels


def get_round_key(key, round):
    return expand_key(key)[round]


def run_test():
    key = ([int(x) for x in bytearray.fromhex("2b7e151628aed2a6abf7158809cf4f3c")])

    plaintexts = np.random.randint(0, 255, (50000, 16))
    print(plaintexts)

    keys = np.zeros((50000, 16))
    for i in range(50000):
        keys[i] = ([int(x) for x in bytearray.fromhex("2b7e151628aed2a6abf7158809cf4f3c")])
    encryption_intermediates = get_encryption_intermediates(plaintexts, keys)
    print(encryption_intermediates["ciphertext"])

    keys = np.zeros((50000, 16))
    for i in range(50000):
        keys[i] = ([int(x) for x in bytearray.fromhex("2b7e151628aed2a6abf7158809cf4f3c")])
    plaintexts_intermediates = get_decryption_intermediates(encryption_intermediates["ciphertext"], keys)
    print(plaintexts_intermediates["plaintext"])

    # plaintext = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    # print(plaintext)
    #
    # key = ([int(x) for x in bytearray.fromhex("2b7e151628aed2a6abf7158809cf4f3c")])
    # ciphertext = encrypt_block(plaintext, key)
    # print(ciphertext)
    #
    # key = ([int(x) for x in bytearray.fromhex("2b7e151628aed2a6abf7158809cf4f3c")])
    # plaintext2 = decrypt_block(ciphertext, key)
    # print(plaintext2)

    # key = ([int(x) for x in bytearray.fromhex("2b7e151628aed2a6abf7158809cf4f3c")])
    # plaintext2 = decrypt_block(ciphertext, key)
    # print(plaintext2)

    # a = np.random.randint(0, 256, 10)
    # x = []
    # for i in range(10):
    #     x.append(xtime(a[i]))
    # print(x)
    # print(list(xtime_multiple(a)))

    # states = np.random.randint(0, 255, (10, 16))
    #
    # states1 = states.copy()
    # single = []
    # for i in range(10):
    #     # single.append(mix_columns(states1[i]))
    #     single.append(inv_mix_columns(mix_columns(states1[i])))
    #
    # states2 = states.copy()
    # # multiple = mix_columns_multiple(states2)
    # multiple = inv_mix_columns_multiple(mix_columns_multiple(states2))
    #
    # for i in range(10):
    #     print(states[i])
    #     print(single[i])
    #     print(multiple[i])


run_test()
