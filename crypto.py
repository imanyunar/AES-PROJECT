import numpy as np
from sbox_utils import generate_sboxes  # pastikan sbox_test.py ada di folder yang sama

class AESCustom:
    def __init__(self, sbox_name="AES", key_bytes=None, random_seed=None):
        """
        Inisialisasi AES custom.
        sbox_name: "AES", "A0", "A1", "A2", "K44", atau "RANDOM"
        key_bytes: bytearray atau list of int (panjang 16 byte)
        random_seed: jika ingin S-box RANDOM deterministik
        """
        self.sboxes = generate_sboxes(include_random=True, seed=random_seed)
        if sbox_name not in self.sboxes:
            raise ValueError(f"S-box '{sbox_name}' tidak tersedia.")
        self.sbox = self.sboxes[sbox_name]
        if key_bytes is None:
            # jika tidak ada key, generate random 16 byte
            self.key = np.random.randint(0, 256, size=16, dtype=np.uint8)
        else:
            if len(key_bytes) != 16:
                raise ValueError("Key harus 16 byte.")
            # FIX: Handle bytes objects properly
            if isinstance(key_bytes, bytes):
                self.key = np.frombuffer(key_bytes, dtype=np.uint8).copy()
            else:
                self.key = np.array(key_bytes, dtype=np.uint8)

        # precompute inverse sbox
        self.inv_sbox = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            self.inv_sbox[self.sbox[i]] = i

    def encrypt(self, plaintext_bytes):
        """
        Encrypt byte array (panjang <=16 bytes)
        """
        if len(plaintext_bytes) > 16:
            raise ValueError("Plaintext maksimal 16 bytes.")
        # FIX: Handle bytes objects properly
        if isinstance(plaintext_bytes, bytes):
            pt = np.frombuffer(plaintext_bytes, dtype=np.uint8)
        else:
            pt = np.array(plaintext_bytes, dtype=np.uint8)
        ct = np.array([self.sbox[b ^ self.key[i]] for i, b in enumerate(pt)])
        return bytes(ct)

    def decrypt(self, cipher_bytes):
        """
        Decrypt byte array (panjang <=16 bytes)
        """
        if len(cipher_bytes) > 16:
            raise ValueError("Ciphertext maksimal 16 bytes.")
        # FIX: Handle bytes objects properly
        if isinstance(cipher_bytes, bytes):
            ct = np.frombuffer(cipher_bytes, dtype=np.uint8)
        else:
            ct = np.array(cipher_bytes, dtype=np.uint8)
        pt = np.array([self.inv_sbox[b] ^ self.key[i] for i, b in enumerate(ct)])
        return bytes(pt)

    def set_sbox(self, sbox_name, random_seed=None):
        """
        Ganti S-box saat runtime
        """
        self.sboxes = generate_sboxes(include_random=True, seed=random_seed)
        if sbox_name not in self.sboxes:
            raise ValueError(f"S-box '{sbox_name}' tidak tersedia.")
        self.sbox = self.sboxes[sbox_name]
        # update inverse sbox
        for i in range(256):
            self.inv_sbox[self.sbox[i]] = i

    def set_key(self, key_bytes):
        """
        Ganti key saat runtime
        """
        if len(key_bytes) != 16:
            raise ValueError("Key harus 16 byte.")
        # FIX: Handle bytes objects properly
        if isinstance(key_bytes, bytes):
            self.key = np.frombuffer(key_bytes, dtype=np.uint8).copy()
        else:
            self.key = np.array(key_bytes, dtype=np.uint8)