import os
import pickle
import base64
import json
import numpy as np
import torch
import tenseal as ts


class CKKSCrypto:
    def __init__(self, poly_modulus_degree=8192, coeff_mod_bit_sizes=None, global_scale=2**40):
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes or [40, 21, 21, 21, 21, 21, 21, 40]
        self.global_scale = global_scale
        self.context = None
        self.public_key = None
        self.private_key = None
        self.relin_keys = None

    def generate_keypair(self):
        print(f'[CKKS] 生成密钥 (poly_degree={self.poly_modulus_degree})...')
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            self.poly_modulus_degree,
            self.coeff_mod_bit_sizes
        )
        self.context.global_scale = self.global_scale
        self.context.generate_galois_keys()
        self.public_key = self.context.public_key()
        self.private_key = self.context.secret_key()
        self.relin_keys = self.context.relin_keys()
        print('[CKKS] 密钥生成完成')
        return self.public_key, self.private_key

    def get_public_key_serialized(self):
        if self.context is None:
            return None
        return base64.b64encode(self.context.serialize()).decode('utf-8')

    def load_public_key_from_serialized(self, serialized):
        self.context = ts.Context.load(base64.b64decode(serialized))
        self.public_key = self.context.public_key()
        self.relin_keys = self.context.relin_keys()

    def get_secret_key_serialized(self):
        if self.private_key is None:
            return None
        return base64.b64encode(self.context.serialize()).decode('utf-8')

    def load_secret_key_from_serialized(self, serialized):
        self.context = ts.Context.load(base64.b64decode(serialized))
        self.private_key = self.context.secret_key()

    def encrypt_tensor(self, tensor):
        if self.context is None:
            raise ValueError('TenSEAL context未初始化')

        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().numpy().flatten().astype(np.float64)
        elif isinstance(tensor, np.ndarray):
            arr = tensor.flatten().astype(np.float64)
        else:
            arr = np.array(tensor).flatten().astype(np.float64)

        encrypted_vector = ts.ckks_vector(self.context, arr)
        return encrypted_vector

    def serialize_encrypted_tensor(self, encrypted_vector):
        return base64.b64encode(encrypted_vector.serialize()).decode('utf-8')

    def deserialize_encrypted_tensor(self, serialized, original_shape=None):
        if self.context is None:
            raise ValueError('TenSEAL context未初始化')
        encrypted_vector = ts.ckks_vector_from(self.context, base64.b64decode(serialized))
        return encrypted_vector

    def decrypt_tensor(self, encrypted_vector, original_shape=None):
        if self.private_key is None:
            raise ValueError('私钥未初始化')

        decrypted = encrypted_vector.decrypt(self.private_key)
        result = np.array(decrypted)

        if original_shape is not None:
            result = result.reshape(original_shape)
        return result

    def aggregate_encrypted(self, encrypted_tensors_list, weights=None):
        if not encrypted_tensors_list:
            return None

        if weights is None:
            weights = [1.0] * len(encrypted_tensors_list)

        result = None
        for enc_tensor, weight in zip(encrypted_tensors_list, weights):
            if weight == 1.0:
                if result is None:
                    result = enc_tensor
                else:
                    result = result + enc_tensor
            else:
                if result is None:
                    result = enc_tensor * weight
                else:
                    result = result + (enc_tensor * weight)
        return result

    def multiply_encrypted(self, enc_a, enc_b):
        result = enc_a * enc_b
        return result

    def encrypt_model_state_dict(self, state_dict):
        encrypted_state = {}
        for key, tensor in state_dict.items():
            encrypted_state[key] = self.encrypt_tensor(tensor)
        return encrypted_state

    def decrypt_model_state_dict(self, encrypted_state_dict, template_state_dict):
        decrypted_state = {}
        for key, encrypted_tensor in encrypted_state_dict.items():
            original_shape = None
            if key in template_state_dict:
                original_shape = template_state_dict[key].shape
            decrypted_state[key] = self.decrypt_tensor(encrypted_tensor, original_shape)
        return decrypted_state

    def save_keys(self, public_key_path, private_key_path=None):
        with open(public_key_path, 'wb') as f:
            f.write(base64.b64decode(self.get_public_key_serialized()))
        if self.private_key is not None and private_key_path:
            with open(private_key_path, 'wb') as f:
                f.write(base64.b64decode(self.get_secret_key_serialized()))

    def load_keys(self, public_key_path, private_key_path=None):
        with open(public_key_path, 'rb') as f:
            self.load_public_key_from_serialized(base64.b64encode(f.read()).decode('utf-8'))
        if private_key_path and os.path.exists(private_key_path):
            with open(private_key_path, 'rb') as f:
                self.load_secret_key_from_serialized(base64.b64encode(f.read()).decode('utf-8'))


def test_crypto():
    print('=' * 60)
    print('测试CKKS同态加密功能')
    print('=' * 60)

    crypto = CKKSCrypto(poly_modulus_degree=8192, global_scale=2**40)
    crypto.generate_keypair()

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    b = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])

    print(f'\n原始数据 A: {a}')
    print(f'原始数据 B: {b}')

    enc_a = crypto.encrypt_tensor(a)
    enc_b = crypto.encrypt_tensor(b)

    print(f'\n加密完成')

    dec_a = crypto.decrypt_tensor(enc_a)
    print(f'\n解密后 A: {dec_a}')

    enc_sum = enc_a + enc_b
    dec_sum = crypto.decrypt_tensor(enc_sum)
    print(f'\n密文相加后解密: {dec_sum}')
    print(f'预期结果: {a + b}')

    enc_weighted = enc_a * 0.5
    dec_weighted = crypto.decrypt_tensor(enc_weighted)
    print(f'\n密文标量乘法(0.5)后解密: {dec_weighted}')
    print(f'预期结果: {a * 0.5}')

    print('\n' + '=' * 60)
    print('测试完成！')
    print('=' * 60)


if __name__ == '__main__':
    test_crypto()