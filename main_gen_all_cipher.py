from gen_aes import genAES

if __name__ == '__main__':
    plaintext = "0123456789abcdeffedcba9876543210"
    key = "0f1571c947d9e8590cb7add6af7f6798"
    print("START !!!")
    genAES(8000, 'my_cipher_2.txt', plaintext, key)
    print("DONE !!!")