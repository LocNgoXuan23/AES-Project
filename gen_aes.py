from AES import encryption, decryption
from engine import allPermutations, concatenatePermutationList
from sklearn.utils import shuffle
import random
random.seed(111)

def genAES(numOGen, fileName):
    plaintext = "0123456789abcdeffedcba9876543210"
    key = "0f1571c947d9e8590cb7add6af7f6798"

    # cirpher = encryption(plaintext, key)
    # plain = decryption(cirpher, key)

    # print('plaintext : ', plaintext)
    # print('key : ', key)

    # print('cirpher : ', cirpher)
    # print('plain : ', plain)

    all_plaintext = allPermutations(plaintext, 4)
    plaintext_list = concatenatePermutationList(all_plaintext, 8)
    print(len(plaintext_list))

    all_key = allPermutations(key, 4)
    key_list = concatenatePermutationList(all_key, 8)
    print(len(key_list))

    plaintext_list, key_list = shuffle(plaintext_list, key_list, random_state=10)

    count = 0
    conti = True
    with open(fileName, "w") as file:
        for p in plaintext_list:
            for k in key_list:
                cirpher = encryption(p, k)
                file.write(str(cirpher) + "\n")
                count += 1
                print(count)
                if count == numOGen:
                    conti = False
                    break

            if not conti:
                break


