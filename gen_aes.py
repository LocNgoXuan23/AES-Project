from AES import encryption, decryption
from engine import allPermutations, concatenatePermutationList
from sklearn.utils import shuffle
import random
random.seed(111)

def genOneTime(plaintext, key):
    cirpher = encryption(plaintext, key)
    plain = decryption(cirpher, key)

    print('plaintext : ', plaintext)
    print('key : ', key)

    print('cirpher : ', cirpher)
    print('plain : ', plain)

def genAES(numOGen, fileName, plaintext, key):
    # GET ALL PERMUTATIONS OF PLAINTEXT, KEY AND SHUFFLE 
    all_plaintext = allPermutations(plaintext, 4)
    all_key = allPermutations(key, 4)
    all_plaintext, all_key = shuffle(all_plaintext, all_key, random_state=10)

    # CONCAT PERMUTATIONS TO GET 128 BIT AND SHUFFLE
    plaintext_list = concatenatePermutationList(all_plaintext, 8)
    key_list = concatenatePermutationList(all_key, 8)

    print(len(plaintext_list))
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


if __name__ == '__main__':
    plaintext = "0123456789abcdeffedcba9876543210"
    key = "0f1571c947d9e8590cb7add6af7f6798"
    genOneTime(plaintext, key)

