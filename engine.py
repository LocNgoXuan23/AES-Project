from itertools import permutations

def allPermutations(str, num):
    permList = permutations(str, num)
    permutation_list = []
    for perm in list(permList):
        permutation_list.append(''.join(perm))
    return permutation_list

def concatenatePermutationList(permutation_list, num):
    new_list = []
    for i in range(0, len(permutation_list), num):
        s = ''
        for j in range(num):
            s += permutation_list[i + j]
        # print(s)
        new_list.append(s)
    return new_list

if __name__ == '__main__':
    str = '0123456789abcdeffedcba9876543210'
    all_str = allPermutations(str, 4)
    # print(all_str)
    # print(len(str))
    # print(len(all_str))
    new_list = concatenatePermutationList(all_str, 8)