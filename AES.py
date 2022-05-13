import numpy as np

aes_sbox = [
    [int('63', 16), int('7c', 16), int('77', 16), int('7b', 16), int('f2', 16), int('6b', 16), int('6f', 16), int('c5', 16), int(
        '30', 16), int('01', 16), int('67', 16), int('2b', 16), int('fe', 16), int('d7', 16), int('ab', 16), int('76', 16)],
    [int('ca', 16), int('82', 16), int('c9', 16), int('7d', 16), int('fa', 16), int('59', 16), int('47', 16), int('f0', 16), int(
        'ad', 16), int('d4', 16), int('a2', 16), int('af', 16), int('9c', 16), int('a4', 16), int('72', 16), int('c0', 16)],
    [int('b7', 16), int('fd', 16), int('93', 16), int('26', 16), int('36', 16), int('3f', 16), int('f7', 16), int('cc', 16), int(
        '34', 16), int('a5', 16), int('e5', 16), int('f1', 16), int('71', 16), int('d8', 16), int('31', 16), int('15', 16)],
    [int('04', 16), int('c7', 16), int('23', 16), int('c3', 16), int('18', 16), int('96', 16), int('05', 16), int('9a', 16), int(
        '07', 16), int('12', 16), int('80', 16), int('e2', 16), int('eb', 16), int('27', 16), int('b2', 16), int('75', 16)],
    [int('09', 16), int('83', 16), int('2c', 16), int('1a', 16), int('1b', 16), int('6e', 16), int('5a', 16), int('a0', 16), int(
        '52', 16), int('3b', 16), int('d6', 16), int('b3', 16), int('29', 16), int('e3', 16), int('2f', 16), int('84', 16)],
    [int('53', 16), int('d1', 16), int('00', 16), int('ed', 16), int('20', 16), int('fc', 16), int('b1', 16), int('5b', 16), int(
        '6a', 16), int('cb', 16), int('be', 16), int('39', 16), int('4a', 16), int('4c', 16), int('58', 16), int('cf', 16)],
    [int('d0', 16), int('ef', 16), int('aa', 16), int('fb', 16), int('43', 16), int('4d', 16), int('33', 16), int('85', 16), int(
        '45', 16), int('f9', 16), int('02', 16), int('7f', 16), int('50', 16), int('3c', 16), int('9f', 16), int('a8', 16)],
    [int('51', 16), int('a3', 16), int('40', 16), int('8f', 16), int('92', 16), int('9d', 16), int('38', 16), int('f5', 16), int(
        'bc', 16), int('b6', 16), int('da', 16), int('21', 16), int('10', 16), int('ff', 16), int('f3', 16), int('d2', 16)],
    [int('cd', 16), int('0c', 16), int('13', 16), int('ec', 16), int('5f', 16), int('97', 16), int('44', 16), int('17', 16), int(
        'c4', 16), int('a7', 16), int('7e', 16), int('3d', 16), int('64', 16), int('5d', 16), int('19', 16), int('73', 16)],
    [int('60', 16), int('81', 16), int('4f', 16), int('dc', 16), int('22', 16), int('2a', 16), int('90', 16), int('88', 16), int(
        '46', 16), int('ee', 16), int('b8', 16), int('14', 16), int('de', 16), int('5e', 16), int('0b', 16), int('db', 16)],
    [int('e0', 16), int('32', 16), int('3a', 16), int('0a', 16), int('49', 16), int('06', 16), int('24', 16), int('5c', 16), int(
        'c2', 16), int('d3', 16), int('ac', 16), int('62', 16), int('91', 16), int('95', 16), int('e4', 16), int('79', 16)],
    [int('e7', 16), int('c8', 16), int('37', 16), int('6d', 16), int('8d', 16), int('d5', 16), int('4e', 16), int('a9', 16), int(
        '6c', 16), int('56', 16), int('f4', 16), int('ea', 16), int('65', 16), int('7a', 16), int('ae', 16), int('08', 16)],
    [int('ba', 16), int('78', 16), int('25', 16), int('2e', 16), int('1c', 16), int('a6', 16), int('b4', 16), int('c6', 16), int(
        'e8', 16), int('dd', 16), int('74', 16), int('1f', 16), int('4b', 16), int('bd', 16), int('8b', 16), int('8a', 16)],
    [int('70', 16), int('3e', 16), int('b5', 16), int('66', 16), int('48', 16), int('03', 16), int('f6', 16), int('0e', 16), int(
        '61', 16), int('35', 16), int('57', 16), int('b9', 16), int('86', 16), int('c1', 16), int('1d', 16), int('9e', 16)],
    [int('e1', 16), int('f8', 16), int('98', 16), int('11', 16), int('69', 16), int('d9', 16), int('8e', 16), int('94', 16), int(
        '9b', 16), int('1e', 16), int('87', 16), int('e9', 16), int('ce', 16), int('55', 16), int('28', 16), int('df', 16)],
    [int('8c', 16), int('a1', 16), int('89', 16), int('0d', 16), int('bf', 16), int('e6', 16), int('42', 16), int('68', 16), int(
        '41', 16), int('99', 16), int('2d', 16), int('0f', 16), int('b0', 16), int('54', 16), int('bb', 16), int('16', 16)]
]

rcon = np.array([[int('01', 16), int('02', 16), int('04', 16), int('08', 16), int('10', 16), int('20', 16), int('40', 16), 
                  int('80', 16), int('1b', 16), int('36', 16)]])
rcon = np.concatenate((rcon, np.zeros((3, 10), dtype=int)))

def printMatrix(input):
  for i in range(4):
    for j in range(4):
      pass

def break_in_grids_of_16(s):
  # s = 16 bytes
  output = []
  for i in range(4):
    sample = (s >> 32 * (3 - i)) & 0xFFFFFFFF
    grid = []
    for j in range(4):
      a = (sample >> 8 * (3 - j)) & 0xFF
      grid.append(a)
    output.append(grid)
  return output

def subBytes(byte):
  x = byte >> 4
  y = byte & 15
  return aes_sbox[x][y]

def rotateBytes(input, n):
  return np.append(input[n:], input[0:n])

def subWord(input):
  return np.array([subBytes(x) for x in input], dtype=int)

def keyExpansion(key):
  word = key
  j = 0
  for i in range(4, 44):
    temp = word[:, i - 1]
    if (i % 4 == 0):
      temp = subWord(rotateBytes(temp, 1)) ^ rcon[:, j]
      j += 1
    new_col = np.expand_dims(word[:, i - 4] ^ temp, axis=1)
    word = np.concatenate((word, new_col), axis=1)
  return word

# KEY SCHEDULE
def subBytesPlaintext(input):
  plaintext = np.empty((4, 4), dtype=int)
  for i in range(4):
    plaintext[:, i] = subWord(input[:, i])
  return plaintext

def shiftRow(input):
  plaintext = np.empty((4, 4), dtype=int)
  plaintext[0, :] = input[0, :]
  for i in range(1, 4):
    plaintext[i, :] = rotateBytes(input[i, :], i)
  return plaintext

def multiply_by_2(v):
  s = v << 1
  s &= 0xff
  if (v & 128) != 0:
    s = s ^ 0x1b
  return s


def multiply_by_3(v):
  return multiply_by_2(v) ^ v


def mix_column(column):
  r = [ multiply_by_2(column[0]) ^ multiply_by_3(column[1]) ^ column[2] ^ column[3],
        multiply_by_2(column[1]) ^ multiply_by_3(column[2]) ^ column[3] ^ column[0],
        multiply_by_2(column[2]) ^ multiply_by_3(column[3]) ^ column[0] ^ column[1],
        multiply_by_2(column[3]) ^ multiply_by_3(column[0]) ^ column[1] ^ column[2],
    ]
  return r

def mixColumns(grid):

  new_grid = [[], [], [], []]
  for i in range(4):
    col = [grid[j][i] for j in range(4)]
    col = mix_column(col)
    for i in range(4):
      new_grid[i].append(col[i])
  return np.array(new_grid)

def addRoundKey(block_grid, key_grid):
    r = []

    # 4 rows in the grid
    for i in range(4):
        r.append([])
        # 4 values on each row
        for j in range(4):
            r[-1].append(block_grid[i][j] ^ key_grid[i][j])
    return np.array(r)

reverse_aes_sbox = [
    [int('52', 16), int('09', 16), int('6a', 16), int('d5', 16), int('30', 16), int('36', 16), int('a5', 16), int('38', 16), int(
        'bf', 16), int('40', 16), int('a3', 16), int('9e', 16), int('81', 16), int('f3', 16), int('d7', 16), int('fb', 16)],
    [int('7c', 16), int('e3', 16), int('39', 16), int('82', 16), int('9b', 16), int('2f', 16), int('ff', 16), int('87', 16), int(
        '34', 16), int('8e', 16), int('43', 16), int('44', 16), int('c4', 16), int('de', 16), int('e9', 16), int('cb', 16)],
    [int('54', 16), int('7b', 16), int('94', 16), int('32', 16), int('a6', 16), int('c2', 16), int('23', 16), int('3d', 16), int(
        'ee', 16), int('4c', 16), int('95', 16), int('0b', 16), int('42', 16), int('fa', 16), int('c3', 16), int('4e', 16)],
    [int('08', 16), int('2e', 16), int('a1', 16), int('66', 16), int('28', 16), int('d9', 16), int('24', 16), int('b2', 16), int(
        '76', 16), int('5b', 16), int('a2', 16), int('49', 16), int('6d', 16), int('8b', 16), int('d1', 16), int('25', 16)],
    [int('72', 16), int('f8', 16), int('f6', 16), int('64', 16), int('86', 16), int('68', 16), int('98', 16), int('16', 16), int(
        'd4', 16), int('a4', 16), int('5c', 16), int('cc', 16), int('5d', 16), int('65', 16), int('b6', 16), int('92', 16)],
    [int('6c', 16), int('70', 16), int('48', 16), int('50', 16), int('fd', 16), int('ed', 16), int('b9', 16), int('da', 16), int(
        '5e', 16), int('15', 16), int('46', 16), int('57', 16), int('a7', 16), int('8d', 16), int('9d', 16), int('84', 16)],
    [int('90', 16), int('d8', 16), int('ab', 16), int('00', 16), int('8c', 16), int('bc', 16), int('d3', 16), int('0a', 16), int(
        'f7', 16), int('e4', 16), int('58', 16), int('05', 16), int('b8', 16), int('b3', 16), int('45', 16), int('06', 16)],
    [int('d0', 16), int('2c', 16), int('1e', 16), int('8f', 16), int('ca', 16), int('3f', 16), int('0f', 16), int('02', 16), int(
        'c1', 16), int('af', 16), int('bd', 16), int('03', 16), int('01', 16), int('13', 16), int('8a', 16), int('6b', 16)],
    [int('3a', 16), int('91', 16), int('11', 16), int('41', 16), int('4f', 16), int('67', 16), int('dc', 16), int('ea', 16), int(
        '97', 16), int('f2', 16), int('cf', 16), int('ce', 16), int('f0', 16), int('b4', 16), int('e6', 16), int('73', 16)],
    [int('96', 16), int('ac', 16), int('74', 16), int('22', 16), int('e7', 16), int('ad', 16), int('35', 16), int('85', 16), int(
        'e2', 16), int('f9', 16), int('37', 16), int('e8', 16), int('1c', 16), int('75', 16), int('df', 16), int('6e', 16)],
    [int('47', 16), int('f1', 16), int('1a', 16), int('71', 16), int('1d', 16), int('29', 16), int('c5', 16), int('89', 16), int(
        '6f', 16), int('b7', 16), int('62', 16), int('0e', 16), int('aa', 16), int('18', 16), int('be', 16), int('1b', 16)],
    [int('fc', 16), int('56', 16), int('3e', 16), int('4b', 16), int('c6', 16), int('d2', 16), int('79', 16), int('20', 16), int(
        '9a', 16), int('db', 16), int('c0', 16), int('fe', 16), int('78', 16), int('cd', 16), int('5a', 16), int('f4', 16)],
    [int('1f', 16), int('dd', 16), int('a8', 16), int('33', 16), int('88', 16), int('07', 16), int('c7', 16), int('31', 16), int(
        'b1', 16), int('12', 16), int('10', 16), int('59', 16), int('27', 16), int('80', 16), int('ec', 16), int('5f', 16)],
    [int('60', 16), int('51', 16), int('7f', 16), int('a9', 16), int('19', 16), int('b5', 16), int('4a', 16), int('0d', 16), int(
        '2d', 16), int('e5', 16), int('7a', 16), int('9f', 16), int('93', 16), int('c9', 16), int('9c', 16), int('ef', 16)],
    [int('a0', 16), int('e0', 16), int('3b', 16), int('4d', 16), int('ae', 16), int('2a', 16), int('f5', 16), int('b0', 16), int(
        'c8', 16), int('eb', 16), int('bb', 16), int('3c', 16), int('83', 16), int('53', 16), int('99', 16), int('61', 16)],
    [int('17', 16), int('2b', 16), int('04', 16), int('7e', 16), int('ba', 16), int('77', 16), int('d6', 16), int('26', 16), int(
        'e1', 16), int('69', 16), int('14', 16), int('63', 16), int('55', 16), int('21', 16), int('0c', 16), int('7d', 16)]
]

def reverseSubBytes(byte):
  x = byte >> 4
  y = byte & 15
  return reverse_aes_sbox[x][y]

def reverseSubWord(input):
  return np.array([reverseSubBytes(x) for x in input], dtype=int)

def subBytesCiphertext(input):
  ciphertext = np.empty((4, 4), dtype=int)
  for i in range(4):
    ciphertext[:, i] = reverseSubWord(input[:, i])
  return ciphertext

def reverseMixColumns(input):
  output = mixColumns(input)
  output = mixColumns(output)
  output = mixColumns(output)
  return output

def reverseShiftRow(input):
  ciphertext = np.empty((4, 4), dtype=int)
  ciphertext[0, :] = input[0, :]
  for i in range(1, 4):
    ciphertext[i, :] = rotateBytes(input[i, :], 4 - i)
  return ciphertext

def get_round_key(key_txt):

  key_hex = int(key_txt, 16)

  key = np.array(break_in_grids_of_16(key_hex)).T
  round_key = keyExpansion(key)

  return round_key

def encryption(plaintext_txt, key_txt):

  plaintext_hex = int(plaintext_txt, 16)

  plaintext = np.array(break_in_grids_of_16(plaintext_hex)).T
  

  after_mix = plaintext
  round_key = get_round_key(key_txt)

  count = 0
  printMatrix(plaintext)

  printMatrix(round_key[:, 0:4]) 

  start_round = addRoundKey(after_mix, round_key[:, 0:4])
  printMatrix(start_round)

  for i in range(4, 40, 4):
    count += 1

    after_sub = subBytesPlaintext(start_round)
    printMatrix(after_sub)

    after_shift = shiftRow(after_sub)
    printMatrix(after_shift)

    after_mix = mixColumns(after_shift)
    printMatrix(after_mix)

    printMatrix(round_key[:, i:i+4]) 

    start_round = addRoundKey(after_mix, round_key[:, i:i+4])
    printMatrix(start_round)


  after_sub = subBytesPlaintext(start_round)
  printMatrix(after_sub)

  after_shift = shiftRow(after_sub)
  printMatrix(after_shift)

  printMatrix(round_key[:, 40:44]) 

  start_round = addRoundKey(after_shift, round_key[:, 40:44])
  printMatrix(start_round)

  hex_string = ''
  
  for i in start_round.flatten('F'):
    hex_string += '{:02x}'.format(i)

  return hex_string

def decryption(ciphertext_txt, key_txt):

  ciphertext_hex = int(ciphertext_txt, 16)
  ciphertext = np.array(break_in_grids_of_16(ciphertext_hex)).T  

  round_key = get_round_key(key_txt)

  count = 0
  printMatrix(ciphertext)

  printMatrix(round_key[:, 40:44]) 

  after_mix = addRoundKey(ciphertext, round_key[:, 40:44])
  printMatrix(after_mix)

  for i in range(40, 4, -4):
    count += 1

    after_shift = reverseShiftRow(after_mix)
    printMatrix(after_shift)

    after_sub = subBytesCiphertext(after_shift)
    printMatrix(after_sub)

    printMatrix(round_key[:, i-4:i]) 

    after_add = addRoundKey(after_sub, round_key[:, i-4:i])
    printMatrix(after_add)

    after_mix = reverseMixColumns(after_add)
    printMatrix(after_mix)


  after_shift = reverseShiftRow(after_mix)
  printMatrix(after_shift)

  after_sub = subBytesCiphertext(after_shift)
  printMatrix(after_sub)

  printMatrix(round_key[:, 0:4]) 

  after_add = addRoundKey(after_sub, round_key[:, 0:4])
  printMatrix(after_add)

  hex_string = ''
  
  for i in after_add.flatten('F'):
    hex_string += '{:02x}'.format(i)

  return hex_string