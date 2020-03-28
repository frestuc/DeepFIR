import numpy as np
import pickle as pkl

num_examples_per_class = 20
num_classes = 3

indexes_start = np.array(range(num_examples_per_class)) # this goes from 0 to 106495
indexes = indexes_start

train_idx_baseline = int(len(indexes) * .54)
valid_idx_baseline = int(len(indexes) * .60)
train_idx_FIR = int(len(indexes) * .87)
valid_idx_FIR = int(len(indexes) * .90)

train_indexes_baseline = indexes[:train_idx_baseline]
valid_indexes_baseline = indexes[train_idx_baseline:valid_idx_baseline]
train_indexes_FIR = indexes[valid_idx_baseline:train_idx_FIR]
valid_indexes_FIR= indexes[train_idx_FIR:valid_idx_FIR]
test_indexes = indexes[valid_idx_FIR:]

print('LOL')
print(train_indexes_baseline)
print(train_indexes_FIR)
print(valid_indexes_baseline)
print(valid_indexes_FIR)
print(test_indexes)

train_indexes_BL_1 = train_indexes_baseline
train_indexes_FIR_1 = train_indexes_FIR
valid_indexes_BL_1 = valid_indexes_baseline
valid_indexes_FIR_1 = valid_indexes_FIR
test_indexes_1 = test_indexes

print('ASD')
print(train_indexes_baseline)
print(train_indexes_FIR)
print(valid_indexes_baseline)
print(valid_indexes_FIR)
print(test_indexes)

# expand this shuffling indexing
for i in range(num_classes - 1):
    train_indexes_BL_1 = np.append(train_indexes_BL_1,[x + (i + 1) * num_examples_per_class for x in train_indexes_baseline])
    train_indexes_FIR_1 = np.append(train_indexes_FIR_1,[x + (i + 1) * num_examples_per_class for x in train_indexes_FIR])
    valid_indexes_BL_1 = np.append(valid_indexes_BL_1,[x + (i + 1) * num_examples_per_class for x in valid_indexes_baseline])
    valid_indexes_FIR_1 = np.append(valid_indexes_FIR_1,[x + (i + 1) * num_examples_per_class for x in valid_indexes_FIR])
    test_indexes_1 = np.append(test_indexes_1,[x + (i + 1) * num_examples_per_class for x in test_indexes])

print('RoFTL')
print(train_indexes_BL_1)
print(train_indexes_FIR_1)
print(valid_indexes_BL_1)
print(valid_indexes_FIR_1)
print(test_indexes_1)



print(len(indexes))
print(len(train_indexes_BL_1) + len(train_indexes_FIR_1) + len(valid_indexes_BL_1) + len(valid_indexes_FIR_1) + len(test_indexes_1))

# Saving the objects:
with open('indexes.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pkl.dump([train_indexes_BL_1,
                 train_indexes_FIR_1,
                 valid_indexes_BL_1,
                 valid_indexes_FIR_1,
                 test_indexes_1], f)

# Getting back the objects:
with open('indexes.pkl','rb') as f:  # Python 3: open(..., 'rb')
    train_indexes_BL_2, train_indexes_FIR_2, valid_indexes_BL_2, valid_indexes_FIR_2, test_indexes_2 = pkl.load(f)

print('RoFTL2')
print(train_indexes_BL_2)
print(train_indexes_FIR_2)
print(valid_indexes_BL_2)
print(valid_indexes_FIR_2)
print(test_indexes_2)