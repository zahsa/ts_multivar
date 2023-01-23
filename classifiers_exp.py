import os
from classifiers import MLmodels as ml

### Classifying vessel type ###
# whole trajectory
print('\n\nRunning classifiers - Vessel type...')
folders_name = os.listdir("./data/coeffs/")
for filename in folders_name:
    ml.apply_ML_vt(f'./data/coeffs/{filename}/')
    print('\n')

# navigating
print('\n\nRunning classifiers - Navigating - Vessel type...')
folders_name = os.listdir("./data/navigating/coeffs/")
for filename in folders_name:
    ml.apply_ML_vt(f'./data/navigating/coeffs/{filename}/')
    print('\n')

# port
print('\n\nRunning classifiers - Port - Vessel type...')
folders_name = os.listdir("./data/port/coeffs/")
for filename in folders_name:
    ml.apply_ML_vt(f'./data/port/coeffs/{filename}/')
    print('\n')


# navigating vs port
print('\n\nRunning classifiers - Port vs Navigating')
folders_name = os.listdir("./data/join/coeffs/")
for filename in folders_name:
    ml.apply_ML_np(f'./data/join/coeffs/{filename}/', port=False)
    print('\n')

# print('\n\nRunning classifiers - Specific Port vs Navigating')
# folders_name = os.listdir("./data/join/coeffs/")
# for filename in folders_name:
#     ml.apply_ML_np(f'./data/join/coeffs/{filename}/', port=True)
#     print('\n')

# port labels
# print('\n\nRunning classifiers Specific Port')
# folders_name= os.listdir("./data/port/coeffs/")
# for filename in folders_name:
#     ml.apply_ML_vt(f'./data/port/coeffs/{filename}/', label=True)
#     print('\n')