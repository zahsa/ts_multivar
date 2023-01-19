import os
from classifiers import MLmodels as ml

### Classifying vessel type ###
# whole trajectory
print('\n\nRunning classifiers - Vessel type...')
folders_name= os.listdir("./data/coeffs/")
for filename in folders_name:
    ml.apply_ML(f'./data/coeffs/{filename}/')
    print('\n')

# navigating
print('\n\nRunning classifiers - Navigating - Vessel type...')
folders_name= os.listdir("./data/navigating/coeffs/")
for filename in folders_name:
    ml.apply_ML(f'./data/navigating/coeffs/{filename}/')
    print('\n')

# port
print('\n\nRunning classifiers - Port - Vessel type...')
folders_name= os.listdir("./data/port/coeffs/")
for filename in folders_name:
    ml.apply_ML(f'./data/port/coeffs/{filename}/')
    print('\n')
