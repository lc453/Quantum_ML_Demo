# Import Libraries
import os
import pennylane as qml
from pennylane import numpy as np
from tdc.single_pred import Tox
from rdkit import Chem
#from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA  # to apply PCA
from sklearn.svm import SVC

# Download the dataset from TDCcommons
data = Tox(name='hERG')
split = data.get_split()
train = split['train']
valid = split['valid']
test = split['test']

# Generate molecular fingerprint using RDKit
def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    Convert a SMILES string to a Morgan fingerprint.

    Args:
        smiles: SMILES string representation of a molecule
        radius: The radius of the Morgan fingerprint (default: 2)
        n_bits: The length of the fingerprint bit vector (default: 2048)

    Returns:
        numpy array of shape (n_bits,) containing the fingerprint
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprint = morgan_gen.GetFingerprint(mol)
    # fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fingerprint)

train_fingerprints = np.array([smiles_to_fingerprint(smiles) for smiles in train['Drug'].values])
test_classifications = test['Y'].values
valid_fingerprints = np.array([smiles_to_fingerprint(smiles) for smiles in valid['Drug'].values])
test_fingerprints = np.array([smiles_to_fingerprint(smiles) for smiles in test['Drug'].values])

# Reduce Dimensionality using PCA
pca = PCA(n_components=5)
pca.fit(train_fingerprints)
data_pca = pca.transform(train_fingerprints)
valid_pca = pca.transform(valid_fingerprints)
test_pca = pca.transform(test_fingerprints)

# Compute quantum Kernel with PennyLane
n_qubits = 5
dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev)
def circuit(params):
    qml.AngleEmbedding(features=params, wires=range(n_qubits), rotation='X')
    return qml.state()

@qml.qnode(dev)
def quantum_kernel(x1, x2):
    circuit(x1)
    qml.adjoint(circuit)(x2)
    return qml.probs()

# obtain the kernel matrix
def compute_kernel_matrix(data1, data2=None):
    if data2 is None:
        n_samples = data1.shape[0]
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1,n_samples):
                kernel_matrix[i, j] = (quantum_kernel(data1[i], data1[j])[0])**2
        kernel_matrix += np.conjugate(kernel_matrix.T)
        kernel_matrix += np.identity(n_samples)
        return kernel_matrix

    # compute cross-kernel matrix
    n_samples1 = data1.shape[0]
    n_samples2 = data2.shape[0]
    kernel_matrix = np.zeros((n_samples1, n_samples2))
    for i in range(n_samples1):
        for j in range(n_samples2):
            kernel_matrix[i, j] = (quantum_kernel(data1[i], data2[j])[0])**2
    return kernel_matrix


# Train SVM Classifier with quantum kernel
n_samples = 0
try:
    n_samples = int(input(f"Enter number of training samples (max={train_fingerprints.shape[0]}) - if input is invalid, default will be used: "))
except:
    n_samples = train_fingerprints.shape[0]
if n_samples <= 0 or n_samples > train_fingerprints.shape[0]:
    n_samples = train_fingerprints.shape[0]
train_indeces = np.random.choice(train_fingerprints.shape[0], n_samples, replace=False)
train_subset = data_pca[train_indeces]
train_labels = train['Y'].values[train_indeces]
train_kernel_matrix = compute_kernel_matrix(train_subset)
print("Kernel matrix computed.")

# Train the SVM
clf = SVC(kernel='precomputed')
clf.fit(train_kernel_matrix, train_labels)
print("SVM trained.")

# Evaluate on test set
valid_or_test = ''
while valid_or_test not in ['v', 't', 'V', 'T']:
    valid_or_test = str(input("Evaluate on validation or test set? (v/t): "))[0]
if valid_or_test.lower() == 'v':
    validation_kernel_matrix = compute_kernel_matrix(train_subset, valid_pca)
    print("Validation kernel matrix computed.")
    predictions = clf.predict(validation_kernel_matrix.T)
    count_correct = 0
    for i, prediction  in enumerate(predictions):
        actual = valid['Y'].values[i]
        if prediction == actual:
            count_correct += 1
        print(f"For {valid['Drug_ID'].values[i]}, predicted: {prediction}. Actual: {actual}")
    print(f"Validation accuracy: {count_correct/len(valid)}")
else:
    test_kernel_matrix = compute_kernel_matrix(train_subset, test_pca)
    print("Test kernel matrix computed.")
    predictions = clf.predict(test_kernel_matrix.T)
    count_correct = 0
    for i, prediction  in enumerate(predictions):
        actual = test['Y'].values[i]
        if prediction == actual:
            count_correct += 1
        print(f"For {test['Drug_ID'].values[i]}, predicted: {prediction}. Actual: {actual}")
    print(f"Test accuracy: {count_correct/len(test)}")

_ = input("Press Enter to exit...")