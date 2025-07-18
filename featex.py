from collections import Counter
import math
import pandas as pd
import joblib
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
from itertools import product
from io import StringIO

# =========================
# Feature Extraction Functions
# =========================


def AAC(fastas, **kw):

    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    header = [f'AAC_{aa}' for aa in AA]

    for record in fastas:
        # Remove gap characters and convert to uppercase.
        sequence = str(record.seq).replace('-', '').upper()
        if len(sequence) == 0:
            # Return a zero vector for empty sequences.
            code = [0.0] * len(AA)
        else:
            count = Counter(sequence)
            code = [count.get(aa, 0) / len(sequence) for aa in AA]
        encodings.append(code)
    return np.array(encodings, dtype=float), header




def PAAC(fastas, lambdaValue=1, w=0.05, **kw):
    # Hard-coded property records.
    records = [
        "#   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V",
        "Hydrophobicity  0.62    -2.53   -0.78   -0.9    0.29    -0.85   -0.74   0.48    -0.4    1.38    1.06    -1.5    0.64    1.19    0.12    -0.18   -0.05   0.81    0.26    1.08",
        "Hydrophilicity  -0.5    3   0.2   3   -1   0.2   3   0   -0.5    -1.8    -1.8    3   -1.3    -2.5    0   0.3   -0.4    -3.4    -2.3    -1.5"
    ]

    # Extract amino acids from the header line.
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {aa: idx for idx, aa in enumerate(AA)}
    AAProperty = []
    AAPropertyNames = []

    # Parse property records.
    for line in records[1:]:
        parts = line.rstrip().split()
        if parts:
            AAProperty.append([float(x) for x in parts[1:]])
            AAPropertyNames.append(parts[0])

    # Normalize the AA properties using z-score normalization.
    AAProperty1 = []
    for prop in AAProperty:
        meanI = sum(prop) / len(prop)
        fenmu = math.sqrt(sum([(x - meanI) ** 2 for x in prop]) / len(prop))
        if fenmu == 0:
            normalized_prop = [0.0 for x in prop]
        else:
            normalized_prop = [(x - meanI) / fenmu for x in prop]
        AAProperty1.append(normalized_prop)

    encodings = []
    header = [f'PAAC_Xc1_{aa}' for aa in AA]
    for j in range(1, lambdaValue + 1):
        header.append(f'PAAC_Xc2_lambda{j}')

    for record in fastas:
        sequence = str(record.seq).replace('-', '').upper()
        code = []
        theta = []

        # Compute theta values for lag distances from 1 to lambdaValue.
        for n in range(1, lambdaValue + 1):
            sum_theta = 0.0
            if len(sequence) > n:
                for j in range(len(AAProperty1)):
                    # Compute the average product of property values for positions separated by n.
                    valid_values = [
                        AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]]
                        for k in range(len(sequence) - n)
                        if sequence[k] in AADict and sequence[k + n] in AADict
                    ]
                    if valid_values:
                        sum_theta += sum(valid_values) / (len(sequence) - n)
            theta.append(sum_theta)

        # Calculate amino acid composition.
        myDict = {aa: sequence.count(aa) for aa in AA}
        total_theta = sum(theta)
        if total_theta == 0:
            total_theta = 1  # Avoid division by zero.

        # Composition features: normalized frequencies.
        code += [myDict[aa] / (1 + w * total_theta) for aa in AA]
        # Sequence-order features: weighted theta values.
        code += [w * value / (1 + w * total_theta) for value in theta]

        encodings.append(code)

    return np.array(encodings, dtype=float), header


def APAAC(fastas, lambdaValue=1, w=0.05, **kw):
    records = [
        "#   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V",
        "Hydrophobicity  0.62    -2.53   -0.78   -0.9    0.29    -0.85   -0.74   0.48    -0.4    1.38    1.06    -1.5    0.64    1.19    0.12    -0.18   -0.05   0.81    0.26    1.08",
        "Hydrophilicity  -0.5    3   0.2   3   -1   0.2   3   0   -0.5    -1.8    -1.8    3   -1.3    -2.5    0   0.3   -0.4    -3.4    -2.3    -1.5"
    ]

    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {aa: idx for idx, aa in enumerate(AA)}
    AAProperty = []
    AAPropertyNames = []

    for line in records[1:]:
        parts = line.rstrip().split()
        if parts:
            AAProperty.append([float(x) for x in parts[1:]])
            AAPropertyNames.append(parts[0])

    # Normalize the AA properties using z-score normalization.
    AAProperty1 = []
    for prop in AAProperty:
        meanI = sum(prop) / len(prop)
        fenmu = math.sqrt(sum([(x - meanI) ** 2 for x in prop]) / len(prop))
        if fenmu == 0:
            normalized_prop = [0.0 for x in prop]
        else:
            normalized_prop = [(x - meanI) / fenmu for x in prop]
        AAProperty1.append(normalized_prop)

    encodings = []
    header = [f'APAAC_Pc1_{aa}' for aa in AA]
    # Create headers for amphiphilic features.
    for j in range(1, lambdaValue + 1):
        for name in AAPropertyNames:
            header.append(f'APAAC_Pc2.{name}.{j}')

    for record in fastas:
        sequence = str(record.seq).replace('-', '').upper()
        code = []
        theta = []

        # Compute theta values for each property and lag.
        for j, prop in enumerate(AAProperty1):
            for n in range(1, lambdaValue + 1):
                sum_theta = 0.0
                if len(sequence) > n:
                    valid_values = []
                    for k in range(len(sequence) - n):
                        aa1 = sequence[k]
                        aa2 = sequence[k + n]
                        if aa1 in AADict and aa2 in AADict:
                            valid_values.append(prop[AADict[aa1]] * prop[AADict[aa2]])
                    if valid_values:
                        sum_theta = sum(valid_values) / (len(sequence) - n)
                theta.append(sum_theta)

        myDict = {aa: sequence.count(aa) for aa in AA}
        total_theta = sum(theta)
        if total_theta == 0:
            total_theta = 1  # Avoid division by zero.

        # Composition features: normalized amino acid frequencies.
        code += [myDict[aa] / (1 + w * total_theta) for aa in AA]
        # Amphiphilic (sequence-order) features.
        code += [w * value / (1 + w * total_theta) for value in theta]

        encodings.append(code)

    return np.array(encodings, dtype=float), header

