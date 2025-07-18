import numpy as np
import math
import re
from collections import Counter

def AAC(fastas, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    header = list(AA)
    for name, seq in fastas:
        sequence = re.sub('-', '', seq)
        count = Counter(sequence)
        total = len(sequence)
        code = [count.get(aa, 0) / total if total > 0 else 0 for aa in AA]
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def APAAC(fastas, lambdaValue=30, w=0.05, **kw):
    records = [
        "#   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V",
        "Hydrophobicity  0.62 -2.53 -0.78 -0.9 0.29 -0.85 -0.74 0.48 -0.4 1.38 1.06 -1.5 0.64 1.19 0.12 -0.18 -0.05 0.81 0.26 1.08",
        "Hydrophilicity  -0.5 3 0.2 3 -1 0.2 3 0 -0.5 -1.8 -1.8 3 -1.3 -2.5 0 0.3 -0.4 -3.4 -2.3 -1.5",
        "SideChainMass   15 101 58 59 47 72 73 1 82 57 57 73 75 91 42 31 45 130 107 43"
    ]
    AA = ''.join(records[0].split()[1:])
    AADict = {aa: i for i, aa in enumerate(AA)}
    AAProperty = []
    AAPropertyNames = []
    for line in records[1:]:
        arr = line.rstrip().split()
        AAProperty.append([float(j) for j in arr[1:]])
        AAPropertyNames.append(arr[0])
    # Normalize
    AAProperty1 = []
    for arr in AAProperty:
        mean_val = sum(arr) / 20
        std = math.sqrt(sum([(j - mean_val) ** 2 for j in arr]) / 20)
        AAProperty1.append([(j - mean_val) / std for j in arr])

    encodings = []
    header = ['Pc1.' + aa for aa in AA]
    for j in range(1, lambdaValue + 1):
        for pname in AAPropertyNames:
            header.append('Pc2.' + pname + '.' + str(j))

    for name, seq in fastas:
        sequence = re.sub('-', '', seq)
        L = len(sequence)
        code = []
        theta = []
        # ป้องกัน sequence สั้น
        for n in range(1, lambdaValue + 1):
            if L - n == 0:
                theta.extend([0.0] * len(AAProperty1))
                continue
            for j in range(len(AAProperty1)):
                th = sum([AAProperty1[j][AADict.get(sequence[k], 0)] * AAProperty1[j][AADict.get(sequence[k + n], 0)]
                         for k in range(L - n)]) / (L - n)
                theta.append(th)
        myDict = Counter(sequence)
        code += [myDict.get(aa, 0) / (1 + w * sum(theta)) if (1 + w * sum(theta)) != 0 else 0 for aa in AA]
        code += [w * val / (1 + w * sum(theta)) if (1 + w * sum(theta)) != 0 else 0 for val in theta]
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def DPC(fastas, gap=0, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    AADict = {aa: i for i, aa in enumerate(AA)}
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    encodings = []
    header = diPeptides
    for name, seq in fastas:
        sequence = re.sub('-', '', seq)
        L = len(sequence)
        tmpCode = [0] * 400
        for j in range(L - gap - 1):
            aa1 = sequence[j]
            aa2 = sequence[j + gap + 1]
            if aa1 in AADict and aa2 in AADict:
                idx = AADict[aa1] * 20 + AADict[aa2]
                tmpCode[idx] += 1
        total = sum(tmpCode)
        code = [i / total if total > 0 else 0 for i in tmpCode]
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def Rvalue(aa1, aa2, AADict, Matrix):
    return sum([(Matrix[i][AADict.get(aa1, 0)] - Matrix[i][AADict.get(aa2, 0)]) ** 2 for i in range(len(Matrix))]) / len(Matrix)

def PAAC(fastas, lambdaValue=30, w=0.05, **kw):
    records = [
        "#   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V",
        "Hydrophobicity  0.62 -2.53 -0.78 -0.9 0.29 -0.85 -0.74 0.48 -0.4 1.38 1.06 -1.5 0.64 1.19 0.12 -0.18 -0.05 0.81 0.26 1.08",
        "Hydrophilicity  -0.5 3 0.2 3 -1 0.2 3 0 -0.5 -1.8 -1.8 3 -1.3 -2.5 0 0.3 -0.4 -3.4 -2.3 -1.5",
        "SideChainMass   15 101 58 59 47 72 73 1 82 57 57 73 75 91 42 31 45 130 107 43"
    ]
    AA = ''.join(records[0].split()[1:])
    AADict = {aa: i for i, aa in enumerate(AA)}
    AAProperty = []
    AAPropertyNames = []
    for line in records[1:]:
        arr = line.rstrip().split()
        AAProperty.append([float(j) for j in arr[1:]])
        AAPropertyNames.append(arr[0])
    # Normalize
    AAProperty1 = []
    for arr in AAProperty:
        mean_val = sum(arr) / 20
        std = math.sqrt(sum([(j - mean_val) ** 2 for j in arr]) / 20)
        AAProperty1.append([(j - mean_val) / std for j in arr])
    encodings = []
    header = ['Xc1.' + aa for aa in AA]
    for n in range(1, lambdaValue + 1):
        header.append('Xc2.lambda' + str(n))
    for name, seq in fastas:
        sequence = re.sub('-', '', seq)
        L = len(sequence)
        code = []
        theta = []
        for n in range(1, lambdaValue + 1):
            if L - n == 0:
                theta.append(0.0)
                continue
            th = sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1)
                      for j in range(L - n)]) / (L - n)
            theta.append(th)
        myDict = Counter(sequence)
        code += [myDict.get(aa, 0) / (1 + w * sum(theta)) if (1 + w * sum(theta)) != 0 else 0 for aa in AA]
        code += [(w * t) / (1 + w * sum(theta)) if (1 + w * sum(theta)) != 0 else 0 for t in theta]
        encodings.append(code)
    return np.array(encodings, dtype=float), header
