import numpy as np
class DataLoader:
    @classmethod
    def load(cls, filename):
        file = open(filename, mode='r')

        whole_text = file.read()
        whole_text = whole_text.strip()
        lines = whole_text.split('\n')

        n = int(lines[0].strip())

        distances = DataLoader.toMatrix(lines[2:2+n])
        flows = DataLoader.toMatrix(lines[3+n:])

        # print(distances)
        # print('')
        # print(flows)

        return n, distances, flows

    @classmethod
    def toMatrix(cls, lines):
        rows = []
        for line in lines:
            row = list(map(lambda x: float(x), line.split()))
            rows.append(row)

        matrix = np.asarray(rows)
        return matrix