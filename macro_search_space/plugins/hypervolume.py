#  assume 2 obj
import numpy as np


class HV_2D:
    def __init__(self, ref_point, pareto_front):
        self.ref_point = ref_point
        self.pareto_front = pareto_front

    def calculate(self):
        n_data_points = len(self.pareto_front)
        f1 = np.zeros(n_data_points)
        f2 = np.zeros(n_data_points)
        for i in range(n_data_points):
            f1[i] = self.pareto_front[i][0]
            f2[i] = self.pareto_front[i][1]
        idx = np.argsort(f1)
        f1_sorted = f1[idx]
        f2_sorted = f2[idx]

        b1 = self.ref_point[0] - f1_sorted
        b2 = self.ref_point[1] - f2_sorted

        n_hv = 0
        hv_set = np.zeros((n_data_points, 2))
        for i in range(n_data_points):
            if (b1[i] > 0) and (b2[i] > 0):
                hv_set[n_hv, 0] = f1_sorted[i]
                hv_set[n_hv, 1] = f2_sorted[i]
                n_hv += 1

        if n_hv > 0:
            hv_set_y1 = np.append(self.ref_point[1], hv_set[0:n_hv - 1, 1])
            x_diff = self.ref_point[0] - hv_set[0:n_hv, 0]
            y_diff = hv_set_y1 - hv_set[0:n_hv, 1]
            area_array = np.multiply(x_diff, y_diff)
            hv_value = np.sum(area_array)

        area_array2 = np.zeros((n_hv, 1))
        for i in range(n_hv):
            if i == 0:
                area_array2[i] = (self.ref_point[0] - hv_set[i, 0]) * (
                        self.ref_point[1] - hv_set[i, 1])
            else:
                area_array2[i] = (self.ref_point[0] - hv_set[i, 0]) * (
                        hv_set[i - 1, 1] - hv_set[i, 1])
        hv_value2 = np.sum(area_array2)

        return hv_value, hv_value2
