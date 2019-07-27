import numpy as np


class Data:

    def __init__(self, y_matrix: np.array, z_matrix: np.array):
        self.y_matrix = np.matrix(y_matrix)
        self.z_matrix = np.matrix(z_matrix)
        self._set_dimensions()
        self._transform()

    def _set_dimensions(self):
        self.n = self.y_matrix.shape[0]
        self.q = self.y_matrix.shape[1]
        self.p = self.z_matrix.shape[1]

    def _transform(self):
        """
        Transform the data into the required form (using eigendecomposition):

        X_trans = L.T * X
        Y_trans = L.T * Y * U
        """
        self._set_ls_matrices()
        self._set_u_matirx()

        self.y_matrix_transform = np.linalg.multi_dot([self.l_matrix.transpose(), self.y_matrix, self.u_matrix])
        self.y_vector_transform = np.reshape(self.y_matrix_transform.transpose(), (self.n * self.q, 1))  # vec(Y)
        self.z_matrix_transform = np.matmul(self.l_matrix.transpose(), self.z_matrix)

    def _set_ls_matrices(self):
        """
        Compute L and S s.t Z*Z_t = L*S*L^T with diag. S and orthogonal L
        """
        zz_t = self.z_matrix * self.z_matrix.transpose()
        l, s, l_t = np.linalg.svd(zz_t)
        s[self.p:] = 0
        self.l_matrix = np.matrix(l)
        self.s_matirx = np.matrix(np.diag(s))

    def get_c_matrix(self, rho=.5):
        """
        Generate equicorrelation matrix

        :param rho: correlation parameter
        :return:
        """
        assert ((rho >= 0) & (rho < 1))
        return np.matrix(
            np.eye(self.q) + rho * (np.ones((self.q, self.q)) - np.eye(self.q))
        )

    def _set_u_matirx(self):
        """
        Generate orthogonal matrix U s.t C=UDU^T and D is diag.
        """
        c_matrix = self.get_c_matrix()
        u_matrix, d_matrix, _ = np.linalg.svd(c_matrix)
        self.u_matrix = np.matrix(u_matrix)

    def get_d_matrix(self, rho):
        """
        Generate the matrix D

        :param rho: correlation parameter
        :return:
        """
        d = [(self.q - 1) * rho + 1] + [(1 - rho) for _ in range(self.q - 1)]  # eigenvalues
        return np.matrix(np.diag(d))
