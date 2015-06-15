import numpy as np

class HilbertCurve(object):
    """Hilbert curve function.

    Pre-calculates the Hilbert space filling curve with a given number
    of iterations. The curve will lie in the square delimited by the
    points (0, 0) and (1, 1).

    Arguments
    ---------
    n : int
        Iterations.
    """
    # Implementation based on
    # http://en.wikipedia.org/w/index.php?title=Hilbert_curve&oldid=633637210

    def __init__(self, n):
        self.n = n
        self.n_corners = (2 ** n) ** 2
        self.corners = np.zeros((self.n_corners, 2))
        self.steps = np.arange(self.n_corners)

        steps = np.arange(self.n_corners)
        for s in 2 ** np.arange(n):
            r = np.empty_like(self.corners, dtype='int')
            r[:, 0] = 1 & (steps // 2)
            r[:, 1] = 1 & (steps ^ r[:, 0])
            self._rot(s, r)
            self.corners += s * r
            steps //= 4

        self.corners /= (2 ** n) - 1

    def _rot(self, s, r):
        swap = r[:, 1] == 0
        flip = np.all(r == np.array([1, 0]), axis=1)

        self.corners[flip] = (s - 1 - self.corners[flip])
        self.corners[swap] = self.corners[swap, ::-1]

    def __call__(self, u):
        """Evaluate pre-calculated Hilbert curve.

        Arguments
        ---------
        u : ndarray (M,)
            Positions to evaluate on the curve in the range [0, 1].

        Returns
        -------
        ndarray (M, 2)
            Two-dimensional curve coordinates.
        """
        step = np.asarray(u * len(self.steps))
        return np.vstack((
            np.interp(step, self.steps, self.corners[:, 0]),
            np.interp(step, self.steps, self.corners[:, 1]))).T
