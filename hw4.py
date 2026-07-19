import numpy as np

class FuzzyPremiseLayer:
  
    def __init__(self, means, sigmas):
        # means و sigmas: آرایه‌های نامپای با ابعاد (2, 3)
        # سطر 0 -> پارامترهای X1 به ترتیب [L1, M1, H1]
        # سطر 1 -> پارامترهای X2 به ترتیب [L2, M2, H2]
        self.means = np.asarray(means, dtype=float)
        self.sigmas = np.asarray(sigmas, dtype=float)

        if self.means.shape != (2, 3) or self.sigmas.shape != (2, 3):
            raise ValueError("ابعاد means و sigmas هر دو باید (2, 3) باشد")

    @staticmethod
    def _gaussian(x, m, sigma):
        # تابع عضویت گوسی: mu_j = exp( -((x - m_j)/sigma_j)^2 )
        z = (x - m) / sigma
        return np.exp(-(z ** 2))

    def forward(self, X):
        # X: لیست یا آرایه نامپای با ابعاد (2,) -> [X1, X2]
        X = np.asarray(X, dtype=float).flatten()
        if X.shape[0] != 2:
            raise ValueError("ورودی X باید دقیقا شامل دو مقدار [X1, X2] باشد")

        X1, X2 = X[0], X[1]

        mu1 = self._gaussian(X1, self.means[0], self.sigmas[0])  # مقادیر عضویت برای [L1, M1, H1]
        mu2 = self._gaussian(X2, self.means[1], self.sigmas[1])  # مقادیر عضویت برای [L2, M2, H2]

        # لایه ضرب (pi-layer): ضرب نظیر به نظیر توابع عضویت برای محاسبه R1 تا R9
        R = np.outer(mu1, mu2).flatten()
        return R


if __name__ == "__main__":
    means = np.array([[2.0, 5.0, 8.0],
                       [1.0, 4.0, 7.0]])
    sigmas = np.array([[1.5, 1.5, 1.5],
                        [1.2, 1.2, 1.2]])

    layer = FuzzyPremiseLayer(means, sigmas)
    R = layer.forward([5.0, 4.0])
    print("R1 تا R9 =", R)
    print("بررسی مجموع R:", R.sum())