import matplotlib.pyplot as plt
import math


class BaselineRegression:
    def __init__(self, data_points: list[tuple | list]):
        self.data_points = data_points
        self.xs = [j[0] for j in self.data_points]
        self.ys = [j[1] for j in self.data_points]

        self.m = 0
        self.b = None

        self.residuals = []

        self.squared_error = None

    def _calculate_mean(self):
        x_sum = 0
        x_n = len(self.xs)
        for k in self.xs:
            x_sum += k
        x_mean = x_sum / x_n

        y_sum = 0
        y_n = len(self.ys)
        for j in self.ys:
            y_sum += j
        y_mean = y_sum / y_n

        return (x_mean, y_mean)

    def _calculate_residuals(self):
        for i in self.data_points:
            x, y = i[0], i[1]
            pred_y = self.predict(x)
            self.residuals.append((x, pred_y - y))

    def _calculate_squared_error(self):
        s = 0
        for j in self.residuals:
            s += j[1] ** 2
        self.squared_error = s

    def fit(self):
        self.b = self._calculate_mean()[1]
        self._calculate_residuals()
        self._calculate_squared_error()

        return (self.m, self.b)

    def predict(self, x: float | int):
        return self.b

    def plot(self):
        plt.figure(figsize=(10, 6))

        line_xs = [min(self.xs), max(self.xs)]
        line_ys = [self.predict(x) for x in line_xs]

        plt.plot(
            line_xs,
            line_ys,
            color="orange",
            label=f"LSR: y={self.m:.2f}x + {self.b:.2f}",
        )

        plt.scatter(self.xs, self.ys, label="Original Data")

        for x, y in zip(self.xs, self.ys):
            pred_y = self.predict(x)
            plt.plot([x, x], [y, pred_y], color="red", linestyle="--", alpha=0.5)

        plt.title("Least Squares Regression")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


class LSR:
    def __init__(self, data_points: list[tuple | list]):
        self.data_points = data_points
        self.xs = [j[0] for j in self.data_points]
        self.ys = [j[1] for j in self.data_points]

        self.baseline_model = BaselineRegression(self.data_points)

        self.mean = None
        self.x_mean = None
        self.y_mean = None

        self.variance = None
        self.x_variance = None
        self.y_variance = None

        self.std_deviation = None
        self.x_std_deviation = None
        self.y_std_deviation = None

        self.r = None
        self.r_squared = None

        self.m = None
        self.b = None

        self.residuals = []

        self.squared_error = None

    def _calculate_mean(self):
        x_sum = 0
        x_n = len(self.xs)
        for k in self.xs:
            x_sum += k
        x_mean = x_sum / x_n

        y_sum = 0
        y_n = len(self.ys)
        for j in self.ys:
            y_sum += j
        y_mean = y_sum / y_n

        return (x_mean, y_mean)

    def _calculate_variance(self):
        # x
        x_n = len(self.xs)
        x_mean_dist_sum = 0
        for x in self.xs:
            x_mean_dist_sum += (x - self.x_mean) ** 2
        x_variance = x_mean_dist_sum / (x_n - 1)  # applies bessel's correction

        # y
        y_n = len(self.ys)
        y_mean_dist_sum = 0
        for y in self.ys:
            y_mean_dist_sum += (y - self.y_mean) ** 2
        y_variance = y_mean_dist_sum / (y_n - 1)  # applies bessel's correction

        return (x_variance, y_variance)

    def _calculate_std_deviation(self):
        # x
        x_std_deviation = math.sqrt(self.x_variance)
        y_std_deviation = math.sqrt(self.y_variance)

        return (x_std_deviation, y_std_deviation)

    def _calculate_zscore(
        self, i: float | int, mean: float | int, std_dev: float | int
    ):
        """
        Calculates the z-score for a single point

        i: Data point
        mean: Mean
        std_dev: Standard deviation
        """
        return (i - mean) / std_dev

    def _calculate_r(self):
        """
        Calculates correlation coefficient for the dataset
        """

        n = len(self.data_points)

        correlation_sum = 0
        for i in self.data_points:
            x, y = i[0], i[1]

            x_zscore = self._calculate_zscore(x, self.x_mean, self.x_std_deviation)
            y_zscore = self._calculate_zscore(y, self.y_mean, self.y_std_deviation)

            correlation_sum += x_zscore * y_zscore

        self.r = correlation_sum / (n - 1)

    def _calculate_residuals(self):
        for i in self.data_points:
            x, y = i[0], i[1]
            pred_y = self.predict(x)
            self.residuals.append((x, pred_y - y))

    def _calculate_squared_error(self):
        s = 0
        for j in self.residuals:
            s += j[1] ** 2
        self.squared_error = s

    def _calculate_r_squared(self):
        self.r_squared = 1 - (self.squared_error / self.baseline_model.squared_error)

    def fit_baseline(self):
        self.baseline_model.fit()
        print("Succesfully fitted Baseline Model.")

    def fit(self):
        """
        Fits the Least Squares Regression model by calculating the means, variances, standard deviations, the correlation coefficient and finally, the slope (m) and intercept (b). All of this of course, based on the data points given upon the object's instantiation.
        """
        self.fit_baseline()

        self.mean = self._calculate_mean()
        self.x_mean = self.mean[0]
        self.y_mean = self.mean[1]
        print(f"Succesfully found mean - x: {self.x_mean} | y: {self.y_mean}")

        self.variance = self._calculate_variance()
        self.x_variance = self.variance[0]
        self.y_variance = self.variance[1]
        print(
            f"Succesfully found variance - x: {self.x_variance} | y: {self.y_variance}"
        )

        self.std_deviation = self._calculate_std_deviation()
        self.x_std_deviation = self.std_deviation[0]
        self.y_std_deviation = self.std_deviation[1]
        print(
            f"Succesfully found standard deviation - x: {self.x_std_deviation} | y: {self.y_std_deviation}"
        )

        self._calculate_r()
        print(f"Succesfully found correlation coefficient: {self.r}")

        self.m = self.r * (self.y_std_deviation / self.x_std_deviation)
        self.b = self.y_mean - (self.m * self.x_mean)

        print(f"Succesfully found m and b - m: {self.m} | b: {self.b}")

        self._calculate_residuals()
        self._calculate_squared_error()
        self._calculate_r_squared()

        print(f"R² = {self.r_squared}")

        return (self.m, self.b)

    def predict(self, x: float | int):
        return self.m * x + self.b

    def plot(self):
        plt.figure(figsize=(10, 6))

        line_xs = [min(self.xs), max(self.xs)]
        line_ys = [self.predict(x) for x in line_xs]

        plt.plot(
            line_xs,
            line_ys,
            color="orange",
            label=f"LSR: y={self.m:.2f}x + {self.b:.2f}",
        )

        plt.scatter(self.xs, self.ys, label="Original Data")

        for x, y in zip(self.xs, self.ys):
            pred_y = self.predict(x)
            plt.plot([x, x], [y, pred_y], color="red", linestyle="--", alpha=0.5)

        plt.title("Least Squares Regression")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
