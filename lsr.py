import matplotlib.pyplot as plt
import math


class LSR:
    def __init__(self, data_points: list[tuple | list]):
        self.data_points = data_points
        self.xs = [j[0] for j in self.data_points]
        self.ys = [j[1] for j in self.data_points]

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

        self.m = None
        self.b = None

        self.residuals = []

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
        x_variance = x_mean_dist_sum / (
            x_n - 1
        )  # Applies Bessel's correction, since this is sample-based

        # y
        y_n = len(self.ys)
        y_mean_dist_sum = 0
        for y in self.ys:
            y_mean_dist_sum += (y - self.y_mean) ** 2
        y_variance = y_mean_dist_sum / (
            y_n - 1
        )  # Applies Bessel's correction, since this is sample-based

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

        # Calculate the sum of the z-scores products
        correlation_sum = 0
        for i in self.data_points:
            x, y = i[0], i[1]

            x_zscore = self._calculate_zscore(x, self.x_mean, self.x_std_deviation)
            y_zscore = self._calculate_zscore(y, self.y_mean, self.y_std_deviation)

            correlation_sum += x_zscore * y_zscore

        return correlation_sum / (n - 1)

    def _calculate_residuals(self):
        for x in self.xs:
            pred_y = self.predict(x)
            self.residuals.append((x, pred_y - x))

    def fit(self):
        """
        Fits the Least Squares Regression model by calculating the means, variances, standard deviations, the correlation coefficient and finally, the slope (m) and intercept (b). All of this of course, based on the data points given upon the object's instantiation.
        """
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

        self.r = self._calculate_r()
        print(f"Succesfully found correlation coefficient: {self.r}")

        self.m = self.r * (self.y_std_deviation / self.x_std_deviation)

        # By simple algebra, (x_mean, y_mean) is a known existing data point on the line
        # ref_point = (self.x_mean, self.y_mean)
        self.b = self.y_mean - (self.m * self.x_mean)

        print(f"Succesfully found m and b - m: {self.m} | b: {self.b}")

        self._calculate_residuals()

        return (self.m, self.b)

    def predict(self, x: float | int):
        return self.m * x + self.b

    def plot(
        self, with_original_points: bool = True, predict: int | float | None = None
    ):
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


my_model = LSR([(1, 1), (2, 2), (2, 3), (3, 6)])
my_model.fit()
print(my_model.predict(3))
my_model.plot()

# Dataset to test the model, simulates study hours vs test scores
my_model_2 = LSR(
    [
        (1.5, 45.2),
        (2.0, 50.5),
        (2.3, 52.8),
        (3.0, 58.0),
        (3.2, 60.1),
        (3.5, 62.4),
        (4.0, 68.3),
        (4.5, 71.0),
        (5.0, 75.2),
        (5.2, 74.8),
        (5.5, 80.1),
        (6.0, 82.5),
        (6.1, 84.0),
        (6.5, 86.2),
        (7.0, 89.5),
        (7.5, 92.1),
        (8.0, 94.0),
        (8.2, 93.5),
        (9.0, 96.8),
        (9.5, 98.2),
    ]
)
my_model_2.fit()
print(my_model_2.predict(3))
my_model_2.plot()
