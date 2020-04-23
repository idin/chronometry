from .get_elapsed import get_elapsed
from .get_elapsed import get_now
from .progress import ProgressBar
from sklearn.linear_model import LinearRegression
from pandas import DataFrame


class Measurement:
	def __init__(self, x, result, elapsed):
		self._x = x
		self._result = result
		self._elapsed = elapsed

	def __lt__(self, other):
		return self.x < other.x

	def __le__(self, other):
		return self.x <= other.x

	def __eq__(self, other):
		return self.x == other.x

	def __gt__(self, other):
		return self.x > other.x

	def __ge__(self, other):
		return self.x >= other.x

	def __ne__(self, other):
		return self.x != other.x

	@property
	def x(self):
		return self._x

	@property
	def elapsed(self):
		return self._elapsed

	@property
	def dictionary(self):
		if isinstance(self._result, dict) and 'elapsed' not in self._result and 'x' not in self._result:
			return dict(x=self._x, elapsed=self.elapsed, **self._result)
		else:
			return {'x': self._x, 'elapsed': self.elapsed, 'result': self._result}


# Estimator gets a single argument function and estimates the time it takes to run the function based on the argument
# the function should accept an int larger than 0
class Estimator:
	def __init__(self, function, unit='s', polynomial_degree=2):
		self._function = function
		self._unit = unit
		self._measurements = {}
		self._polynomial_degree = polynomial_degree
		self._model = None
		self._reverse_model = None
		self._max_x = 0

	def measure(self, x):
		"""
		:type x: int
		:rtype: Measurement
		"""
		if x in self._measurements:
			return self._measurements[x]
		else:
			start_time = get_now()
			result = self._function(x)
			elapsed = get_elapsed(start=start_time, unit=self._unit)
			measurement = Measurement(x=x, result=result, elapsed=elapsed)
			self._measurements[x] = measurement
			self._model = None
			self._reverse_model = None
			self._max_x = max(self._max_x, x)
			return measurement

	@property
	def data(self):
		"""
		:rtype: DataFrame
		"""
		return DataFrame.from_records(
			[measurement.dictionary for measurement in self.measurements]
		)

	@property
	def measurements(self):
		"""
		:rtype: list[Measurement]
		"""
		return sorted(list(self._measurements.values()))

	def get_X(self, x):
		"""
		:rtype: DataFrame
		"""
		if not isinstance(x, (list, tuple)):
			x = [x]
		x_data = DataFrame({'x': x})
		for i in range(1, self._polynomial_degree):
			x_data[f'x_{i + 1}'] = x_data['x'] ** (i + 1)
		return x_data

	@property
	def training_X(self):
		"""
		:rtype: DataFrame
		"""
		x = []
		for measurement in self.measurements:
			x += [measurement.x] * round(measurement.x)

		return self.get_X(x)

	@property
	def training_y(self):
		"""
		:rtype: list
		"""
		y = []
		for measurement in self.measurements:
			y += [measurement.elapsed] * round(measurement.x)
		return y

	@property
	def model(self):
		"""
		:rtype: LinearRegression
		"""
		if self._model is None:
			self._model = LinearRegression()
			self._model.fit(self.training_X, self.training_y)
		return self._model

	def predict(self, x):
		"""
		:rtype: list or float or int
		"""
		if not isinstance(x, (list, tuple)) and x in self._measurements:
			return self.measure(x=x).elapsed

		x_data = self.get_X(x=x)
		predictions = self.model.predict(x_data)
		if isinstance(x, (list, tuple)):
			return list(predictions)
		else:
			return list(predictions)[0]

	def get_next_x_and_next_estimate(self, x, max_x, max_time, start_time):
		next_x = min(int(round(x * 1.5) + 1), max_x)
		next_estimate = self.predict(next_x)
		total_time = get_elapsed(start=start_time, unit=self._unit)
		remaining_time = max_time - total_time
		while next_estimate > remaining_time * 1.1 and next_x > self._max_x + 1:
			next_x = min(int(round((next_x + self._max_x) / 2)), max_x)
			next_estimate = self.predict(next_x)

			total_time = get_elapsed(start=start_time, unit=self._unit)
			remaining_time = max_time - total_time
		return next_x, next_estimate

	def estimate(self, max_x, max_time):
		start_time = get_now()
		total_time = get_elapsed(start=start_time, unit=self._unit)
		x = self._max_x + 1
		min_data_points = 1 + self._polynomial_degree

		progress_bar = ProgressBar(total=min_data_points)

		progress_amount = 0
		while len(self.measurements) < min_data_points:
			progress_bar.show(amount=progress_amount, text=f'part 1: next x = {x}')
			self.measure(x=x)
			progress_amount += x
			x = int(round(x * 1.6 + 1))
			progress_bar.set_total(total=progress_amount + x)
			total_time = get_elapsed(start=start_time, unit=self._unit)

		progress_bar.show(amount=progress_amount, text=f'part 2: next x = {x}')
		x, next_estimate = self.get_next_x_and_next_estimate(
			x=x, max_x=max_x, max_time=max_time, start_time=start_time
		)
		while total_time + next_estimate < max_time and x <= max_x:
			progress_bar.show(amount=progress_amount, text=f'part 2: x = {x}, estimate: {round(next_estimate, 1)}{self._unit}')
			self.measure(x=x)
			progress_amount += x
			if x == max_x:
				break

			next_x, next_estimate = self.get_next_x_and_next_estimate(
				x=x, max_x=max_x, max_time=max_time, start_time=start_time
			)
			if next_x == x:
				break
			x = next_x
			progress_bar.set_total(total=progress_amount + x)
			total_time = get_elapsed(start=start_time, unit=self._unit)

		progress_amount += x
		progress_bar.show(amount=progress_amount)
		return self.predict(max_x)

	def plot(self):
		return self.data.plot(x='x', y='elapsed')
