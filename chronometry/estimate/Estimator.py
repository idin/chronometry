
from slytherin.hash import hash_object
from slytherin.functions import get_function_arguments
from ravenclaw.preprocessing import Polynomial, Normalizer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from pandas import DataFrame, concat
from random import randint, random, choice
from func_timeout import func_timeout, FunctionTimedOut
import matplotlib.pyplot as plt
from numpy import where

from .create_arguments import create_arguments
from .Measurement import Measurement
from ..time import get_elapsed
from ..time import get_now
from ..progress import ProgressBar


# Estimator gets a single argument function and estimates the time it takes to run the function based on the argument
# the function should accept an int larger than 0
class Estimator:
	def __init__(self, function, args=None, unit='s', polynomial_degree=2, timeout=20):
		self._function = function
		self._function_arguments = get_function_arguments(function=function)
		self._unit = unit
		self._measurements = {}
		self._polynomial_degree = polynomial_degree
		self._model = None
		self._normalizer = None
		self._error_model = None
		self._max_x = None
		self._args = args
		self._timeout = timeout
		self._num_errors = 0
		self._num_regular_runs = 0
		self._num_timeouts = 0
		self._x_data_columns = {}
		self._error_x_data_columns = {}

	@staticmethod
	def get_key(**kwargs):
		return hash_object(kwargs)

	def check_arguments(self, kwargs, method_name):
		unknown_arguments = [key for key in kwargs.keys() if key not in self._function_arguments]
		missing_arguments = [key for key in self._function_arguments if key not in kwargs]

		if len(missing_arguments) == 1:
			return f'{method_name}() is missing the argument "{missing_arguments[0]}"'
		elif len(missing_arguments) > 1:
			arguments_string = '", "'.join(missing_arguments)
			return f'{method_name}() is missing arguments "{arguments_string}"'

		if len(unknown_arguments) == 0:
			return False
		elif len(unknown_arguments) == 1:
			return f'{method_name}() got an unexpected argument "{unknown_arguments[0]}"'
		else:
			arguments_string = '", "'.join(unknown_arguments)
			return f'{method_name}() got unexpected arguments "{arguments_string}"'

	def get_arguments(self, arguments, **kwargs):
		if arguments is None and len(kwargs) == 0:
			raise ValueError('either arguments should be provided or kwargs!')
		elif arguments is not None and len(kwargs) > 0:
			raise ValueError('only one of arguments and kwargs should be provided!')
		elif arguments is None:
			arguments = kwargs
		return arguments

	def measure(self, timeout=None, arguments=None, **kwargs):
		"""
		:type timeout: int or float
		:type arguments: NoneType or dict
		:rtype: Measurement
		"""

		kwargs = self.get_arguments(arguments=arguments, **kwargs)

		if self.check_arguments(kwargs=kwargs, method_name='measure'):
			raise TypeError(self.check_arguments(kwargs=kwargs, method_name='measure'))

		key = self.get_key(**kwargs)
		if key in self._measurements:
			return self._measurements[key]
		else:
			start_time = get_now()
			if not timeout:
				try:
					result = self._function(**kwargs)
					timeout_error = False
					other_error = False
					self._num_regular_runs += 1

				except Exception as e:
					result = None
					timeout_error = False
					other_error = True
					self._num_errors += 1

			else:
				def run_function():
					return self._function(**kwargs)

				try:
					result = func_timeout(timeout, run_function)
					timeout_error = False
					other_error = False
					self._num_regular_runs += 1

				except FunctionTimedOut:
					result = None
					timeout_error = True
					other_error = False
					self._num_timeouts += 1

				except Exception as e:
					result = None
					timeout_error = False
					other_error = True
					self._num_errors += 1

			elapsed = get_elapsed(start=start_time, unit=self._unit)
			measurement = Measurement(
				x=kwargs, result=result, elapsed=elapsed, timeout_error=timeout_error, other_error=other_error
			)
			self._measurements[key] = measurement
			self._model = None
			self._normalizer = None
			self._error_model = None
			if self._max_x is None:
				self._max_x = kwargs
			else:
				self._max_x = {key: max(value, kwargs[key]) for key, value in self._max_x.items()}
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
		measurements = sorted(list(self._measurements.values()))
		# set the weights
		min_elapsed = measurements[0].elapsed_time
		for measurement in measurements:
			if min_elapsed > 0:
				measurement._weight = 1 + (measurement.elapsed_time / min_elapsed) ** 0.5
			else:
				measurement._weight = 1
		return measurements

	@property
	def num_measurements(self):
		"""
		:rtype: int
		"""
		return len(self._measurements)

	@property
	def num_errors(self):
		return self._num_errors

	@property
	def num_regular_runs(self):
		return self._num_regular_runs

	@property
	def num_timeouts(self):
		return self._num_timeouts

	@property
	def num_runs(self):
		return self.num_errors + self.num_regular_runs

	def get_x_data(self, x, degree=None):
		"""
		:type x: DataFrame or dict or list
		:type degree: NoneType or int
		:rtype: DataFrame
		"""
		if isinstance(x, dict):
			if all([isinstance(value, (list, tuple)) for value in x.values()]):
				data = DataFrame(x)
			else:
				data = DataFrame.from_records([x])

		elif isinstance(x, list) and all([isinstance(element, dict) for element in x]):
			data = DataFrame.from_records(x)

		elif isinstance(x, list):
			data = DataFrame({'x': x})

		else:
			data = x

		degree = degree or self._polynomial_degree

		if degree > 1:
			polynomial = Polynomial(degree=degree)
			result = polynomial.fit_transform(data=data)

		else:
			result = data.copy()

		if degree not in self._x_data_columns:
			self._x_data_columns[degree] = list(result.columns)
		return result[self._x_data_columns[degree]]

	def get_error_x_data(self, x):
		data = self.get_x_data(x=x, degree=1)

		for column in self._function_arguments:
			data[f'{column}_is_zero'] = data[column] == 0
			data[f'{column}_is_positive'] = 1 * (data[column] > 0)
			data[f'{column}_rounded'] = data[column].round()
			data[f'{column}_is_int'] = data[f'{column}_rounded'] == data[column]

		if 1 not in self._error_x_data_columns:
			self._error_x_data_columns[1] = list(data.columns)
		return data[self._error_x_data_columns[1]]

	@property
	def training_x_y_weights(self):
		"""
		:rtype: tuple[DataFrame, list, list]
		"""
		records = []
		y = []
		weights = []
		for measurement in self.measurements:
			if not measurement.timeout_error and not measurement.other_error:
				records.append(measurement.x)
				y.append(measurement.elapsed_time)
				weights.append(measurement.elapsed_time ** 2)
		x = self.get_x_data(DataFrame.from_records(records))
		return x, y, weights

	@property
	def error_training_x_y_weights(self):
		"""
		:rtype: tuple[DataFrame, list]
		"""
		records = []
		y = []
		weights = []
		error_count = self.num_errors
		other_count = len(self._measurements) - error_count

		for measurement in self._measurements.values():
			if not measurement.timeout_error:
				records.append(measurement.x)
				if measurement.other_error:
					y.append(1)
					weights.append(other_count)
				else:
					y.append(0)
					weights.append(error_count)
		x = self.get_error_x_data(DataFrame.from_records(records))
		return x, y, weights

	@property
	def model(self):
		"""
		:rtype: LinearRegression
		"""
		if self._model is None:
			x, y, weights = self.training_x_y_weights
			self._model = LinearRegression()
			normalized_x = self.normalizer.transform(X=x)
			self._model.fit(normalized_x, y, sample_weight=weights)
		return self._model

	@property
	def normalizer(self):
		"""
		:type Normalizer
		"""
		if self._normalizer is None:
			x, y, weights = self.training_x_y_weights
			self._normalizer = Normalizer()
			self._normalizer.fit(X=x)
		return self._normalizer

	@property
	def error_model(self):
		"""
		:rtype: RandomForestClassifier
		"""
		if self._error_model is None:
			x, y, weights = self.error_training_x_y_weights
			self._error_model = RandomForestClassifier()
			self._error_model.fit(x, y, sample_weight=weights)
		return self._error_model

	def is_in_measurements(self, **kwargs):
		return self.get_key(**kwargs) in self._measurements

	def predict(self, **kwargs):
		error_probability = self.predict_error_probability(**kwargs)
		time_prediction = self.predict_time(**kwargs)
		return {'error_probability': error_probability, 'time_prediction': time_prediction}

	def predict_data(self, data):
		"""
		:type data: DataFrame
		:rtype: DataFrame
		"""
		error_probability = list(1 - self.error_model.predict_proba(self.get_error_x_data(data))[:, 0])
		error = self.error_model.predict(self.get_error_x_data(data))
		normalized = self.normalizer.transform(self.get_x_data(data))
		elapsed = self.model.predict(normalized)
		return DataFrame({'error_probability': error_probability, 'error': error, 'time': elapsed})

	def predict_time(self, **kwargs):
		"""
		:rtype: list or float or int
		"""
		if self.check_arguments(kwargs=kwargs, method_name='predict'):
			raise TypeError(self.check_arguments(kwargs=kwargs, method_name='predict'))

		key = self.get_key(**kwargs)
		if key in self._measurements:
			return self._measurements[key].elapsed_time

		x_data = self.get_x_data(x=kwargs)
		normalized = self.normalizer.transform(x_data)
		predictions = self.model.predict(normalized)
		if x_data.shape[0] > 1:
			return list(predictions)
		else:
			return list(predictions)[0]

	def predict_error(self, **kwargs):
		"""
		:rtype: list or float or int
		"""
		if self.check_arguments(kwargs=kwargs, method_name='predict_error'):
			raise TypeError(self.check_arguments(kwargs=kwargs, method_name='predict_error'))

		key = self.get_key(**kwargs)
		if key in self._measurements:
			return 1 * self._measurements[key].other_error

		x_data = self.get_error_x_data(x=kwargs)
		predictions = self.error_model.predict(x_data)
		if x_data.shape[0] > 1:
			return list(predictions)
		else:
			return list(predictions)[0]

	def predict_error_probability(self, **kwargs):
		"""
		:rtype: list or float or int
		"""
		if self.check_arguments(kwargs=kwargs, method_name='predict_error'):
			raise TypeError(self.check_arguments(kwargs=kwargs, method_name='predict_error'))

		key = self.get_key(**kwargs)
		if key in self._measurements:
			return 1 * self._measurements[key].other_error

		x_data = self.get_error_x_data(x=kwargs)
		predictions = 1 - self.error_model.predict_proba(x_data)[:, 0]
		if x_data.shape[0] > 1:
			return list(predictions)
		else:
			return list(predictions)[0]

	@staticmethod
	def choose_value(values):
		if isinstance(values, list):
			return choice(values)
		elif isinstance(values, tuple) and len(values) == 2:
			if isinstance(values[0], int) and isinstance(values[1], int):
				return randint(values[0], values[1])
			else:
				return min(values) + random() * (max(values) - min(values))
		else:
			return values

	def auto_explore(self, timeout=None, max_order=5):
		timeout = timeout or self._timeout

		if self.num_measurements < 3:
			for i in range(3):
				self.measure(timeout=timeout, arguments={key: i for key in self._function_arguments})

		for main_key in self._function_arguments:
			for order_of_magnitude in range(max_order):
				argument_range = (-1 * 10 ** order_of_magnitude, 10 ** order_of_magnitude)
				print(main_key, argument_range)
				arguments = {key: argument_range if key == main_key else 1 for key in self._function_arguments}
				self.explore(
					timeout=timeout,
					num_rounds=10,
					arguments=arguments
				)

		for order_of_magnitude in range(max_order):
			argument_range = (-1 * 10 ** order_of_magnitude, 10 ** order_of_magnitude)
			print(argument_range)
			self.explore(
				timeout=timeout,
				num_rounds=10,
				arguments={key: argument_range for key in self._function_arguments}
			)

	def explore(self, timeout, num_rounds=1000, arguments=None, random_state=42, **kwargs):
		start_time = get_now()
		arguments = self.get_arguments(arguments=arguments, **kwargs)

		progress_bar = ProgressBar(total=timeout)
		elapsed = get_elapsed(start=start_time, unit=self._unit)
		progress_bar.show(amount=elapsed, total=timeout)

		candidates = create_arguments(num_rounds=num_rounds, arguments=arguments)

		# remove candidates that have already been tried
		tried_candidates = self.data.drop(columns=['time', 'result', 'timeout_error', 'other_error'])
		candidates = concat([candidates, tried_candidates, tried_candidates])
		candidates.drop_duplicates(keep=False, inplace=True)

		candidates['_id_'] = range(candidates.shape[0])
		candidates = candidates.sample(frac=1)

		elapsed = get_elapsed(start=start_time, unit=self._unit)
		num_points = num_errors = num_timeouts = max_elapsed = 0
		num_candidates = candidates.shape[0]

		while timeout > elapsed and num_candidates > 0:
			progress_bar.show(
				amount=elapsed,
				total=timeout,
				text=f'n:{num_points}, e:{num_errors}, o:{num_timeouts}, t:{round(max_elapsed, 2)}, c:{num_candidates}'
			)
			candidates.reset_index(drop=True, inplace=True)
			predictions = self.predict_data(data=candidates.drop(columns='_id_'))

			predictions.reset_index(drop=True, inplace=True)
			predictions['_id_'] = candidates['_id_']

			# if we are out of time, stop
			elapsed = get_elapsed(start=start_time, unit=self._unit)
			remaining_time = timeout - elapsed
			if remaining_time < 0:
				break

			next_candidates = candidates.copy()

			# if we have had too many errors, try to not go after errors
			if self.num_errors > self._num_regular_runs and predictions['error'].sum() < predictions.shape[0]:
				next_candidates = candidates[predictions['error'] == 0]
				predictions = predictions[predictions['error'] == 0]

			# only try the ones that are reasonable timewise
			if (predictions['time'] < remaining_time).sum() > 0:
				next_candidates = next_candidates[predictions['time'] < remaining_time]
				predictions = predictions[predictions['time'] < remaining_time]
			else:
				# there will be no candidate, let's at least try our best with the shortest elapsed time
				next_candidates = next_candidates[predictions['time'] == predictions['time'].min()].head(1)
				predictions = predictions[predictions['time'] == predictions['time'].min()].head(1)

			# stop if there are no candidates
			if next_candidates.shape[0] == 0:
				break

			# choose longest time:
			# best_candidate = next_candidates[predictions['time'] == predictions['time'].max()].head(1)

			# choose randomly
			best_candidate = next_candidates.sample(n=1, random_state=random_state)
			x = best_candidate.to_dict('records')[0]
			best_candidate_id = x['_id_']
			del x['_id_']
			measurement = self.measure(timeout=timeout, **x)

			# remove the candidate that was measured
			candidates = candidates[candidates['_id_'] != best_candidate_id]
			num_candidates = candidates.shape[0]

			if measurement.timeout_error:
				num_timeouts += 1
			elif measurement.other_error:
				num_errors += 1
			else:
				num_points += 1
				max_elapsed = max(max_elapsed, measurement.elapsed_time)
			elapsed = get_elapsed(start=start_time, unit=self._unit)

		progress_bar.show(
			amount=timeout + 1,
			total=timeout,
			text=f'n:{num_points}, e:{num_errors}, o:{num_timeouts}, t:{round(max_elapsed, 2)}, c:{num_candidates}'
		)

	def plot(self, x, ignore_errors=True, marker='.', linestyle='', ms=12):
		data = self.data.copy()
		data['error'] = where(data.timeout_error, 'timeout', where(data.other_error, 'error', 'normal'))
		data.drop(columns=['timeout_error', 'other_error'], inplace=True)

		if ignore_errors:
			fig, ax = plt.subplots()
			ax.plot(data[x], data['time'], marker=marker, linestyle=linestyle, ms=ms)

		else:
			groups = data.groupby('error')
			fig, ax = plt.subplots()
			ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
			for name, group in groups:
				ax.plot(group[x], group['time'], marker=marker, linestyle=linestyle, ms=ms, label=name)
			ax.legend()
		plt.show()
		return fig, ax
