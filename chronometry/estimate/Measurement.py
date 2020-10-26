class Measurement:
	def __init__(self, x, result, elapsed, timeout_error, other_error):
		"""
		:type x: dict
		:type result: object
		:type elapsed: float
		:type timeout_error: bool
		"""
		if not isinstance(x, dict):
			raise TypeError('x should be a dictionary!')
		self._x = x
		self._result = result
		self._elapsed_time = elapsed
		self._timeout_error = timeout_error
		self._other_error = other_error
		self._weight = None

	@property
	def weight(self):
		return self._weight

	@property
	def timeout_error(self):
		return self._timeout_error

	@property
	def other_error(self):
		return self._other_error

	def __lt__(self, other):
		return self.elapsed_time < other.elapsed_time

	def __le__(self, other):
		return self.elapsed_time <= other.elapsed_time

	def __eq__(self, other):
		return self.elapsed_time == other.elapsed_time

	def __gt__(self, other):
		return self.elapsed_time > other.elapsed_time

	def __ge__(self, other):
		return self.elapsed_time >= other.elapsed_time

	def __ne__(self, other):
		return self.elapsed_time != other.elapsed_time

	@property
	def x(self):
		"""
		:rtype: dict
		"""
		return self._x

	@property
	def elapsed_time(self):
		return self._elapsed_time

	@property
	def dictionary(self):
		if isinstance(self._result, dict) and 'elapsed' not in self._result and 'x' not in self._result:
			return dict(
				x=self._x, time=self.elapsed_time, timeout_error=self._timeout_error, other_error=self._other_error,
				**self._result
			)
		else:
			return {
				**self._x, 'time': self.elapsed_time, 'result': self._result,
				'timeout_error': self._timeout_error, 'other_error': self._other_error
			}
