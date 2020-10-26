from .get_time import get_now
from .get_elapsed import get_elapsed
from datetime import timedelta


class Timer:
	def __init__(self, unit='ms', start_now=True):
		self._unit = unit
		self._start_time = None
		self._duration = None
		self._end_time = None

		self._last_start = None
		self._last_end = None

		if start_now:
			self.start()

	def __hashkey__(self):
		return (self.__class__.__name__, self.unit, self._start_time, self._duration, self._end_time)

	def start(self):
		"""
		:rtype: NoneType
		"""
		if self._start_time is not None:
			raise RuntimeError('Timer has already started!')
		self._duration = None
		self._last_end = self._end_time
		self._end_time = None
		self._start_time = get_now()

	def reset_timer(self):
		"""
		:rtype: NoneType
		"""
		self._last_start = self._start_time
		self._last_end = self._end_time
		self._start_time = None
		self._end_time = None

	@property
	def last_start(self):
		return self._last_start

	@property
	def last_end(self):
		return self._last_end

	def get_elapsed(self, end_time=None):
		"""
		:rtype: float or timedelta
		"""
		if end_time is None:
			end_time = get_now()
		now = end_time
		if self._start_time is not None:
			if self._duration is None:
				return get_elapsed(start=self._start_time, end=now, unit=self._unit)
			else:
				return get_elapsed(start=self._start_time, end=now, unit=self._unit) + self.duration
		else:
			raise RuntimeError('Timer is not activated yet!')

	def stop(self):
		"""
		:rtype: NoneType
		"""
		self._end_time = get_now()
		self._duration = self.get_elapsed(end_time=self._end_time)
		self.reset_timer()

	@property
	def duration(self):
		"""
		:rtype: float or timedelta
		"""
		return self._duration

	@property
	def unit(self):
		return self._unit
