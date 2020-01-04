class TimeMeasurement:
	def __init__(self, duration, unit, count=1):
		self._duration = duration
		self._unit = unit
		self._count = count

	def __hashkey__(self):
		return (self.__class__.__name__, self.unit, self.duration, self.count)

	def copy(self):
		return self.__class__(duration=self._duration, unit=self._unit, count=self._count)

	@property
	def unit(self):
		return self._unit

	@property
	def count(self):
		return self._count

	@property
	def duration(self):
		return self._duration

	@property
	def mean_duration(self):
		return self.duration / self.count

	def __add__(self, other):
		if self.unit != other.unit:
			raise TypeError('cannot add two measurements of different type!')
		return self.__class__(
			duration=self.duration + other.duration,
			unit=self.unit, count=self.count + other.count
		)
