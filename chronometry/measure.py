from .Timer import Timer

import functools


def measure(function, measurement_set, name, unit='ms'):
	"""
	:param callable function: function to be timed
	:type name: str
	:type unit: str
	:param .MeasurementSet_class.MeasurementSet measurement_set:
	:rtype: callable
	"""

	@functools.wraps(function)
	def wrapper(*args, **kwargs):
		timer = Timer(start_now=True, unit=unit)
		result = function(*args, **kwargs)
		timer.stop()
		measurement_set.add_measurement(name=name, timer=timer)
		return result
	wrapper.measurement_set = measurement_set

	return wrapper
