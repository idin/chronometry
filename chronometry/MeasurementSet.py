from .TimeMeasurement import TimeMeasurement
from .measure import measure
from .Timer import Timer
from .get_elapsed import get_elapsed, find_best_unit, convert

from pandas import DataFrame, concat
from numpy import median
from datetime import timedelta


class MeasurementSet:
	def __init__(self):
		self._measurements = {}
		self._start_times = {}
		self._end_times = {}

	def __str__(self):
		return '\n'.join([f'{name} - {measurement}' for name, measurement in self._measurements.items()])

	def __getstate__(self):
		return self._measurements, self._start_times, self._end_times

	def __setstate__(self, state):
		if len(state) == 3:
			self._measurements, self._start_times, self._end_times = state
		else:
			self._measurements = state
			self._start_times = self._end_times = {}

	def __hashkey__(self):
		return (self.__class__.__name__, self._measurements)

	@property
	def measurements(self):
		"""
		:rtype: dict[str, TimeMeasurement]
		"""
		return self._measurements

	@property
	def start_times(self):
		"""
		:rtype: dict[str, float or timedelta]
		"""
		return self._start_times

	@property
	def end_times(self):
		"""
		:rtype: dict[str, float or timedelta]
		"""
		return self._end_times

	def __add__(self, other):
		"""
		:type other: MeasurementSet
		:rtype: MeasurementSet
		"""
		result = self.__class__()
		for name in set(self.measurements.keys()).union(other.measurements.keys()):

			if name in self.measurements and name in other.measurements:
				result.measurements[name] = self.measurements[name] + other.measurements[name]

			elif name in self._measurements:
				result.measurements[name] = self.measurements[name].copy()

			else:
				result.measurements[name] = other.measurements[name].copy()

		return result

	def log(self, name, timer):
		"""
		:type timer: Timer
		"""
		if name in self._start_times:
			self._start_times[name].append(timer.last_start)
			self._end_times[name].append(timer.last_end)
		else:
			self._start_times[name] = [timer.last_start]
			self._end_times[name] = [timer.last_end]

	def add_measurement(self, name, timer):
		"""
		:param str name:
		:type timer: Timer
		"""
		self.log(name=name, timer=timer)

		if name in self.measurements:
			self.measurements[name] += TimeMeasurement(duration=timer.duration, unit=timer.unit)
		else:
			self.measurements[name] = TimeMeasurement(duration=timer.duration, unit=timer.unit)

	@property
	def performance_summary(self):
		"""
		:rtype: DataFrame
		"""
		names = []
		total_durations = []
		units = []
		counts = []
		mean_durations = []
		for name, measurement in self.measurements.items():
			names.append(name)
			total_durations.append(measurement.duration)
			units.append(measurement.unit)
			counts.append(measurement.count)
			mean_durations.append(measurement.mean_duration)

		data = DataFrame({
			'name': names, 'total_duration': total_durations, 'unit': units, 'count': counts
		})

		if all([unit == 'timedelta' for unit in units]):
			median_duration = median(mean_durations)
			best_unit = find_best_unit(delta=median_duration)
			data['unit'] = best_unit
			data['total_duration'] = data['total_duration'].apply(lambda duration: convert(delta=duration, to_unit=best_unit))

		data['mean_duration'] = data['total_duration'] / data['count']
		return data.sort_values('total_duration', ascending=False).reset_index(drop=True)

	@property
	def timestamps(self):
		"""
		:rtype: DataFrame
		"""
		dataframes = []
		for name in self._start_times.keys():
			dataframes.append(DataFrame({
				'name': name,
				'attempt': range(1, len(self._start_times[name]) + 1),
				'start': self._start_times[name],
				'end': self._end_times[name]
			}))
		result = concat(dataframes).reset_index(drop=True)
		all_times = list(result['start']) + list(result['end'])

		if len(all_times) == 0:
			return None
		else:
			min_time = min(all_times)
			differences = [t - min_time for t in all_times]
			best_unit = find_best_unit(delta=median(differences))
			first_time = all_times[0]

			result['standard_unit'] = best_unit
			result['standard_start'] = result['start'].apply(lambda t: get_elapsed(start=first_time, end=t, unit=best_unit))
			result['standard_end'] = result['end'].apply(lambda t: get_elapsed(start=first_time, end=t, unit=best_unit))
			return result

	def measure(self, function, name, unit='ms'):
		"""
		:param callable function: function to be timed
		:type name: str
		:type unit: str
		:rtype: callable
		"""
		return measure(measurement_set=self, function=function, name=name, unit=unit)
