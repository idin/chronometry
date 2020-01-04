from .get_time import get_now
from datetime import datetime
SECONDS_IN_A_DAY = 60.0 * 60.0 * 24.0


def _get_elapsed(start, end=None):
	"""
	:type start: datetime or date
	:type end: datetime or date or NoneType
	:rtype timedelta
	"""
	end = end or get_now()
	if not isinstance(end, datetime) and isinstance(start, datetime):
		start = start.date()
	elif isinstance(end, datetime) and not isinstance(start, datetime):
		end = end.date()
	return end - start


def convert(delta, to_unit='timedelta'):
	u = to_unit[0].lower()
	if u == 't':
		return delta

	# microsecond
	elif to_unit[:2] == 'us' or to_unit[:3] == 'mic':
		return delta.days*SECONDS_IN_A_DAY*1e6 + delta.seconds*1e6 + delta.microseconds

	# millisecond
	elif to_unit[:2] == 'ms' or to_unit[:3] == 'mil':
		return delta.days*SECONDS_IN_A_DAY*1e3 + delta.seconds*1e3 + delta.microseconds / 1E3

	# second
	elif u == 's':
		return delta.days*SECONDS_IN_A_DAY + delta.seconds + delta.microseconds / 1E6

	# minute
	elif to_unit[:3] == 'min':
		return (delta.days*SECONDS_IN_A_DAY + delta.seconds + delta.microseconds / 1E6) / 60

	# hour
	elif u == 'h':
		return (delta.days*SECONDS_IN_A_DAY + delta.seconds + delta.microseconds / 1E6) / 3600

	# day
	elif u == 'd':
		return delta.days + delta.seconds/SECONDS_IN_A_DAY + delta.microseconds / 1E6 / SECONDS_IN_A_DAY

	else:
		raise ValueError(f'Cannot convert time delta to {to_unit}')


def get_elapsed_months(start, end=None):
	"""
	:type start: datetime
	:type end: datetime or NoneType
	:rtype: float
	"""
	end = end or get_now()
	return (end.year - start.year) * 12 + end.month - start.year + (end.day - start.day)/31


def get_elapsed_years(start, end=None):
	"""
	:type start: datetime
	:type end: datetime or NoneType
	:rtype: float
	"""
	return get_elapsed_months(start=start, end=end) / 12


def get_elapsed(start, end=None, unit='timedelta'):
	"""
	:param datetime or date start: start time
	:param datetime or or date or NoneType end: end time, the current time is used if not provided (None entered)
	:param str unit: 'timedelta', 'seconds', 'ms' (milliseconds), 'minutes', 'hours', 'days', 'months', or 'years'
	:rtype: float
	"""
	unit = unit.lower()
	u = unit[0]
	if u in ['t', 's', 'd', 'h'] or unit[:2] in ['ms', 'us'] or unit[:3] in ['min', 'mic', 'mil']:
		return convert(delta=_get_elapsed(start=start, end=end), to_unit=unit)
	elif unit[:2] == 'mo':  	# months
		return get_elapsed_months(start=start, end=end)
	elif u == 'y':
		return get_elapsed_years(start=start, end=end)
	else:
		raise ValueError(f'unit:{unit} is not recognizable.')


def find_best_unit(delta):
	seconds = convert(delta=delta, to_unit='s')
	if seconds < 1e-3:
		return 'microsecond'
	elif seconds < 1:
		return 'millisecond'
	elif seconds < 60:
		return 'second'
	elif seconds < 3600:
		return 'minute'
	elif seconds < 24 * 3600:
		return 'hour'
	else:
		return 'day'
