SECONDS_IN_A_DAY = 60.0 * 60.0 * 24.0


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
