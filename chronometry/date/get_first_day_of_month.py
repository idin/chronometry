import datetime


def get_first_day_of_month(year=None, month=None, date=None):
	"""
	:rtype: datetime.date
	"""
	if date is None and year is not None and month is not None:
		return datetime.date(year, month, 1)
	elif date is not None and year is None and month is None:
		return date.replace(day=1)
	else:
		raise RuntimeError('either year and month should be provided or date and not all!')


def get_last_day_of_month(year=None, month=None, date=None):
	"""
	:rtype: datetime.date
	"""
	if date is None:
		date = datetime.date(year, month, 28)
	else:
		date = date.replace(day=28)

	next_month = date + datetime.timedelta(days=4)
	return next_month - datetime.timedelta(days=next_month.day)
