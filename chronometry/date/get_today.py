from datetime import date


def get_today():
	"""
	:rtype: date
	"""
	return date.today()


def get_today_str():
	"""
	:rtype:  str
	"""
	return str(get_today())


def get_date(year=None, month=None, day=None):
	year = year or get_today().year
	month = month or get_today().month
	day = day or get_today().day
	return date(year, month, day)
