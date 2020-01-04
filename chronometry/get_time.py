from datetime import date, datetime
import time as _time


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


def get_now():
	"""
	:rtype: datetime
	"""
	return datetime.now()


def sleep(seconds):
	_time.sleep(seconds)


get_time = get_now


get_date = get_today
