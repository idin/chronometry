from datetime import datetime
import time as _time


def get_now():
	"""
	:rtype: datetime
	"""
	return datetime.now()


def sleep(seconds):
	_time.sleep(seconds)


get_time = get_now
