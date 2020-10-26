from datetime import date


def get_number_of_days(year, month=None):
	"""
	returns the number of days in a month or a year
	:param int year: the year at question
	:param int or NoneType month: if None, the function returns the number of days in a year
	:rtype: int
	"""
	if month:
		if month == 12:
			return (date(year + 1, 1, 1) - date(year, 12, 1)).days
		else:
			return (date(year, month + 1, 1) - date(year, month, 1)).days
	else:
		return (date(year + 1, 1, 1) - date(year, 1, 1)).days