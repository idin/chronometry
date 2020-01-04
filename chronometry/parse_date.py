import datetime as dt
from datetime import date


FORMATS = (
	'%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d',
	'%Y-%m', '%Y/%m', '%Y.%m', '%Y',
	'%y-%m-%d', '%y/%m/%d', '%y.%m.%d',
	'%b %d, %Y', '%b %d, %Y', '%B %d, %Y', '%B %d %Y', '%m/%d/%Y', '%m/%d/%y', '%b %Y',' %B%Y', '%b %d,%Y'
)


def parse_date(string, format=None):
	"""
	:type string: str
	:type format: NoneType or str
	:rtype: NoneType or date
	"""
	if format is None:
		formats = FORMATS
	else:
		formats = [format]

	for f in formats:
		try:
			return dt.datetime.strptime(string, f).date()
		except ValueError as e:
			pass
	return None
