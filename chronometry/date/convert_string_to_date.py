from datetime import date


def convert_string_to_date(s):
	"""
	converts a date formated as YYYY-MM-DD to date
	:type s: str
	:rtype: date
	"""
	year = int(s[:4])
	month = int(s[5:7])
	day = int(s[8:10])
	return date(year, month, day)
