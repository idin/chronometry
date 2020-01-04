
import re
from datetime import date as datetime_date
from .add_time import add_months, add_days
from calendar import day_name, day_abbr
from .YearMonth import YearMonth
from .YearQuarter import YearQuarter
from .to_quarter import get_quarter
from .remove_non_alphanumeric import remove_non_alphanumeric

def to_monthname(date, abr = False):
	try:

		result = date.strftime('%B')
	except:
		print('date:', date, type(date))
		raise ValueError('oops')
	if abr:
		return result[:3]
	else:
		return result


def to_weekday(date):
	return date.weekday()


def to_weekdayname(date, abr=False):
	weekday = to_weekday(date)
	if abr:
		return day_abbr[weekday]
	else:
		return day_name[weekday]


def to_date_part(date, date_part, sep ='-', abr = False):
	date_part = date_part.lower()
	date_part = remove_non_alphanumeric(s = date_part, replace_with = '', keep_underscore = False)
	if date_part == 'day': return date.day
	elif date_part == 'month': return date.month
	elif date_part == 'monthname': return to_monthname(date=date, abr=abr)
	elif date_part == 'quarter': return get_quarter(date=date)
	elif date_part == 'yearmonth': return YearMonth(date=date, sep=sep)
	elif date_part == 'yearmonthabr': return YearMonth(date=date, sep=sep, month_as='abr')
	elif date_part == 'yearmonthname': return YearMonth(date=date, sep=sep, month_as='name')
	elif date_part == 'yearquarter': return YearQuarter(date=date, sep=sep)
	elif date_part == 'monthname': return to_monthname(date = date, abr = abr)
	elif date_part == 'weekday': return to_weekday(date)
	elif date_part == 'weekdayname': return to_weekdayname(date=date, abr=abr)
	else: return date.year


def yearmonth_to_date(x, day = 1):
	if type(x) is int:
		year = x // 100
		month = x % 100
	elif type(x) is str:
		x = str(x)
		year = int(re.findall(pattern = '^\d+', string = x)[0][:4])
		month = int(re.findall(pattern = '\d+$', string = x)[0][-2:])
	elif type(x) is YearMonth:
		year = x.year.value
		month = x.month.value
	else:
		return None

	try:
		result = datetime_date(year = year, month = month, day = day)
	except:
		# There is an exception if the day of the month we're in does not exist in the target month
		# Go to the FIRST of the month AFTER, then go back one day.
		first_day_of_the_month = datetime_date(year = year, month = month, day = 1)
		first_day_of_next_month = add_months(date = first_day_of_the_month, months = 1)
		last_day_of_the_month = add_days(date = first_day_of_next_month, days = -1)
		result = last_day_of_the_month

	return result


