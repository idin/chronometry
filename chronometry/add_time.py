import datetime as dt
from .remove_non_alphanumeric import remove_non_alphanumeric

# DAY
def add_days(date, days):
	if date is None:
		return None
	else:
		return date + dt.timedelta(days=days)


# WEEK
def add_weeks(date, weeks):
	if date is None:
		return None
	else:
		return add_days(date=date, days=weeks*7)


# MONTH
def add_months(date, months, round='down'):
	if date is None:
		return None
	else:
		targetmonth = months + date.month
		try:
			date = date.replace(year=date.year+int(targetmonth/12), month=(targetmonth-1)%12+1 )
		except:
			# There is an exception if the day of the month we're in does not exist in the target month
			# Go to the FIRST of the month AFTER
			date = date.replace(year=date.year + int((targetmonth + 1) / 12), month=targetmonth % 12 + 1, day=1)
			if round == 'down':
				# then go back one day.
				date += dt.timedelta(days=-1)
		return date


def add_quarters(date, quarters, round = 'down'):
	if date is None:
		return None
	else:
		return add_months(date=date, months=quarters*3, round=round)


# YEAR
def add_years(date, years, round='down'):
	"""Return a date that's `years` years after the date (or datetime)
	object `d`. Return the same calendar date (month and day) in the
	destination year, if it exists, otherwise use the following day
	(thus changing February 29 to Feb 28).

	"""
	if date is None:
		return None
	else:
		try:
			return date.replace(year=date.year + years)
		except ValueError:
			if round == 'down':
				return date.replace(year=date.year + years, day=date.day - 1)
			else:
				return date.replace(year=date.year + years, month = date.month + 1, day = 1)


# TIME
def add_time(t, amount, unit ='day', round='down'):
	unit = remove_non_alphanumeric(s=unit.lower(), replace_with='')
	if unit == 'day':
		return add_days(date=t, days=amount)
	if unit == 'week':
		return add_weeks(date=t, weeks=amount)
	elif unit == 'month':
		return add_months(date=t, months=amount, round=round)
	elif unit == 'year':
		return add_years(date=t, years=amount, round=round)
	elif unit == 'quarter':
		return add_quarters(date=t, quarters=amount, round=round)
	else:
		return t

