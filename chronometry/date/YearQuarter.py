import re
from ..numbers import FlexibleNumber, NumberPart
from .get_quarter import get_quarter
from .get_first_day_of_month import get_first_day_of_month
from .get_first_day_of_month import get_last_day_of_month
import datetime


class YearQuarter(FlexibleNumber):

	def __init__(self, x=None, year=None, quarter=None, date=None, parts=None, sep='-'):
		if type(x) is int:
			year = x // 10
			quarter = x % 10
		elif type(x) is str:
			x = str(x)
			year = int(re.findall(pattern='^\d+', string=x)[0][:4])
			quarter = int(re.findall(pattern='\d+$', string=x)[0][-1:])

		try:
			year = date.year
			quarter = get_quarter(date=date)
		except:
			pass

		try:
			year_part = NumberPart(value=year, base=None, digits=4)
			quarter_part = NumberPart(value=quarter, base=4, start=1)
		except:
			year_part = parts[0]
			quarter_part = parts[1]

		if sep is None:
			sep = '-'

		super().__init__(parts=[year_part, quarter_part], labels=['year', 'quarter'], sep=sep)
		self.adjust()

	@property
	def year(self):
		return self.get_value('year')

	@property
	def quarter(self):
		return self.get_value('quarter')

	def __str__(self):
		return f'{self.year}Q{self.quarter}'

	def __repr__(self):
		return self.__str__()

	@year.setter
	def year(self, year):
		self.set(value=int(year), label='year')

	@quarter.setter
	def quarter(self, quarter):
		self.set(value=int(quarter), label='quarter')

	def to_int(self):
		return self.year*100 + self.quarter

	def to_quarters(self):
		return self.get_total(label='quarter')

	def to_years(self):
		return self.get_total(label='year')

	@property
	def first_month(self):
		return self.last_month - 2

	@property
	def last_month(self):
		return self.quarter * 3

	@property
	def first_day(self):
		"""
		:rtype: datetime.date
		"""
		return get_first_day_of_month(year=self.year, month=self.first_month)

	@property
	def last_day(self):
		"""
		:rtype: datetime.date
		"""
		return get_last_day_of_month(year=self.year, month=self.last_month)
