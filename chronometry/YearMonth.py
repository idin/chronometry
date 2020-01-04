import re
from .numbers import FlexibleNumber, NumberPart
from .get_month import get_month_num, get_month_name


class YearMonth(FlexibleNumber):

	def __init__(self, x = None, year=None, month=None, date=None, parts=None, sep='-', month_as='num'):
		# display_month can be 'num' 'abr' or 'ful'
		try:
			year = date.year
			month = date.month
		except:
			pass

		if type(x) is int:
			year = x // 100
			month = x % 100
		else:
			try:
				year = x.year
				month = x.month
			except:
				x = str(x)
				if re.match(pattern='^\d{4}.*\d{2}$', string=x):
					year = int(re.findall(pattern='^\d+', string=x)[0][:4])
					month = int(re.findall(pattern='\d+$', string=x)[0][-2:])
				elif re.match(pattern='^\d{4}\W*[a-zA-Z]+$', string=x):
					year = int(re.match(pattern='^\d+', string=x)[0][:4])
					month = get_month_num(name=re.search(pattern='[a-zA-Z]+$', string=x)[0])

		try:
			year_part = NumberPart(value=year, base=None, digits=4)
			month_part = NumberPart(value=month, base=12, start=1)
		except:
			year_part = parts[0]
			month_part = parts[1]

		if sep is None:
			sep = '-'

		self._month_as = month_as

		super().__init__(parts=[year_part, month_part], labels=['year', 'month'], sep=sep)
		self.adjust()



	@property
	def year(self):
		return self.get('year')

	@property
	def month(self):
		return self.get('month')

	@year.setter
	def year(self, year):
		self.set(value=int(year), label='year')

	@month.setter
	def month(self, month):
		self.set(value=int(month), label='month')

	def get_monthname(self, abr = False):
		return get_month_name(number=self.month.value, abr=abr)

	def get_year_monthname(self, abr = False):
		return str(self.year) + self._sep + self.get_monthname(abr=abr)

	def to_int(self):
		self.adjust()
		return self.year.value*100 + self.month.value

	def to_months(self):
		return self.get_total(label = 'month')

	def to_years(self):
		return self.get_total(label = 'year')

	def __str__(self):
		if self._month_as== 'num':
			return super().__str__()
		elif self._month_as== 'abr':
			return self.get_year_monthname(abr=True)
		else:
			return self.get_year_monthname(abr=False)

	def __repr__(self):
		return f'{str(self.year)}-{str(self.month)}'

	def __hash__(self):
		return hash(self.to_int())

	# object comparisons
	def __eq__(self, other):
		"""
		:type other: YearMonth
		:rtype: bool
		"""
		return self.to_int()==other.to_int()

	def __lt__(self, other):
		return self.to_int() < other.to_int()

	def __le__(self, other):
		return self.to_int() <= other.to_int()

	def __ne__(self, other):
		return self.to_int() != other.to_int()

	def __gt__(self, other):
		return self.to_int() > other.to_int()

	def __ge__(self, other):
		return self.to_int() >= other.to_int()


