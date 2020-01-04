from sys import stdout
from datetime import datetime
from time import sleep
import pandas as pd
from math import floor
_map = map

SECONDS_IN_A_DAY = 60.0 * 60.0 * 24.0


def elapsed(start_time, end_time):
	delta = end_time - start_time
	return delta.days * SECONDS_IN_A_DAY + delta.seconds + delta.microseconds / 1E6


try:
	from slytherin.colour import colour
except ModuleNotFoundError:
	from .colour import colour


class ProgressBar:
	def __init__(
			self, total, bar_length=20, animation='clock',
			full_colour='blue', empty_colour='grey', text_colour='grey', next_line=True,
			echo=1, parent=None, disappear=False, display_wait=0.5,
			bar_characters=None
	):
		"""
		:type total: int or float or NoneType
		:type bar_length: int
		:type bar_full: str
		:type bar_empty: str
		:type animation: str
		:type full_colour: str
		:type empty_colour: str
		:type text_colour: str
		:type next_line: bool
		:type echo: int or bool
		:type parent: ProgressBar or NoneType
		:type display_wait: float or int
		"""
		self._total = total
		self._amount = None
		self._safe_total = total
		self._bar_characters = bar_characters or self.FILLING_SQUARE

		self.amount = 0

		self._bar_length = bar_length
		self._start_time = datetime.now()
		self._animation_counter = -1
		if animation == 'vertical_bar':
			animation_clips = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
			animation_clips = animation_clips + animation_clips[::-1]
		elif animation == 'ball':
			animation_clips = ['⠁', '⠂', '⠄', '⡀', '⢀', '⠠', '⠐', '⠈']
		elif animation == 'clock':
			animation_clips = ['🕐', '🕑', '🕒', '🕓', '🕔', '🕕', '🕖', '🕗', '🕘', '🕙', '🕚', '🕛']
		elif animation == 'big_ball':
			animation_clips = ['●    ', '●    ', ' ●   ', ' ●   ', '  ●  ', '   ● ', '   ● ', '    ●', '    ●']
			animation_clips = animation_clips + animation_clips[::-1]
		elif animation == 'line':
			animation_clips = ['\\', '|', '/', '-']
		else:
			animation_clips = None
		self._animation_clips = animation_clips
		self._full_colour = full_colour
		self._empty_colour = empty_colour
		self._text_colour = text_colour
		self._next_line = next_line
		self._disappear = disappear
		self._completed = False
		self._display_wait = display_wait
		self._display_time = datetime.now()

		if isinstance(echo, self.__class__):
			self._echo = echo._echo
			self._parent = echo
			self._next_line = False
		else:
			self._echo = max(echo, 0)
			self._parent = None

		if parent is not None:
			self._parent = parent
			self._next_line = False

		self._last_text = ''
		self._last_formatted_bar = ''

		if self._parent:
			self._max_lengths = self._parent._max_lengths
		else:
			self._max_lengths = {
				'animation': 0, 'elapsed': 0, 'remaining': 0, 'text': 0, 'percent': 0, 'with_colour': 0
			}

	def __sub__(self, other):
		if isinstance(other, int):
			result = self.__class__(total=self._total, full_colour='green', echo=self._echo - other, parent=self, next_line=False)
			result._amount = self._amount
			return result
		else:
			raise TypeError('subtract only works with integers')

	def set_total(self, total):
		self._total = total
		return self

	@property
	def animation(self):
		if self._animation_clips is None:
			return ''
		else:
			self._animation_counter += 1
			return self._animation_clips[self._animation_counter % len(self._animation_clips)] + ' '

	def complete(self):
		self._completed = True

	@property
	def completed(self):
		return self._completed

	@property
	def total(self):
		return self._safe_total

	@property
	def amount(self):
		return self._amount

	@amount.setter
	def amount(self, amount):
		try:
			self._amount = min(amount, self._total)
		except TypeError:
			self._amount = amount
		self._safe_total = self._total or (amount + 10)

	@property
	def percent(self):
		if self._parent:
			return self._parent.percent

		try:
			return self.amount / self.total * 100
		except ZeroDivisionError:
			if self._parent:
				return self._parent.percent
			else:
				return 0
		except TypeError:
			if self._parent:
				return self._parent.percent
			else:
				return 0

	@property
	def percent_formatted(self):
		formatted_percent = '{0: >#06.2f}'.format(float(self.percent)) + '%'
		return formatted_percent

	@property
	def elapsed_seconds(self):
		if self._parent:
			return self._parent.elapsed_seconds
		delta = datetime.now() - self._start_time
		delta_seconds = delta.seconds + delta.microseconds / 1E6
		return delta_seconds

	@property
	def remaining_seconds(self):
		if self._parent:
			return self._parent.remaining_seconds

		delta_seconds = self.elapsed_seconds

		try:
			speed = self.amount / delta_seconds

			if self.amount >= self.total:
				return 0

			else:
				if speed == 0 or speed is None:
					return None
				else:
					return (self.total - self.amount) / speed

		except ZeroDivisionError:
			return None
		except TypeError:
			return None

	@staticmethod
	def format_time(time):
		"""
		:type time: float or int
		:rtype: str
		"""
		# convert to the right unit
		if time is None:
			return ''
		else:
			if time > 3600:
				unit = 'h'
				time = time / 3600
			elif time > 60:
				unit = 'm'
				time = time / 60
			else:
				unit = 's'
				time = time

			return '{0: >#04.1f}'.format(float(time)) + unit

	FADE_CHARACTERS = [' ', '░', '▒', '▓', '█']
	LEFT_TO_RIGHT_GROWTH = ["　", "▏", "▎", "▍", "▋", "▊", "▉"]
	FOUR_SQUARES = ['　', '▖', '▞', '▛', '▉']
	FILLING_SQUARE = ['◻', '▢', '▧', '▨', '▦', '▩', '◼']

	@property
	def bar(self):
		"""
		:rtype: str
		"""

		if self.total == 0 or self.amount is None or self.total is None:
			full_part_len = 0
			partial_character = ''
		else:
			bar_length = self.amount*self._bar_length / self.total
			full_part_len = floor(bar_length)
			partial_part = bar_length - full_part_len
			if full_part_len < bar_length:
				partial_character = self._bar_characters[round(partial_part*(len(self._bar_characters)-1))]
			else:
				partial_character = ''

		empty_part_len = self._bar_length - full_part_len - len(partial_character)


		full_part_text = self._bar_characters[-1] * full_part_len + partial_character
		empty_part_text = self._bar_characters[0] * empty_part_len

		if self._full_colour is not None:
			character = '▅'  # character = self._bar_empty
			full_part = colour(text=full_part_text, text_colour=self._full_colour)
			empty_part = colour(text=empty_part_text, text_colour=self._empty_colour)
		else:
			full_part = full_part_text
			empty_part = empty_part_text

		return full_part + empty_part

	@staticmethod
	def write(string, flush=True):
		"""
		:type string: str
		"""
		stdout.write('\r'+string)
		if flush:
			stdout.flush()

	@property
	def _animation_string(self):
		animation = self.animation
		self._max_lengths['animation'] = max(self._max_lengths['animation'], len(animation))
		return colour(text=animation.ljust(self._max_lengths['animation']), text_colour=self._empty_colour)

	@property
	def _elapsed_time_string(self):
		elapsed_text = f'e:{self.format_time(self.elapsed_seconds)} '
		self._max_lengths['elapsed'] = max(self._max_lengths['elapsed'], len(elapsed_text))
		elapsed_text = elapsed_text.ljust(self._max_lengths['elapsed'])
		return colour(text=elapsed_text, text_colour=self._full_colour)

	@property
	def _remaining_time_string(self):
		remaining_text = f'r:{self.format_time(self.remaining_seconds)} '
		self._max_lengths['remaining'] = max(self._max_lengths['remaining'], len(remaining_text))
		remaining_text = remaining_text.ljust(self._max_lengths['remaining'])
		return colour(text=remaining_text, text_colour=self._empty_colour)

	@property
	def _percent_string(self):
		percent_text = f' {self.percent_formatted}'
		self._max_lengths['percent'] = max(self._max_lengths['percent'], len(percent_text))
		percent_text = percent_text.rjust(self._max_lengths['percent'])
		return colour(text=percent_text, text_colour=self._full_colour)

	def _get_text_string(self, text):
		text_text = f' {text}'
		self._max_lengths['text'] = max(self._max_lengths['text'], len(text_text))
		text_text = text_text.ljust(self._max_lengths['text'])
		return colour(text=text_text, text_colour=self._text_colour)

	@property
	def _main_bar(self):
		if self._parent:
			return self._parent
		else:
			return self

	def show(self, amount, extra_amount=0, percent=True, bar=True, time=True, text=''):
		"""
		:type amount: int or float or NoneType
		:type extra_amount: int or float or NoneType
		:type percent: bool
		:type bar: bool
		:type time: bool
		:type text: str
		"""
		if amount is not None:
			self.amount = amount + extra_amount
		else:
			self.amount = amount



		time_now = datetime.now()
		if elapsed(start_time=self._display_time, end_time=time_now) > self._display_wait or self.amount >= self.total:
			self._display_time = time_now

			string = ''
			if self._echo > 0:
				try:
					string += self._animation_string

					if time:
						string += self._main_bar._elapsed_time_string
						string += self._main_bar._remaining_time_string

					if bar:
						self._last_formatted_bar = self.bar
						string += self._last_formatted_bar

					if percent:
						string += self._main_bar._percent_string

					if self._parent is None:
						self._last_text = text
					else:
						if len(self._parent._last_text) > 0:
							self._last_text = self._parent._last_text + ' / ' + text
						else:
							self._last_text = text

					string += self._get_text_string(text=self._last_text)

					self._max_lengths['with_colour'] = max(self._max_lengths['with_colour'], len(string))
					string = string.ljust(self._max_lengths['with_colour'])


				except Exception as e:
					self.write(string=f'progress bar error: {e}')
					raise e

			if self.amount >= self.total:
				self.complete()

			if self._echo:
				if self._completed:
					if self._disappear:
						self.write(string=' ' * self._max_lengths['with_colour'] + ' ')
						#self.write(string=' '*15)
						#stdout.write("\033[F")  # back to previous line
						#stdout.write("\033[K")  # clear line
						#print('', end="")
					else:
						self.write(string=string)
						#stdout.flush()

					if self._next_line:
						print('')
				else:
					self.write(string=string, flush=True)

	@classmethod
	def map(
			cls, function, iterable, percent=True, bar=True, time=True, text='', echo=1,
			next_line=True, iterable_text=None,
			**kwargs
	):
		echo = max(0, echo)

		def _func(x, _progress, _progress_bar, _text=''):
			if progress['amount'] == 0:
				progress_bar.show(
					amount=progress['amount'], percent=percent, bar=bar, time=time, text=text + _text
				)

			function_result = function(x)

			_progress['amount'] += 1
			if elapsed(start_time=progress['update_time'], end_time=datetime.now()) > 0.5 or _progress['amount'] >= total:
				_progress_bar.show(amount=_progress['amount'], percent=percent, bar=bar, time=time, text=text + _text)
				_progress['update_time'] = datetime.now()

			return function_result

		total = len(iterable)

		progress = {'amount': 0, 'update_time': datetime.now()}
		progress_bar = cls(total=total, next_line=next_line, **kwargs)

		if echo:
			if iterable_text is None:
				result = _map(
					lambda x: _func(x=x, _progress=progress, _progress_bar=progress_bar),
					iterable
				)
			else:
				result = _map(
					lambda x: _func(x=x[0], _text=x[1], _progress=progress, _progress_bar=progress_bar),
					zip(iterable, iterable_text)
				)

			return result
		else:
			return _map(function, iterable)

	@classmethod
	def apply(
			cls, function, data=None, series=None, percent=True, bar=True, time=True, text='',
			axis=1, echo=1, next_line=True, **kwargs
	):
		"""
		:type function: function
		:type data: pd.DataFrame
		:type series: pd.Series
		:type percent: bool
		:type bar: bool
		:type time: bool
		:type text: str
		:type axis: int
		:type echo: bool or int
		:type next_line: bool
		:rtype: pd.DataFrame or pd.Series
		"""
		echo = max(0, echo)
		# either data or series should be provided:
		if data is None and series is None:
			raise ValueError('either data or series should be provided')

		if data is not None and series is not None:
			raise ValueError('both data and series cannot be provided')

		def _func(x, _progress, _progress_bar):
			function_result = function(x)
			_progress['amount'] += 1

			if elapsed(start_time=_progress['update_time'], end_time=datetime.now()) > 0.5 or _progress['amount'] >= total:
				_progress_bar.show(amount=_progress['amount'], percent=percent, bar=bar, time=time, text=text)
				_progress['update_time'] = datetime.now()

			return function_result

		if data is not None:
			if axis == 1:
				total = data.shape[0]
			else:
				total = data.shape[1]
		else:
			total = series.shape[0]

		if total == 0:
			return None

		progress = {'amount': 0, 'update_time': datetime.now()}
		progress_bar = cls(total=total, next_line=next_line, **kwargs)
		if echo:
			progress_bar.show(amount=0, percent=percent, bar=bar, time=time, text=text)
			if data is not None:
				result = data.apply(func=lambda x: _func(x=x, _progress=progress, _progress_bar=progress_bar), axis=axis)
			else:
				result = series.apply(func=lambda x: _func(x=x, _progress=progress, _progress_bar=progress_bar))

		else:
			if data is not None:
				result = data.apply(func=function, axis=axis)
			else:
				result = series.apply(func=function)

		return result

	def __gt__(self, other):
		if isinstance(other, self.__class__):
			return self._echo > other._echo
		else:
			try:
				return self._echo > other
			except TypeError:
				return True

	def __lt__(self, other):
		if isinstance(other, self.__class__):
			return self._echo < other._echo
		else:
			try:
				return self._echo < other
			except TypeError:
				return False

	def __ge__(self, other):
		return not (self < other)

	def __le__(self, other):
		return not (self > other)

	def __eq__(self, other):
		if isinstance(other, self.__class__):
			return self._echo == other._echo
		else:
			return False

	def __ne__(self, other):
		return not (self == other)

	@classmethod
	def test(cls, time=1, n=10000, bar_characters=None):
		bar = cls(total=n, display_wait=0.1, bar_characters=bar_characters)
		for i in range(n):
			bar.show(amount=i, text='testing ...')
			sleep(time / n / 2)
		bar.show(amount=n, text='test done!')
