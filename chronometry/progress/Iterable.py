from .ProgressBar import ProgressBar


class Iterable:
	def __init__(self, inner, progress_bar=None, text='', echo=1, echo_items=False):
		"""
		:type inner: iterable
		:type progress_bar: ProgressBar
		:type text: str
		:type echo: bool or int or ProgressBar
		:type echo_items: bool or list[str]
		"""
		self._idx = 0
		try:
			self._total = len(inner)
			self._inner = inner
		except TypeError:
			self._inner = list(inner)
			self._total = len(self._inner)
		self._text = text
		self._echo_items = echo_items
		self._echo = echo

		if progress_bar is None:
			self._progress_bar = ProgressBar(total=self._total, echo=echo)
		else:
			self._progress_bar = progress_bar
			self._progress_bar._total = len(self._inner)

	def __iter__(self):
		return self

	def __next__(self):
		self._idx += 1
		progress_amount = self._idx - 1
		try:
			result = self._inner[self._idx - 1]
			if self._echo:
				text = f'{self._text} {result}' if self._echo_items else self._text
				self._progress_bar.show(amount=progress_amount, text=text)
			return result
		except IndexError:
			self._idx = 0
			self._progress_bar.show(amount=progress_amount + 1, text=self._text)
			raise StopIteration


def iterate(iterable, progress_bar=None, text='', echo=1, echo_items=False):
	"""
	:type iterable: Iterable
	:type progress_bar: ProgressBar
	:type text: str
	:type echo: bool or int or ProgressBar
	:type echo_items: bool or list[str]
	:rtype: Iterable
	"""
	return Iterable(inner=list(iterable), progress_bar=progress_bar, text=text, echo=echo, echo_items=echo_items)
