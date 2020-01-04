from pandas import DataFrame
from .ProgressBar import ProgressBar
_map = map


def map(function, iterable, percent=True, bar=True, time=True, text=''):
	"""
	:type function: function
	:type iterable: iterable
	:type progress_step: int
	:type percent: bool
	:type bar: bool
	:type time: bool
	:type text: str
	:rtype: iterable
	"""
	def _func(x, _progress, _progress_bar):
		result = function(x)
		_progress['amount'] += 1
		if _progress['max_step']>1:
			if _progress['amount'] % _progress['step']==0 or _progress['amount']>=total:
				_progress_bar.show(amount=_progress['amount'], percent=percent, bar=bar, time=time, text=text)


		return result

	total = len(iterable)
	progress = {'amount':0, 'step':1}
	progress_bar = ProgressBar(total=total)

	return _map(lambda x: _func(x=x, _progress=progress, _progress_bar=progress_bar), iterable)


def apply(function, data, progress_step=None, percent=True, bar=True, time=True, text=''):
	"""
	:type function: function
	:type data: DataFrame
	:type progress_step: int
	:type percent: bool
	:type bar: bool
	:type time: bool
	:type text: str
	:rtype: DataFrame
	"""
	def _func(x, _progress, _progress_bar):
		"""
		:type progress: dict[str, int]
		:type progress_bar: ProgressBar
		"""
		result = function(x)
		_progress['amount'] += 1
		if _progress['amount'] % _progress['step']==0 or _progress['amount']>=total:
			_progress_bar.show(amount=_progress['amount'], percent=percent, bar=bar, time=time, text=text)
		return result
	total = data.shape[0]
	if progress_step is None:
		progress_step = max(round(total/10),1) # we don't want progress step to be 0
	progress = {'amount': 0, 'step': 1, 'max_step': progress_step}
	progress_bar = ProgressBar(total=total)

	return data.apply(func=lambda x: _func(x=x, _progress=progress, _progress_bar=progress_bar))
