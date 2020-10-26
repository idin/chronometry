from pandas import DataFrame
from slytherin.collections import Series


def create_arguments(num_rounds, arguments):
	"""
	:type num_rounds: int
	:type arguments: dict[str, list or tuple or int or float]
	"""
	result = DataFrame({'_id_': [1]})
	for key, value in arguments.items():
		if isinstance(value, tuple) and len(value) == 2:
			values = Series(start=value[0], end=value[1], count=num_rounds, include_start=True, include_end=True).list

		elif isinstance(value, tuple) and len(value) == 3:
			values = Series(start=value[0], end=value[1], step=value[2], include_start=True, include_end=True).list

		elif isinstance(value, list):
			values = value

		elif not isinstance(value, (list, tuple)):
			values = [0.9 * value, value, 1.1 * value]

		data = DataFrame({key: values})
		data['_id_'] = 1
		result = result.merge(data, on='_id_')

	return result.drop(columns='_id_').reset_index(drop=True)