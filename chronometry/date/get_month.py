def get_month_name(number, abr=False):
	months_map = {
		1: 'January', 2: 'February', 3: 'March',
		4: 'April', 5: 'May', 6: 'June',
		7: 'July', 8: 'August', 9: 'September',
		10: 'October', 11: 'November', 12: 'December'
	}
	result = months_map[number]
	if abr:
		return result[:3]
	else:
		return result


def get_month_num(name):
	months_map = {
		'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
		'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
	}
	return months_map[str(name).lower()[:3]]


def get_months_list(abr=False):
	return [get_month_name(abr=abr, number=i) for i in range(1, 13)]
