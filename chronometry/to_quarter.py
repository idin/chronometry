# to avoid circular dependency we have to define it separately
def get_quarter(date):
	return date.month//4 + 1

