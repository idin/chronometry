from colouration import Colour


def colour(text, text_colour='blue', background_colour=None):
	return Colour(name=text_colour).colourize(string=text, background=background_colour)
