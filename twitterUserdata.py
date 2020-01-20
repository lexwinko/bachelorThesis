from subprocess import call
import sys


def getUserData(nameFile):

	usernames = []
	with open(nameFile, "r", encoding='utf-8') as f:
		for x in f:
			print(x)
			usernames.append(x.split()[0])
	print(usernames)

	for user in usernames:
		call(["twitterscraper", "'from:"+user+"'", "--lang", "en", "--csv", "-o", user+".csv", "-l", "150"])


if __name__ == "__main__":
	names = sys.argv[1]
	print("Scraping userdata from "+names)
	getUserData(names)
	print("Done")